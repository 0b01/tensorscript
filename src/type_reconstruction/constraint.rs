use std::collections::HashSet;

use typed_ast::Type;
use typed_ast::type_env::{ModName, TypeEnv, AliasType};
use typed_ast::typed_term::*;
use typed_ast::typed_term::Ty;

#[derive(Debug, Hash, Eq, PartialEq, Clone)]
pub struct Equals(pub Type, pub Type);

#[derive(Debug, Clone)]
pub struct Constraints(pub HashSet<Equals>);

impl Constraints {
    pub fn new() -> Self {
        Constraints(HashSet::new())
    }

    fn add(&mut self, a: Type, b: Type) {
        self.0.insert(Equals(a, b));
    }

    pub fn collect(&mut self, typed_term: &TyTerm, tenv: &mut TypeEnv) {
        use self::TyTerm::*;
        let module = tenv.module();
        // println!("{}", typed_term);
        match typed_term {
            &TyProgram(ref decls) => decls
                .iter()
                .map(|decl| collect_decl(self, &decl, tenv))
                .collect(),
            &TyInteger(_, _) => (),
            &TyFloat(_, _) => (),
            &TyList(ref terms) => terms.iter().map(|t| self.collect(&t, tenv)).collect(),
            &TyIdent(ref t, ref name) => self.add(
                t.clone(),
                tenv.resolve_type(&module, &name)
                    .expect(&format!("{:#?}", tenv))
                    .clone(),
            ),
            // &TyFieldAccess(TyFieldAccess),
            &TyFnApp(ref fn_app) => collect_fn_app(self, &fn_app, tenv),
            &TyBlock { ref stmts, ref ret } => {
                tenv.push_scope_collection(&module);
                self.collect(&stmts, tenv);
                self.collect(&ret, tenv);
                tenv.pop_scope(&module);
            }
            &TyExpr { ref items, ref ty } => {
                self.collect(&items, tenv);
                self.add(ty.clone(), items.ty());
            }
            &TyStmt { ref items } => self.collect(&items, tenv),
            &TyViewFn(ref view_fn) => {
                self.collect(&view_fn.arg.arg, tenv);
            }
            _ => {
                panic!("{:#?}", typed_term);
            }
        }
    }
}

fn collect_decl(cs: &mut Constraints, decl: &TyDecl, tenv: &mut TypeEnv) {
    use self::TyDecl::*;
    match decl {
        TyGraphDecl(d) => collect_graph_decl(cs, d, tenv),
        TyNodeDecl(d) => collect_node_decl(cs, d, tenv),
        TyUseStmt(d) => collect_use_stmt(cs, d, tenv),
        TyWeightsDecl(d) => collect_weights_decl(cs, d, tenv),
    }
    tenv.set_module(ModName::Global);
}

fn collect_graph_decl(cs: &mut Constraints, decl: &TyGraphDecl, tenv: &mut TypeEnv) {
    // type decl should be the same
    tenv.set_module(ModName::Named(decl.name.clone()));
    let graph_ty_sig = tenv.resolve_type(&ModName::Global, &AliasType::Variable(decl.name.clone()))
        .unwrap()
        .clone();

    cs.add(Type::Module(decl.name.to_owned(), Some(box decl.ty_sig.clone())), graph_ty_sig);
    // collect fn_decls
    decl.fns
        .iter()
        .map(|f| collect_fn_decl(cs, &f, tenv))
        .collect::<Vec<_>>();
}

fn collect_fn_decl(cs: &mut Constraints, decl: &TyFnDecl, tenv: &mut TypeEnv) {
    let module = tenv.module();
    tenv.push_scope_collection(&module);

    cs.collect(&decl.func_block, tenv);
    cs.add(decl.return_ty.clone(), decl.func_block.ty());
    // cs.add(
    //     decl.fn_ty.clone(),
    //     Type::FUN(
    //         Box::new(decl.param_ty.clone()),
    //         Box::new(decl.return_ty.clone()),
    //     ),
    // );

    // if decl.name == "example" {
    //     panic!("{:#?}", decl.return_ty);
    // }
    tenv.pop_scope(&module);
    // ...
}

fn collect_node_decl(cs: &mut Constraints, decl: &TyNodeDecl, tenv: &mut TypeEnv) {
    tenv.set_module(ModName::Named(decl.name.clone()));
    // type decl should be the same
    let graph_ty_sig = tenv.resolve_type(&ModName::Global, &AliasType::Variable(decl.name.clone()))
        .unwrap()
        .clone();
    cs.add(Type::Module(decl.name.to_owned(), Some(box decl.ty_sig.clone())), graph_ty_sig);
}

fn collect_weights_decl(cs: &mut Constraints, decl: &TyWeightsDecl, tenv: &mut TypeEnv) {
    tenv.set_module(ModName::Named(decl.name.clone()));
    // type decl should be the same
    let graph_ty_sig = tenv.resolve_type(&ModName::Global, &AliasType::Variable(decl.name.clone()))
        .unwrap()
        .clone();
    cs.add(Type::Module(decl.name.to_owned(), Some(box decl.ty_sig.clone())), graph_ty_sig);

    // collect weight assigns
    decl.inits
        .iter()
        .map(|w| collect_weights_assign(cs, &w, tenv))
        .collect::<Vec<_>>();
}

fn collect_use_stmt(_cs: &mut Constraints, _decl: &TyUseStmt, _tenv: &TypeEnv) {
    ()
}

fn collect_weights_assign(cs: &mut Constraints, w_a: &TyWeightsAssign, tenv: &mut TypeEnv) {
    let mod_name = &w_a.mod_name;
    // convert into a fn_app and collect on `self.new` method
    collect_fn_app(cs, &TyFnApp {
        mod_name: Some(mod_name.to_string()),
        orig_name: "".to_owned(), // don't need to supply one because it won't be used below
        name: AliasType::Function("new".to_owned()),
        arg_ty: w_a.arg_ty.clone(),
        ret_ty: tenv.fresh_var(),
        args: w_a.fn_args.clone(),
    }, tenv);
    ()
}

fn collect_fn_app(cs: &mut Constraints, fn_app: &TyFnApp, tenv: &mut TypeEnv) {
    let current_mod = tenv.module();
    // println!("{:#?}", fn_app);
    // println!("{}", fn_app.name);
    // println!("{:#?}", cs);

    let symbol_name = fn_app.mod_name.clone().unwrap();
    let symbol_mod_ty = tenv.resolve_type(&current_mod, &AliasType::Variable(symbol_name.clone())).unwrap().clone();
    let symbol_modname = ModName::Named(symbol_mod_ty.as_str().to_owned());
    let fn_name = &fn_app.name;
    println!("{} | {} | {:?} | {:?} | {:?} ", fn_app.orig_name, symbol_name, symbol_mod_ty, symbol_modname, fn_name);
    let ty = tenv.resolve_type(&symbol_modname, &fn_name).unwrap();


    if let Type::UnresolvedModuleFun(_,_,_) = ty {
    //     let inits = tenv.resolve_init(&current_mod, &fn_app.orig_name);
    //     // if let Some(resolved_fn_ty) = tenv.resolve_unresolved(ty.clone(), &symbol_name, &symbol_mod_ty, fn_name, inits) {
    //     //     cs.add(resolved_fn_ty, fun!(fn_app.arg_ty.clone(), fn_app.ret_ty.clone()));
    //     // } else {
    //     // }
    //     cs.add(ty.clone(), fun!(fn_app.arg_ty.clone(), fn_app.ret_ty.clone()));
        tenv.add_unverified(ty.clone());
    }
    cs.add(
        ty.clone(),
        fun!(fn_app.arg_ty.clone(), fn_app.ret_ty.clone())
    );
    cs.add(
        fn_app.arg_ty.clone(),
        fn_app.args.to_ty()
    );

    if let "forward" = fn_name.as_str() {
        if let Type::Module(_, Some(box supplied_ty)) = symbol_mod_ty {
            cs.add(ty.clone(), supplied_ty.clone());
        }
    }

    fn_app
        .args
        .iter()
        .map(|a| cs.collect(&a.arg, tenv))
        .collect::<Vec<_>>();
}
