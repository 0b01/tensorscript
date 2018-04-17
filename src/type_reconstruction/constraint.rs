use std::collections::HashSet;

use typed_ast::Type;
use typed_ast::type_env::{ModName, TypeEnv};
use typed_ast::typed_term::*;

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
        let module = tenv.module().clone();
        // println!("{}", typed_term);
        match typed_term {
            &TyProgram(ref decls) => decls
                .iter()
                .map(|decl| collect_decl(self, &decl, tenv))
                .collect(),
            &TyInteger(_,_) => (),
            &TyFloat(_,_) => (),
            &TyList(ref terms) => terms.iter().map(|t| self.collect(&t, tenv)).collect(),
            &TyIdent(ref t, ref name) => self.add(
                t.clone(),
                tenv.resolve_type(&module, name.as_str()).expect(&format!("{:#?}",tenv)).clone(),
            ),
            // &TyFieldAccess(TyFieldAccess),
            &TyFnApp(ref fn_app) => collect_fn_app(self, &fn_app, tenv),
            &TyBlock { ref stmts, ref ret } => {
                tenv.push_scope_collection(&module);
                self.collect(&stmts, tenv);
                self.collect(&ret, tenv);
                tenv.pop_scope(&module);
            }
            &TyExpr { ref items, ty: _ } => self.collect(&items, tenv), // ... need to use ty?
            &TyStmt { ref items } => self.collect(&items, tenv),
            &TyViewFn(ref view_fn) => {
                self.collect(&view_fn.arg.arg, tenv);
            },
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
    let graph_ty_sig = tenv.resolve_type(&ModName::Global, decl.name.as_str())
        .unwrap()
        .clone();
    cs.add(decl.ty_sig.clone(), graph_ty_sig);
    // collect fn_decls
    decl.fns
        .iter()
        .map(|f| collect_fn_decl(cs, &f, tenv))
        .collect::<Vec<_>>();
}

fn collect_fn_decl(cs: &mut Constraints, decl: &TyFnDecl, tenv: &mut TypeEnv) {
    let module = tenv.module().clone();
    tenv.push_scope_collection(&module);

    cs.collect(&decl.func_block, tenv);
    cs.add(decl.return_ty.clone(), decl.func_block.ty());
    cs.add(
        decl.fn_ty.clone(),
        Type::FUN(
            Box::new(decl.param_ty.clone()),
            Box::new(decl.return_ty.clone()),
        ),
    );
    tenv.pop_scope(&module);
    // ...
}

fn collect_node_decl(cs: &mut Constraints, decl: &TyNodeDecl, tenv: &mut TypeEnv) {
    tenv.set_module(ModName::Named(decl.name.clone()));
    // type decl should be the same
    let graph_ty_sig = tenv.resolve_type(&ModName::Global, decl.name.as_str())
        .unwrap()
        .clone();
    cs.add(decl.ty_sig.clone(), graph_ty_sig);
}

fn collect_weights_decl(cs: &mut Constraints, decl: &TyWeightsDecl, tenv: &mut TypeEnv) {
    tenv.set_module(ModName::Named(decl.name.clone()));
    // type decl should be the same
    let graph_ty_sig = tenv.resolve_type(&ModName::Global, decl.name.as_str())
        .unwrap()
        .clone();
    cs.add(decl.ty_sig.clone(), graph_ty_sig);

    // collect weight assigns
    decl.inits
        .iter()
        .map(|w| collect_weights_assign(cs, &w, tenv))
        .collect::<Vec<_>>();
}

fn collect_use_stmt(_cs: &mut Constraints, _decl: &TyUseStmt, _tenv: &TypeEnv) {
    ()
}

fn collect_weights_assign(_cs: &mut Constraints, _w_a: &TyWeightsAssign, _tenv: &TypeEnv) {
    // w_a.fn_ty
    // TODO:
    // ...
    ()
}

fn collect_fn_app(cs: &mut Constraints, fn_app: &TyFnApp, tenv: &mut TypeEnv) {
    let module = tenv.module().clone();
    // println!("{:#?}", fn_app);
    let looked_up_fn_ty = match fn_app.mod_name {

        // is of the form `fc1.init_normal()`
        Some(ref mod_name) => {
            let defined_module = tenv.resolve_type(&module, &mod_name).unwrap().clone();
            let mod_name = match defined_module {
                Type::Module(s, _) => s, // ...
                _ => panic!("must be of type: Module but is {:?}", defined_module),
            };
            tenv.resolve_type(&ModName::Named(mod_name), &fn_app.name).unwrap().clone()
        },

        // must be defined in current scope or global scope
        None => {
            tenv.resolve_type(&module, &fn_app.name).unwrap().clone()
        }
    };
    
    // println!("{:#?}", Equals(looked_up_fn_ty.clone(), Type::FUN(Box::new(fn_app.arg_ty.clone()), Box::new(fn_app.ret_ty.clone()))));
    cs.add(looked_up_fn_ty, Type::FUN(Box::new(fn_app.arg_ty.clone()), Box::new(fn_app.ret_ty.clone())));
    fn_app.args.iter().map(|a| cs.collect(&a.arg, tenv)).collect::<Vec<_>>();

}
