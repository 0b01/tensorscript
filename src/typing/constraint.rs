use std::collections::BTreeSet;

use typing::type_env::{Alias, ModName, TypeEnv};
use typing::typed_term::Ty;
use typing::typed_term::*;
use typing::Type;
use std::rc::Rc;
use std::cell::RefCell;
use errors::{ EmitErr, Emitter, TensorScriptDiagnostic };

use span::CSpan;

#[derive(Debug, Hash, Eq, PartialEq, Clone, PartialOrd, Ord)]
pub struct Equals(pub Type, pub Type);

#[derive(Debug, Clone)]
pub struct Constraints {
    pub set: BTreeSet<Equals>,
    pub emitter: Rc<RefCell<Emitter>>,
}

impl EmitErr for Constraints {
    fn emit_err(&self) {
        self.emitter.borrow().print_errs();
    }
}

impl Constraints {

    pub fn new(emitter: Rc<RefCell<Emitter>>) -> Self {
        Constraints {
            set: BTreeSet::new(),
            emitter,
        }
    }

    pub fn is_empty(&self) -> bool {
        self.set.is_empty()
    }

    fn add(&mut self, a: Type, b: Type) {
        // println!("{:?} {:?}", a, b);
        self.set.insert(Equals(a, b));
    }

    pub fn collect(&mut self, typed_term: &TyTerm, tenv: &mut TypeEnv) {
        use self::TyTerm::*;
        let module = tenv.module();
        // println!("{}", typed_term);
        match typed_term {
            TyProgram(ref decls) => decls
                .iter()
                .map(|decl| self.collect_decl(&decl, tenv))
                .collect(),
            TyInteger(_, _, _) => (),
            TyFloat(_, _, _) => (),
            TyList(ref terms) => terms.iter().map(|t| self.collect(&t, tenv)).collect(),
            TyTuple(_, ref terms, _) => terms.iter().map(|t| self.collect(&t, tenv)).collect(),
            TyIdent(ref t, ref name, ref sp) => self.add(
                t.clone(),
                tenv.resolve_type(&module, &name)
                    .expect(&format!("{:#?}", tenv))
                    .clone()
                    .with_span(&sp),
            ),
            // &TyFieldAccess(TyFieldAccess),
            TyFnApp(ref fn_app) => self.collect_fn_app(&fn_app, tenv),
            TyBlock { ref stmts, ref ret, .. } => {
                tenv.push_scope_collection(&module);
                self.collect(&stmts, tenv);
                self.collect(&ret, tenv);
                tenv.pop_scope(&module);
            }
            TyExpr(ref items, ref ty, _) => {
                self.collect(&items, tenv);
                self.add(ty.clone(), items.ty());
            }
            TyStmt(ref items, _) => self.collect(&items, tenv),
            TyNone => (),
            _ => {
                panic!("{:#?}", typed_term);
            }
        }
    }
    fn collect_decl(&mut self, decl: &TyDecl, tenv: &mut TypeEnv) {
        use self::TyDecl::*;
        match decl {
            TyGraphDecl(d) => self.collect_graph_decl(d, tenv),
            TyNodeDecl(d) => self.collect_node_decl(d, tenv),
            TyUseStmt(d) => self.collect_use_stmt(d, tenv),
            TyWeightsDecl(d) => self.collect_weights_decl(d, tenv),
        }
        tenv.set_module(ModName::Global);
    }

    fn collect_graph_decl(&mut self, decl: &TyGraphDecl, tenv: &mut TypeEnv) {
        // type decl should be the same
        tenv.set_module(ModName::Named(decl.name.clone()));
        let graph_ty_sig = tenv
            .resolve_type(&ModName::Global, &Alias::Variable(decl.name.clone()))
            .unwrap()
            .clone();

        self.add(
            Type::Module(
                decl.name.to_owned(),
                Some(box decl.ty_sig.clone()),
                decl.span,
            ),
            graph_ty_sig,
        );

        // collect fn_decls
        for f in &decl.fns {
            self.collect_fn_decl(&f, tenv);
        }
    }

    fn collect_fn_decl(&mut self, decl: &TyFnDecl, tenv: &mut TypeEnv) {
        let module = tenv.module();
        tenv.push_scope_collection(&module);

        self.collect(&decl.func_block, tenv);
        let func =
            Type::FUN(
                module.as_str().to_owned(),
                decl.name.as_str().to_owned(),
                box decl.fn_params.to_ty(&decl.span),
                box decl.func_block.ty(),
                decl.span
                );

        self.add(decl.fn_ty.clone(), func.clone());

        // if decl.name == Alias::Function("forward".to_owned()) {
        //     panic!("{:?}, {:?}", decl.fn_ty, func);
        // }

        tenv.pop_scope(&module);
        // ...
    }

    fn collect_node_decl(&mut self, decl: &TyNodeDecl, tenv: &mut TypeEnv) {
        tenv.set_module(ModName::Named(decl.name.clone()));
        // type decl should be the same
        let graph_ty_sig = tenv.resolve_type(&ModName::Global, &Alias::Variable(decl.name.clone()))
            .unwrap()
            .clone();
        self.add(
            Type::Module(
                decl.name.to_owned(),
                Some(box decl.ty_sig.clone()),
                decl.span,
            ),
            graph_ty_sig,
        );
    }

    fn collect_weights_decl(&mut self, decl: &TyWeightsDecl, tenv: &mut TypeEnv) {
        tenv.set_module(ModName::Named(decl.name.clone()));
        // type decl should be the same
        let graph_ty_sig = tenv.resolve_type(&ModName::Global, &Alias::Variable(decl.name.clone()))
            .unwrap()
            .clone();
        self.add(
            Type::Module(
                decl.name.to_owned(),
                Some(box decl.ty_sig.clone()),
                decl.span,
            ),
            graph_ty_sig,
        );

        // collect weight assigns
        for w in &decl.inits {
            self.collect_weights_assign(&w, tenv);
        }
    }

    fn collect_use_stmt(&mut self, _decl: &TyUseStmt, _tenv: &TypeEnv) {
        ()
    }

    fn collect_weights_assign(&mut self, w_a: &TyWeightsAssign, tenv: &mut TypeEnv) {
        let mod_name = &w_a.mod_name;
        // convert into a fn_app and collect on `self.new` method
        self.collect_fn_app(
            &TyFnApp {
                mod_name: Some(mod_name.to_string()),
                orig_name: None,
                name: Alias::Function("new".to_owned()),
                arg_ty: w_a.arg_ty.clone(),
                ret_ty: tenv.fresh_var(&w_a.span),
                args: w_a.fn_args.clone(),
                span: w_a.span,
            },
            tenv,
        );
        ()
    }

    fn collect_fn_app(&mut self, fn_app: &TyFnApp, tenv: &mut TypeEnv) {
        let current_mod = tenv.module();
        // println!("{:#?}", fn_app);
        // println!("{}", fn_app.name);
        // println!("{:#?}", cs);

        let symbol_name = fn_app.mod_name.clone().unwrap();
        let symbol_mod_ty = match &fn_app.orig_name {
            Some(ref orig_name) => tenv.resolve_type(&current_mod, &Alias::Variable(orig_name.clone())).unwrap().clone(),
            None => tenv.resolve_type(&current_mod, &Alias::Variable(symbol_name.clone())).unwrap().clone(),
        };

        let symbol_modname = ModName::Named(symbol_mod_ty.as_str().to_owned());
        let fn_name = &fn_app.name;
        let ty = tenv.resolve_type(&symbol_modname, &fn_name);
        let ty = match ty {
            Some(ty) => ty,
            None => {
                let e = TensorScriptDiagnostic::SymbolNotFound(fn_name.as_str().to_owned(), fn_app.span);
                self.emitter.borrow_mut().add(e); // ...
                return;
            }
        };

        // println!(
        //     "{:?} | {:?} | {} | {:?} | {:?} | {:?} ",
        //     ty, fn_app.orig_name, symbol_name, symbol_mod_ty, symbol_modname, fn_name
        // );

        if let Type::UnresolvedModuleFun(..) = ty {
            let resolution = if fn_app.orig_name.is_none() { // this is in a weight assign fn
                // println!("{:?}, {:?}", &fn_app.mod_name.clone().unwrap().as_str(), fn_app.name);
                tenv.resolve_unresolved(
                    &ty,
                    &fn_app.name.as_str(),
                    fn_app.arg_ty.clone(),
                    fn_app.ret_ty.clone(),
                    fn_app.args.clone(),
                    None
                )
            } else {
                let inits = tenv.resolve_init(&current_mod, &fn_app.orig_name.clone().unwrap());
                tenv.resolve_unresolved(
                    &ty,
                    fn_name.as_str(),
                    fn_app.arg_ty.clone(),
                    fn_app.ret_ty.clone(),
                    fn_app.args.clone(),
                    inits
                )
            };
            if let Some(resolved_fn_ty) = resolution {
                // println!("{:#?}", resolved_fn_ty);
                self.add(
                    resolved_fn_ty,
                    fun!(
                        symbol_name,
                        fn_app.name.as_str(),
                        fn_app.arg_ty.clone(),
                        fn_app.ret_ty.clone()
                    )
                );
            } else {
                tenv.add_unverified(ty.clone());
            }
        }

        self.add(
            ty.clone(),
            fun!(symbol_name, fn_app.name.as_str(), fn_app.arg_ty.clone(), fn_app.ret_ty.clone()),
        );

        self.add(fn_app.arg_ty.clone(), fn_app.args.to_ty(&fn_app.span));

        if let "forward" = fn_name.as_str() {
            if let Type::Module(_, Some(box supplied_ty), _) = symbol_mod_ty {
                if let Type::FUN(_,_,box p,box r, _) = supplied_ty {
                    self.add(fn_app.arg_ty.clone().clone(),
                        args!(arg!("x",p.clone())));
                    self.add(fn_app.ret_ty.clone(), r.clone());
                }
            }
        }

        for a in &fn_app.args {
            self.collect(&a.arg, tenv);
        }
    }
}

