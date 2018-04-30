use std::collections::BTreeSet;

use typing::type_env::{Alias, ModName, TypeEnv};
use typing::typed_term::Ty;
use typing::typed_term::*;
use typing::Type;
use std::rc::Rc;
use std::process::exit;
use std::cell::RefCell;
use errors::{ Emitter, Diag };

use span::CSpan;

#[derive(Debug, Hash, Eq, PartialEq, Clone, PartialOrd, Ord)]
pub struct Equals(pub Type, pub Type);

#[derive(Debug, Clone)]
pub struct Constraints {
    pub set: BTreeSet<Equals>,
    pub emitter: Rc<RefCell<Emitter>>,
    pub tenv: Rc<RefCell<TypeEnv>>,
}

impl Constraints {

    pub fn new(emitter: Rc<RefCell<Emitter>>, tenv: Rc<RefCell<TypeEnv>>) -> Self {
        Constraints {
            set: BTreeSet::new(),
            emitter,
            tenv
        }
    }

    pub fn is_empty(&self) -> bool {
        self.set.is_empty()
    }

    fn add(&mut self, a: Type, b: Type) {
        // println!("{:?} {:?}", a, b);
        self.set.insert(Equals(a, b));
    }

    pub fn collect(&mut self, typed_term: &TyTerm) {
        use self::TyTerm::*;
        let module = { self.tenv.borrow_mut().module().clone() };
        // println!("{}", typed_term);
        match typed_term {
            TyProgram(ref decls) => decls
                .iter()
                .map(|decl| self.collect_decl(&decl))
                .collect(),
            TyInteger(_, _, _) => (),
            TyFloat(_, _, _) => (),
            TyList(ref terms) => terms.iter().map(|t| self.collect(&t)).collect(),
            TyTuple(_, ref terms, _) => terms.iter().map(|t| self.collect(&t)).collect(),
            TyIdent(ref t, ref name, ref sp) => {
                let ty =  self.tenv.borrow_mut()
                    .resolve_type(&module, &name)
                    .unwrap() // ... todo:
                    .clone()
                    .with_span(&sp);
                self.add(t.clone(), ty);
            }
            // &TyFieldAccess(TyFieldAccess),
            TyFnApp(ref fn_app) => self.collect_fn_app(&fn_app),
            TyBlock { ref stmts, ref ret, .. } => {
                self.tenv.borrow_mut().push_scope_collection(&module);
                self.collect(&stmts);
                self.collect(&ret);
                self.tenv.borrow_mut().pop_scope(&module);
            }
            TyExpr(ref items, ref ty, _) => {
                self.collect(&items);
                self.add(ty.clone(), items.ty());
            }
            TyStmt(ref items, _) => self.collect(&items),
            TyNone => (),
            _ => {
                panic!("{:#?}", typed_term);
            }
        }
    }
    fn collect_decl(&mut self, decl: &TyDecl) {
        use self::TyDecl::*;
        match decl {
            TyGraphDecl(d) => self.collect_graph_decl(d),
            TyNodeDecl(d) => self.collect_node_decl(d),
            TyUseStmt(d) => self.collect_use_stmt(d),
            TyWeightsDecl(d) => self.collect_weights_decl(d),
        }
        self.tenv.borrow_mut().set_module(ModName::Global);
    }

    fn collect_graph_decl(&mut self, decl: &TyGraphDecl) {
        // type decl should be the same
        self.tenv.borrow_mut().set_module(ModName::Named(decl.name.clone()));
        let graph_ty_sig = self.tenv.borrow_mut()
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
            self.collect_fn_decl(&f);
        }
    }

    fn collect_fn_decl(&mut self, decl: &TyFnDecl) {
        let module = self.tenv.borrow_mut().module();
        self.tenv.borrow_mut().push_scope_collection(&module);

        self.collect(&decl.func_block);
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

        self.tenv.borrow_mut().pop_scope(&module);
        // ...
    }

    fn collect_node_decl(&mut self, decl: &TyNodeDecl) {
        self.tenv.borrow_mut().set_module(ModName::Named(decl.name.clone()));
        // type decl should be the same
        let graph_ty_sig = self.tenv.borrow_mut().resolve_type(&ModName::Global, &Alias::Variable(decl.name.clone()))
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

    fn collect_weights_decl(&mut self, decl: &TyWeightsDecl) {
        self.tenv.borrow_mut().set_module(ModName::Named(decl.name.clone()));
        // type decl should be the same
        let graph_ty_sig = self.tenv.borrow_mut().resolve_type(&ModName::Global, &Alias::Variable(decl.name.clone()))
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
            self.collect_weights_assign(&w);
        }
    }

    fn collect_use_stmt(&mut self, _decl: &TyUseStmt) {
        ()
    }

    fn collect_weights_assign(&mut self, w_a: &TyWeightsAssign) {
        let mod_name = &w_a.mod_name;
        // convert into a fn_app and collect on `self.new` method
        let ret_ty = self.tenv.borrow_mut().fresh_var(&w_a.span);
        self.collect_fn_app(
            &TyFnApp {
                mod_name: Some(mod_name.to_string()),
                orig_name: None,
                name: Alias::Function("new".to_owned()),
                arg_ty: w_a.arg_ty.clone(),
                ret_ty,
                args: w_a.fn_args.clone(),
                span: w_a.span,
            },
        );
        ()
    }

    fn collect_fn_app(&mut self, fn_app: &TyFnApp) {
        let current_mod = self.tenv.borrow_mut().module();
        // println!("{:#?}", fn_app);
        // println!("{}", fn_app.name);
        // println!("{:#?}", cs);

        let symbol_name = fn_app.mod_name.clone().unwrap();
        let symbol_mod_ty = match &fn_app.orig_name {
            Some(ref orig_name) => self.tenv.borrow_mut().resolve_type(&current_mod, &Alias::Variable(orig_name.clone())).unwrap().clone(),
            None => {
                let resolved_ty = self.tenv.borrow_mut()
                    .resolve_type(
                        &current_mod,
                        &Alias::Variable(symbol_name.clone())
                    );
                match resolved_ty {
                    Some(ty) => ty,
                    None => {
                        let e = Diag::SymbolNotFound(symbol_name.to_owned(), fn_app.span());
                        self.emitter.borrow_mut().add(e);
                        self.emitter.borrow().print_errs();
                        exit(-1);
                    }
                }
            }
        };

        let symbol_modname = ModName::Named(symbol_mod_ty.as_str().to_owned());
        let fn_name = &fn_app.name;
        let ty = self.tenv.borrow_mut().resolve_type(&symbol_modname, &fn_name);
        let ty = match ty {
            Some(ty) => ty,
            None => {
                let e = Diag::SymbolNotFound(fn_name.as_str().to_owned(), fn_app.span);
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
                self.tenv.borrow_mut().resolve_unresolved(
                    &ty,
                    &fn_app.name.as_str(),
                    fn_app.arg_ty.clone(),
                    fn_app.ret_ty.clone(),
                    fn_app.args.clone(),
                    None
                )
            } else {
                let inits = self.tenv.borrow_mut().resolve_init(&current_mod, &fn_app.orig_name.clone().unwrap());
                self.tenv.borrow_mut().resolve_unresolved(
                    &ty,
                    fn_name.as_str(),
                    fn_app.arg_ty.clone(),
                    fn_app.ret_ty.clone(),
                    fn_app.args.clone(),
                    inits
                )
            };

            match resolution {
                Ok(Some(resolved_fn_ty)) =>
                    self.add(
                        resolved_fn_ty,
                        fun!(
                            symbol_name,
                            fn_app.name.as_str(),
                            fn_app.arg_ty.clone(),
                            fn_app.ret_ty.clone()
                        )
                    ),
                Ok(None) =>
                    self.tenv.borrow_mut().add_unverified(ty.clone()),
                Err(e) => {
                    self.emitter.borrow_mut().add(e);
                }
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
            self.collect(&a.arg);
        }

    }
}

