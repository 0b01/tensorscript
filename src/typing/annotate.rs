use codespan::ByteSpan;
use parsing::term::{Decl, FieldAccess, FnApp, FnAppArg, FnDecl, FnDeclParam, FnTySig, TensorTy,
                   Term, ViewFn, WeightsAssign};
use span::CSpan;
use typing::type_env::{Alias, ModName, TypeEnv};
use typing::typed_term::{ArgsVecInto, Ty};
use typing::typed_term::{TyDecl, TyFieldAccess, TyFnApp, TyFnAppArg, TyFnDecl, TyFnDeclParam,
                            TyGraphDecl, TyNodeDecl, TyTerm, TyUseStmt, TyWeightsAssign,
                            TyWeightsDecl, TyAliasAssign};
use typing::Type;
use std::rc::Rc;
use std::cell::RefCell;
use std::process::exit;
use errors::{Diag, Emitter};

pub struct Annotator {
    pub emitter: Rc<RefCell<Emitter>>,
    pub tenv: Rc<RefCell<TypeEnv>>,
}

impl Annotator {
    pub fn new(emitter: Rc<RefCell<Emitter>>, tenv: Rc<RefCell<TypeEnv>>) -> Self {
        Self {
            emitter,
            tenv,
        }
    }

    pub fn annotate(&self, term: &Term) -> TyTerm {
        use self::Term::*;
        use self::TyTerm::*;
        // println!("{:#?}", term);
        let module = self.tenv.borrow().module();
        match term {
            Ident(ref id, ref span) => {
                let ty = self.tenv.borrow_mut()
                    .resolve_type(&module, &Alias::Variable(id.clone()))
                    .unwrap_or_else(|| {
                        let mut em = self.emitter.borrow_mut();
                        em.add(Diag::SymbolNotFound(id.to_string(), *span));
                        em.print_errs();
                        exit(-1);
                    })
                    .with_span(&span);
                let alias = Alias::Variable(id.to_owned());
                TyTerm::TyIdent(ty, alias, *span)
            }

            Program(ref decls) => TyProgram({
                decls.iter()
                    .map(|d|self.annotate_decl(d))
                    .collect::<Result<_,_>>()
                    .unwrap_or_else(|e| {
                        let mut em = self.emitter.borrow_mut();
                        em.add(e);
                        em.print_errs();
                        exit(-1);
                    })
            }),

            Expr(ref items, ref span) => {
                let ret = self.annotate(&items);
                let ret_ty = ret.ty();
                TyExpr(
                    box ret,
                    ret_ty,
                    *span,
                )
            }

            Integer(i, s) => TyInteger(Type::INT(*s), *i, *s),
            Float(i, s) => TyFloat(Type::FLOAT(*s), *i, *s),
            Block {
                ref stmts,
                ref ret,
                ref span,
            } => {
                let module = self.tenv.borrow_mut().module();
                self.tenv.borrow_mut().push_scope(&module);
                let ret = TyBlock {
                    stmts: Box::new(self.annotate(&stmts)),
                    ret: Box::new(self.annotate(&ret)),
                    span: *span,
                };
                self.tenv.borrow_mut().pop_scope(&module);
                ret
            }
            List(ref stmts) => TyList(stmts.iter().map(|s| self.annotate(&s)).collect()),
            Stmt(ref items, ref span) => TyStmt(
                box self.annotate(&items),
                *span,
            ),
            FieldAccess(ref f_a) => self.annotate_field_access(f_a),
            None => TyNone,
            Pipes(ref pipes) => self.annotate_pipes(pipes),
            Tuple(ref terms, ref s) => self.annotate_tuples(terms, s),
            _ => unimplemented!(),
        }
    }

    fn annotate_tuples(&self, tup: &[Term], s: &ByteSpan) -> TyTerm {
        let (vs, tys)  = tup.iter().map(|i| {
            let tyterm = self.annotate(i);
            let ty = tyterm.ty();
            (tyterm, ty)
        })
        .unzip();

        TyTerm::TyTuple(Type::Tuple(tys, *s), vs, *s)
    }

    fn annotate_pipes(&self, pipes: &[Term]) -> TyTerm {
        let module = self.tenv.borrow().module();
        let mut it = pipes.iter();

        let p0 = it.next().expect("Does not have the first item. Parser is broken.");
        let mut term0 = self.annotate(p0);

        for t in it {
            let prev_arg = TyFnAppArg {
                name: Some(String::from("x")),
                arg: Box::new(term0.clone()),
                span: term0.span(),
            };
            let t = match t {
                // this may be `fc1`
                Term::Ident(ref id, ref span) => {
                    let arg_ty = self.tenv.borrow_mut().fresh_var(span);
                    let ret_ty = self.tenv.borrow_mut().fresh_var(span);
                    TyTerm::TyFnApp(box TyFnApp {
                        mod_name: Some(
                            self.tenv.borrow().resolve_type(&module, &Alias::Variable(id.clone()))
                                .unwrap()
                                .as_str()
                                .to_owned(),
                        ),
                        orig_name: Some(id.to_owned()),
                        name: Alias::Function("forward".to_owned()),
                        arg_ty,
                        args: vec![prev_arg],
                        ret_ty,
                        span: *span,
                    })
                }
                Term::FnApp(ref fn_app) => {
                    let mut typed_fn_app = self.annotate_fn_app(&fn_app);
                    if typed_fn_app.mod_name.is_none() {
                        // log_softmax(dim=1)
                        typed_fn_app.mod_name = Some(
                            self.tenv.borrow_mut().resolve_type(&module, &typed_fn_app.name)
                                .unwrap()
                                .as_str()
                                .to_owned(),
                        );
                        typed_fn_app.name = Alias::Function("forward".to_owned());
                    }
                    typed_fn_app.extend_arg(&prev_arg);
                    TyTerm::TyFnApp(box typed_fn_app)
                }
                Term::FieldAccess(ref f_a) => {
                    let typed_f_a = self.annotate_field_access(&f_a);
                    match typed_f_a {
                        TyTerm::TyFnApp(ref fn_app) => {
                            let mut fn_app = fn_app.clone();
                            fn_app.extend_arg(&prev_arg);
                            TyTerm::TyFnApp(fn_app)
                        }
                        _ => panic!("Error: for field access in a pipeline, use parenthesis: f()"),
                    }
                }
                Term::ViewFn(ref v_f) => TyTerm::TyFnApp(box self.annotate_view_fn(&v_f, &prev_arg)),
                _ => unimplemented!(),
            };
            term0 = t.clone();
        }

        term0
    }

    fn annotate_view_fn(&self, v_fn: &ViewFn, arg: &TyFnAppArg) -> TyFnApp {
        let module = self.tenv.borrow().module();
        let tsr = self.tenv.borrow_mut().create_tensor(&module, &v_fn.dims, &v_fn.span);
        TyFnApp {
            mod_name: Some("view".to_string()),
            orig_name: None,
            name: Alias::Function("forward".to_owned()),
            arg_ty: args!(arg!("x", arg.arg.ty())),
            ret_ty: tsr.clone(),
            args: vec![arg.clone()],
            span: v_fn.span,
        }
    }

    fn annotate_decl(&self, decl: &Decl) -> Result<TyDecl, Diag> {
        use self::Decl::*;
        let ret = match decl {
            NodeDecl(ref decl) => {
                let module = ModName::Named(decl.name.to_owned());
                self.tenv.borrow_mut().set_module(module.clone());
                let assigns = &decl.defs;
                for a in assigns {
                    self.tenv.borrow_mut()
                        .import_node_assign(&module, a)
                        .unwrap_or_else(|e| self.emitter.borrow_mut().add(e));
                }

                self.tenv.borrow_mut().upsert_module(&module);
                // if some dimension alias are not imported, create them
                self.tenv.borrow_mut()
                    .import_top_level_ty_sig(&module, &decl.ty_sig.from)
                    .unwrap_or_else(|e| self.emitter.borrow_mut().add(e));
                self.tenv.borrow_mut()
                    .import_top_level_ty_sig(&module, &decl.ty_sig.to)
                    .unwrap_or_else(|e| self.emitter.borrow_mut().add(e));

                let ty_sig = self.annotate_fn_ty_sig(
                    decl.name.to_owned(),
                    "forward".to_owned(),
                    &decl.ty_sig,
                    &decl.span)?;
                let mod_ty_sig = Type::Module(
                    decl.name.clone(),
                    Some(box ty_sig.clone()),
                    decl.span,
                );

                // add current name into global scope
                self.tenv.borrow_mut().add_type(
                    &ModName::Global,
                    &Alias::Variable(decl.name.to_string()),
                    mod_ty_sig.clone(),
                )
                .unwrap_or_else(|e|self.emitter.borrow_mut().add(e));

                // add "self" into module scope
                self.tenv.borrow_mut().add_type(
                    &module,
                    &Alias::Variable("self".to_owned()),
                    mod_ty_sig.clone(),
                )
                .unwrap_or_else(|e|self.emitter.borrow_mut().add(e));

                // // add "forward" function into module scope
                // tenv.add_type(&module, "self.forward", ty_sig.clone());

                TyDecl::TyNodeDecl(TyNodeDecl {
                    name: decl.name.clone(),
                    ty_sig,
                    span: decl.span,
                })
            }
            WeightsDecl(ref decl) => {
                self.tenv.borrow_mut().set_module(ModName::Named(decl.name.to_owned()));
                TyDecl::TyWeightsDecl(TyWeightsDecl {
                    name: decl.name.clone(),
                    ty_sig: self.annotate_fn_ty_sig(decl.name.to_owned(), "forward".to_owned(),&decl.ty_sig, &decl.span)?,
                    inits: decl.inits
                        .iter()
                        .map(|t| self.annotate_weights_assign(t))
                        .collect(),
                    span: decl.span,
                })
            }
            GraphDecl(ref decl) => {
                self.tenv.borrow_mut().set_module(ModName::Named(decl.name.to_owned()));
                TyDecl::TyGraphDecl(TyGraphDecl {
                    name: decl.name.clone(),
                    ty_sig: self.annotate_fn_ty_sig(decl.name.to_owned(), "forward".to_owned(),&decl.ty_sig, &decl.span)?,
                    fns: decl.fns.iter().map(|f| self.annotate_fn_decl(f)).collect(),
                    span: decl.span,
                })
            }
            UseStmt(ref decl) => {
                // in global scope
                // import names into scope
                // also import module and its associated functions
                for name in &decl.imported_names {
                    let ty = Type::Module(name.to_string(), None, decl.span);
                    self.tenv.borrow_mut()
                        .add_type(
                            &ModName::Global,
                            &Alias::Variable(name.to_string()),
                            ty)
                        .unwrap_or_else(|e|self.emitter.borrow_mut().add(e));
                    let import_result = self.tenv.borrow_mut().import_module(&decl.mod_name, &name);
                    match import_result {
                        Some(Ok(())) => (),
                        Some(Err(e)) => self.emitter.borrow_mut().add(e),
                        None => self.emitter.borrow_mut()
                            .add(Diag::ImportError(name.to_string(), decl.span))
                    }
                }

                TyDecl::TyUseStmt(TyUseStmt {
                    mod_name: decl.mod_name.clone(),
                    imported_names: decl.imported_names.clone(),
                    span: decl.span,
                })
            }
            AliasAssign(ref assign) => {
                let module = ModName::Global;
                self.tenv.borrow_mut().set_module(module.clone());
                self.tenv.borrow_mut()
                    .import_node_assign(&module, assign)
                    .unwrap_or_else(|e| self.emitter.borrow_mut().add(e));
                TyDecl::TyAliasAssign(TyAliasAssign::Placeholder)
            }
        };
        self.tenv.borrow_mut().set_module(ModName::Global);
        Ok(ret)
    }

    fn annotate_fn_ty_sig(&self, modname: String, name: String, sig: &FnTySig, span: &ByteSpan) -> Result<Type, Diag> {
        Ok(Type::FUN(
            modname,
            name,
            Box::new(self.annotate_tensor_ty_sig(&sig.from, span)?),
            Box::new(self.annotate_tensor_ty_sig(&sig.to, span)?),
            *span,
        ))
    }

    fn annotate_tensor_ty_sig(&self, sig: &TensorTy, _span: &ByteSpan) -> Result<Type, Diag> {
        use self::TensorTy::*;
        let module = self.tenv.borrow_mut().module();
        match sig {
            Generic(ref dims, ref sp) => Ok(self.tenv.borrow_mut().create_tensor(&module, dims, sp)),
            Tensor(ref als, ref sp) => {
                let ty = self.tenv.borrow_mut()
                    .resolve_type(&module, &Alias::Variable(als.clone()));
                match ty {
                    Some(t) => Ok(t.clone().with_span(sp)),
                    None =>
                        Err(Diag::SymbolNotFound(als.clone(), *sp)),
                }
            }
        }
    }

    fn annotate_weights_assign(&self, w_assign: &WeightsAssign) -> TyWeightsAssign {
        let name = w_assign.name.clone();
        let fn_ty = w_assign
            .clone()
            .mod_sig
            .map(|sig|
                box self.annotate_fn_ty_sig(
                    w_assign.mod_name.to_owned(),
                    "forward".to_owned(),
                    &sig,
                    &w_assign.span
                ).unwrap()
            );

        let module = self.tenv.borrow_mut().module();
        self.tenv.borrow_mut().add_type(
            &module,
            &Alias::Variable(name.to_owned()),
            Type::Module(
                w_assign.mod_name.to_owned(),
                fn_ty.clone(),
                w_assign.span,
            ),
        )
        .unwrap_or_else(|e|self.emitter.borrow_mut().add(e));

        let fn_args: Vec<TyFnAppArg> = w_assign
            .fn_args
            .iter()
            .map(|a| self.annotate_fn_app_arg(a))
            .collect();

        self.tenv.borrow_mut().add_init(&module, &name, fn_args.clone());

        TyWeightsAssign {
            name,
            mod_name: w_assign.mod_name.clone(),
            fn_name: w_assign.fn_name.clone(),
            arg_ty: fn_args.to_ty(&w_assign.span),
            fn_args,
            span: w_assign.span,
        }
    }

    fn annotate_fn_app_arg(&self, call: &FnAppArg) -> TyFnAppArg {
        let tyterm = Box::new(self.annotate(&call.arg));
        // println!("{}", tyterm);
        TyFnAppArg {
            name: Some(call.name.clone()),
            arg: tyterm,
            span: call.span,
        }
    }

    fn annotate_fn_app(&self, fn_app: &FnApp) -> TyFnApp {
        let FnApp {
            ref name,
            ref args,
            ref span,
        } = fn_app;
        let t_args: Vec<TyFnAppArg> = args.iter().map(|a| self.annotate_fn_app_arg(&a)).collect();
        let arg_ty = t_args.to_ty(&fn_app.span);
        TyFnApp {
            mod_name: None,
            orig_name: Some(name.to_owned()),
            name: Alias::Variable(name.to_owned()),
            arg_ty,
            args: t_args,
            ret_ty: self.tenv.borrow_mut().fresh_var(&span),
            span: *span,
        }
    }

    fn annotate_fn_decl(&self, f: &FnDecl) -> TyFnDecl {
        let module = self.tenv.borrow_mut().module();
        self.tenv.borrow_mut().push_scope(&module);
        let mod_ty = self.tenv.borrow_mut().resolve_type(
            &ModName::Global,
            &Alias::Variable(module.as_str().to_owned()),
        ).unwrap().clone().with_span(&f.span);

        let mut decl = TyFnDecl {
            name: Alias::Function(f.name.clone()),
            fn_params: {
                match (&f.fn_params, f.name.as_str()) {
                    (None, "forward") =>
                        vec![],
                    (None, _) =>
                        panic!("function parameter ellided for a function other than forward!"),
                    (Some(params), _)=> {
                        params
                        .iter()
                        .map(|p| self.annotate_fn_decl_param(p))
                        .collect()
                    }
                }
            },
            // fn_ty: tenv.fresh_var(),
            fn_ty: self.tenv.borrow_mut().fresh_var(&f.span),
            func_block: Box::new(TyTerm::TyNone),  // same here
            span: f.span,
        };

        match f.name.as_str() {
            // special function signatures
            "new" => {
                decl.fn_ty = Type::FUN(
                    module.as_str().to_owned(),
                    "new".to_owned(),
                    box decl.fn_params.to_ty(&f.span),
                    box mod_ty.clone(),
                    f.span
                );
            }
            "forward" => {
                let name0 = match decl.fn_params.get(0) {
                    Some(e) => e.name.clone(),
                    None => "x".to_string(),
                };
                if let Type::Module(ref modn, Some(box Type::FUN(_,_,ref p, ref r, _)), _) = mod_ty.clone() {
                    let ty_sig = *p.clone();

                    // type the first parameter
                    decl.fn_params = vec![TyFnDeclParam {
                        name: name0.clone(),
                        ty: ty_sig.clone(),
                        span: f.span,
                    }];

                    // override the old first argument which is []
                    unsafe {
                        self.tenv.borrow_mut().add_type_allow_replace(&module, &Alias::Variable(name0.clone()), ty_sig);
                    }

                    // // type the function return parameter
                    decl.fn_ty = Type::FUN(
                        modn.to_owned(),
                        "forward".to_owned(),
                        box decl.fn_params.to_ty(&f.span),
                        r.clone(),
                        f.span
                    );

                // type the function itself...
                // decl.fn_ty = Type::FUN(p.clone(), r.clone());
                } else {
                    panic!("Signature of module is incorrect!");
                };
            }
            _ => {
                // decl.return_ty = tenv.resolve_tensor(&module, &f.return_ty, &f.span);
                // decl.fn_ty = Type::FUN(box decl.param_ty.clone(), box decl.return_ty.clone());
            }
        };

        // decl.param_ty = decl.fn_params.to_ty(&f.span).clone();

        // if decl.name == "test" {
        //     panic!("{:#?}", decl);
        // }

        decl.func_block = Box::new(self.annotate(&f.func_block));

        self.tenv.borrow_mut().pop_scope(&module);

        // insert this function into typeenv
        self.tenv.borrow_mut().add_type(
            &module,
            &Alias::Function(f.name.clone()),
            decl.fn_ty.clone(),
        )
        .unwrap_or_else(|e|self.emitter.borrow_mut().add(e));

        decl
    }

    fn annotate_fn_decl_param(&self, p: &FnDeclParam) -> TyFnDeclParam {
        let module = self.tenv.borrow_mut().module();
        let name = p.name.clone();
        let ty = self.tenv.borrow_mut().resolve_tensor(&module, &p.ty_sig, &p.span);
        self.tenv.borrow_mut()
            .add_type(&module, &Alias::Variable(name.clone()), ty.clone())
            .unwrap_or_else(|e| {
                let mut em = self.emitter.borrow_mut();
                em.add(e);
            });
        TyFnDeclParam {
            name,
            ty,
            span: p.span,
        }
    }

    fn annotate_field_access(&self, f_a: &FieldAccess) -> TyTerm {
        let module = self.tenv.borrow_mut().module();
        match module {
            ModName::Global => panic!("Cannot use field access in global scope"),
            ModName::Named(ref _mod_name) => match f_a.func_call {
                None => TyTerm::TyFieldAccess(TyFieldAccess {
                    mod_name: f_a.mod_name.clone(),
                    field_name: f_a.field_name.clone(),
                    ty: self.tenv.borrow_mut().fresh_var(&f_a.span),
                    span: f_a.span,
                }),
                Some(ref v) => {
                    let args: Vec<TyFnAppArg> =
                        v.iter().map(|arg| self.annotate_fn_app_arg(arg)).collect();
                    let args_ty = args.to_ty(&f_a.span);
                    TyTerm::TyFnApp(box TyFnApp {
                        mod_name: Some(f_a.mod_name.clone()),
                        orig_name: Some(f_a.mod_name.clone()),
                        name: Alias::Function(f_a.field_name.clone()),
                        arg_ty: args_ty,
                        args,
                        ret_ty: self.tenv.borrow_mut().fresh_var(&f_a.span),
                        span: f_a.span,
                    })
                }
            },
        }
    }
}
