use codespan::ByteSpan;
use parser::term::{Decl, FieldAccess, FnApp, FnAppArg, FnDecl, FnDeclParam, FnTySig, TensorTy,
                   Term, ViewFn, WeightsAssign};
use span::CSpan;
use typing::type_env::{Alias, ModName, TypeEnv};
use typing::typed_term::{ArgsVecInto, Ty};
use typing::typed_term::{TyDecl, TyFieldAccess, TyFnApp, TyFnAppArg, TyFnDecl, TyFnDeclParam,
                            TyGraphDecl, TyNodeDecl, TyTerm, TyUseStmt, TyWeightsAssign,
                            TyWeightsDecl};
use typing::Type;

pub fn annotate(term: &Term, tenv: &mut TypeEnv) -> TyTerm {
    use self::Term::*;
    use self::TyTerm::*;
    // println!("{:#?}", term);
    let module = tenv.module();
    match term {
        Ident(ref id, ref span) => {
            let ty = tenv.resolve_type(&module, &Alias::Variable(id.clone()))
                .unwrap().with_span(&span);
            let alias = Alias::Variable(id.to_owned());
            TyTerm::TyIdent(ty, alias, span.clone())
        }

        &Program(ref decls) => TyProgram(decls.iter().map(|d| annotate_decl(d, tenv)).collect()),

        &Expr(ref items, ref span) => {
            let ret = annotate(&items, tenv);
            let ret_ty = ret.ty();
            TyExpr(
                box ret,
                ret_ty,
                span.clone(),
            )
        }

        &Integer(i, s) => TyInteger(Type::INT(s.clone()), i, s),
        &Float(i, s) => TyFloat(Type::FLOAT(s.clone()), i, s),
        &Block {
            ref stmts,
            ref ret,
            ref span,
        } => {
            let module = tenv.module();
            tenv.push_scope(&module);
            let ret = TyBlock {
                stmts: Box::new(annotate(&stmts, tenv)),
                ret: Box::new(annotate(&ret, tenv)),
                span: span.clone(),
            };
            tenv.pop_scope(&module);
            ret
        }
        &List(ref stmts) => TyList(stmts.iter().map(|s| annotate(&s, tenv)).collect()),
        &Stmt(ref items, ref span) => TyStmt(
            box annotate(&items, tenv),
            span.clone(),
        ),
        &FieldAccess(ref f_a) => annotate_field_access(f_a, tenv),
        &None => TyNone,
        &Pipes(ref pipes) => annotate_pipes(pipes, tenv),
        &Tuple(ref terms, ref s) => annotate_tuples(terms, s, tenv),
        _ => unimplemented!(),
    }
}

fn annotate_tuples(tup: &[Term], s: &ByteSpan, tenv: &mut TypeEnv) -> TyTerm {
    let (vs, tys)  = tup.iter().map(|i| {
        let tyterm = annotate(i, tenv);
        let ty = tyterm.ty();
        (tyterm, ty)
    })
    .unzip();

    TyTerm::TyTuple(Type::Tuple(tys, s.clone()), vs, s.clone())
}

fn annotate_pipes(pipes: &[Term], tenv: &mut TypeEnv) -> TyTerm {
    let module = tenv.module();
    let mut it = pipes.iter();

    let p0 = it.next().unwrap();
    let mut term0 = annotate(p0, tenv);

    while let Some(t) = it.next() {
        let prev_arg = TyFnAppArg {
            name: Some(String::from("x")),
            arg: Box::new(term0.clone()),
            span: term0.span(),
        };
        let t = match t {
            // this may be `fc1`
            &Term::Ident(ref id, ref span) => TyTerm::TyFnApp(TyFnApp {
                mod_name: Some(
                    tenv.resolve_type(&module, &Alias::Variable(id.clone()))
                        .unwrap()
                        .as_str()
                        .to_owned(),
                ),
                orig_name: Some(id.to_owned()),
                name: Alias::Function("forward".to_owned()),
                arg_ty: tenv.fresh_var(span),
                args: vec![prev_arg],
                ret_ty: tenv.fresh_var(span),
                span: span.clone(),
            }),
            &Term::FnApp(ref fn_app) => {
                let mut typed_fn_app = annotate_fn_app(&fn_app, tenv);
                if typed_fn_app.mod_name.is_none() {
                    // log_softmax(dim=1)
                    typed_fn_app.mod_name = Some(
                        tenv.resolve_type(&module, &typed_fn_app.name)
                            .unwrap()
                            .as_str()
                            .to_owned(),
                    );
                    typed_fn_app.name = Alias::Function("forward".to_owned());
                }
                typed_fn_app.extend_arg(prev_arg);
                TyTerm::TyFnApp(typed_fn_app)
            }
            &Term::FieldAccess(ref f_a) => {
                let typed_f_a = annotate_field_access(&f_a, tenv);
                match typed_f_a {
                    TyTerm::TyFnApp(ref fn_app) => {
                        let mut fn_app = fn_app.clone();
                        fn_app.extend_arg(prev_arg);
                        TyTerm::TyFnApp(fn_app)
                    }
                    _ => panic!("Error: for field access in a pipeline, use parenthesis: f()"),
                }
            }
            &Term::ViewFn(ref v_f) => TyTerm::TyFnApp(annotate_view_fn(&v_f, prev_arg, tenv)),
            _ => unimplemented!(),
        };
        term0 = t.clone();
    }

    term0
}

fn annotate_view_fn(v_fn: &ViewFn, arg: TyFnAppArg, tenv: &mut TypeEnv) -> TyFnApp {
    let module = tenv.module();
    let tsr = tenv.create_tensor(&module, &v_fn.dims, &v_fn.span);
    TyFnApp {
        mod_name: Some("view".to_string()),
        orig_name: None,
        name: Alias::Function("forward".to_owned()),
        arg_ty: args!(arg!("x", arg.arg.ty())),
        ret_ty: tsr.clone(),
        args: vec![arg.clone()],
        span: v_fn.span.clone(),
    }
}

fn annotate_decl(decl: &Decl, tenv: &mut TypeEnv) -> TyDecl {
    use self::Decl::*;
    let ret = match decl {
        &NodeDecl(ref decl) => {
            let module = ModName::Named(decl.name.to_owned());
            tenv.set_module(module.clone());
            let assigns = &decl.defs;
            assigns
                .into_iter()
                .map(|a| tenv.import_node_assign(&module, a))
                .collect::<Vec<()>>();

            tenv.upsert_module(&module);
            // if some dimension alias are not imported, create them
            tenv.import_top_level_ty_sig(&module, &decl.ty_sig.from);
            tenv.import_top_level_ty_sig(&module, &decl.ty_sig.to);

            let ty_sig = annotate_fn_ty_sig(decl.name.to_owned(), "forward".to_owned(), &decl.ty_sig, tenv, &decl.span);
            let mod_ty_sig = Type::Module(
                decl.name.clone(),
                Some(box ty_sig.clone()),
                decl.span.clone(),
            );

            // add current name into global scope
            tenv.add_type(
                &ModName::Global,
                &Alias::Variable(decl.name.to_string()),
                mod_ty_sig.clone(),
            );

            // add "self" into module scope
            tenv.add_type(
                &module,
                &Alias::Variable("self".to_owned()),
                mod_ty_sig.clone(),
            );

            // // add "forward" function into module scope
            // tenv.add_type(&module, "self.forward", ty_sig.clone());

            TyDecl::TyNodeDecl(TyNodeDecl {
                name: decl.name.clone(),
                ty_sig,
                span: decl.span.clone(),
            })
        }
        &WeightsDecl(ref decl) => {
            tenv.set_module(ModName::Named(decl.name.to_owned()));
            TyDecl::TyWeightsDecl(TyWeightsDecl {
                name: decl.name.clone(),
                ty_sig: annotate_fn_ty_sig(decl.name.to_owned(), "forward".to_owned(),&decl.ty_sig, tenv, &decl.span),
                inits: decl.inits
                    .iter()
                    .map(|t| annotate_weights_assign(t, tenv))
                    .collect(),
                span: decl.span.clone(),
            })
        }
        &GraphDecl(ref decl) => {
            tenv.set_module(ModName::Named(decl.name.to_owned()));
            TyDecl::TyGraphDecl(TyGraphDecl {
                name: decl.name.clone(),
                ty_sig: annotate_fn_ty_sig(decl.name.to_owned(), "forward".to_owned(),&decl.ty_sig, tenv, &decl.span),
                fns: decl.fns.iter().map(|f| annotate_fn_decl(f, tenv)).collect(),
                span: decl.span.clone(),
            })
        }
        &UseStmt(ref decl) => {
            // in global scope
            // import names into scope
            // also import module and its associated functions
            for ref name in decl.imported_names.iter() {
                let ty = Type::Module(name.to_string(), None, decl.span.clone());
                tenv.add_type(&ModName::Global, &Alias::Variable(name.to_string()), ty);
                tenv.import_module(&decl.mod_name, &name);
            }

            TyDecl::TyUseStmt(TyUseStmt {
                mod_name: decl.mod_name.clone(),
                imported_names: decl.imported_names.clone(),
                span: decl.span.clone(),
            })
        }
    };
    tenv.set_module(ModName::Global);
    ret
}

fn annotate_fn_ty_sig(modname: String, name: String, sig: &FnTySig, tenv: &mut TypeEnv, span: &ByteSpan) -> Type {
    Type::FUN(
        modname,
        name,
        Box::new(annotate_tensor_ty_sig(&sig.from, tenv, span)),
        Box::new(annotate_tensor_ty_sig(&sig.to, tenv, span)),
        span.clone(),
    )
}

fn annotate_tensor_ty_sig(sig: &TensorTy, tenv: &mut TypeEnv, span: &ByteSpan) -> Type {
    use self::TensorTy::*;
    let module = tenv.module();
    match sig {
        &Generic(ref dims, ref sp) => tenv.create_tensor(&module, dims, sp),
        &Tensor(ref als, ref sp) => tenv.resolve_type(&module, &Alias::Variable(als.clone()))
            .unwrap()
            .clone()
            .with_span(sp),
    }
}

fn annotate_weights_assign(w_assign: &WeightsAssign, tenv: &mut TypeEnv) -> TyWeightsAssign {
    let name = w_assign.name.clone();
    let fn_ty = w_assign
        .clone()
        .mod_sig
        .map(|sig| Box::new(annotate_fn_ty_sig(w_assign.mod_name.to_owned(),"forward".to_owned(), &sig, tenv, &w_assign.span)));

    let module = tenv.module();
    tenv.add_type(
        &module,
        &Alias::Variable(name.to_owned()),
        Type::Module(
            w_assign.mod_name.to_owned(),
            fn_ty.clone(),
            w_assign.span.clone(),
        ),
    );

    let fn_args: Vec<TyFnAppArg> = w_assign
        .fn_args
        .iter()
        .map(|a| annotate_fn_app_arg(a, tenv))
        .collect();

    tenv.add_init(&module, &name, fn_args.clone());

    TyWeightsAssign {
        name: name,
        mod_name: w_assign.mod_name.clone(),
        fn_name: w_assign.fn_name.clone(),
        arg_ty: fn_args.to_ty(&w_assign.span),
        fn_args: fn_args,
        span: w_assign.span.clone(),
    }
}

fn annotate_fn_app_arg(call: &FnAppArg, tenv: &mut TypeEnv) -> TyFnAppArg {
    let tyterm = Box::new(annotate(&call.arg, tenv));
    // println!("{}", tyterm);
    TyFnAppArg {
        name: Some(call.name.clone()),
        arg: tyterm,
        span: call.span.clone(),
    }
}

fn annotate_fn_app(fn_app: &FnApp, tenv: &mut TypeEnv) -> TyFnApp {
    let FnApp {
        ref name,
        ref args,
        ref span,
    } = fn_app;
    let t_args: Vec<TyFnAppArg> = args.iter().map(|a| annotate_fn_app_arg(&a, tenv)).collect();
    let arg_ty = t_args.to_ty(&fn_app.span);
    TyFnApp {
        mod_name: None,
        orig_name: Some(name.to_owned()),
        name: Alias::Variable(name.to_owned()),
        arg_ty: arg_ty,
        args: t_args,
        ret_ty: tenv.fresh_var(&span),
        span: span.clone(),
    }
}

fn annotate_fn_decl(f: &FnDecl, tenv: &mut TypeEnv) -> TyFnDecl {
    let module = tenv.module();
    tenv.push_scope(&module);
    let mod_ty = tenv.resolve_type(
        &ModName::Global,
        &Alias::Variable(module.as_str().to_owned()),
    ).unwrap().clone().with_span(&f.span);

    let mut decl = TyFnDecl {
        name: Alias::Function(f.name.clone()),
        fn_params: f.fn_params
            .iter()
            .map(|p| annotate_fn_decl_param(p, tenv))
            .collect(),
        // fn_ty: tenv.fresh_var(),
        fn_ty: tenv.fresh_var(&f.span),
        func_block: Box::new(TyTerm::TyNone),  // same here
        span: f.span.clone(),
    };

    match f.name.as_str() {
        // special function signatures
        "new" => {
            decl.fn_ty = Type::FUN(
                module.as_str().to_owned(),
                "new".to_owned(),
                box decl.fn_params.to_ty(&f.span),
                box mod_ty.clone(),
                f.span.clone()
            );
        }
        "forward" => {
            if decl.fn_params.len() == 0 {
                panic!("forward(x) must have at least 1 param");
            }
            let name0 = decl.fn_params[0].name.clone();
            if let Type::Module(ref modn, Some(box Type::FUN(_,_,ref p, ref r, _)), _) = mod_ty.clone() {
                let ty_sig = *p.clone();

                // type the first parameter
                decl.fn_params = vec![TyFnDeclParam {
                    name: String::from(name0.clone()),
                    ty: ty_sig.clone(),
                    span: f.span.clone(),
                }];

                // override the old first argument which is []
                unsafe {
                    tenv.add_type_allow_dup(&module, &Alias::Variable(name0.clone()), ty_sig);
                }

                // // type the function return parameter
                decl.fn_ty = Type::FUN(
                    modn.to_owned(),
                    "forward".to_owned(),
                    box decl.fn_params.to_ty(&f.span),
                    r.clone(),
                    f.span.clone()
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

    decl.func_block = Box::new(annotate(&f.func_block, tenv));

    tenv.pop_scope(&module);

    // insert this function into typeenv
    tenv.add_type(
        &module,
        &Alias::Function(f.name.clone()),
        decl.fn_ty.clone(),
    );

    decl
}

fn annotate_fn_decl_param(p: &FnDeclParam, tenv: &mut TypeEnv) -> TyFnDeclParam {
    let module = tenv.module();
    let name = p.name.clone();
    let ty = tenv.resolve_tensor(&module, &p.ty_sig, &p.span);
    tenv.add_type(&module, &Alias::Variable(name.clone()), ty.clone());
    let ret = TyFnDeclParam {
        name: name,
        ty: ty,
        span: p.span.clone(),
    };
    ret
}

fn annotate_field_access(f_a: &FieldAccess, tenv: &mut TypeEnv) -> TyTerm {
    let module = tenv.module();
    match module {
        ModName::Global => panic!("Cannot use field access in global scope"),
        ModName::Named(ref _mod_name) => match f_a.func_call {
            None => TyTerm::TyFieldAccess(TyFieldAccess {
                mod_name: f_a.mod_name.clone(),
                field_name: f_a.field_name.clone(),
                ty: tenv.fresh_var(&f_a.span),
                span: f_a.span.clone(),
            }),
            Some(ref v) => {
                let args: Vec<TyFnAppArg> =
                    v.iter().map(|arg| annotate_fn_app_arg(arg, tenv)).collect();
                let args_ty = args.to_ty(&f_a.span);
                TyTerm::TyFnApp(TyFnApp {
                    mod_name: Some(f_a.mod_name.clone()),
                    orig_name: Some(f_a.mod_name.clone()),
                    name: Alias::Function(f_a.field_name.clone()),
                    arg_ty: args_ty,
                    args,
                    ret_ty: tenv.fresh_var(&f_a.span),
                    span: f_a.span.clone(),
                })
            }
        },
    }
}
