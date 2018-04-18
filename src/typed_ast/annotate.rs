use parser::term::{Decl, FieldAccess, FnApp, FnAppArg, FnDecl, FnDeclParam, FnTySig, TensorTy,
                   Term, ViewFn, WeightsAssign};
use typed_ast::Type;
use typed_ast::type_env::{ModName, TypeEnv};
use typed_ast::typed_term::{Ty, ArgsVecInto};
use typed_ast::typed_term::{TyDecl, TyFieldAccess, TyFnApp, TyFnAppArg, TyFnDecl, TyFnDeclParam,
                            TyGraphDecl, TyNodeDecl, TyTerm, TyUseStmt, TyViewFn, TyWeightsAssign,
                            TyWeightsDecl};

pub fn annotate(term: &Term, tenv: &mut TypeEnv) -> TyTerm {
    use self::Term::*;
    use self::TyTerm::*;
    // println!("{:#?}", term);
    let module = tenv.module();
    match term {
        Ident(ref id) => TyTerm::TyIdent(tenv.resolve_type(&module, id).unwrap(), id.to_owned()),
        &Program(ref decls) => TyProgram(decls.iter().map(|d| annotate_decl(d, tenv)).collect()),
        &Expr { ref items } => {
            let ret = Box::new(annotate(&items, tenv));
            TyExpr {
                ty: ret.ty(),
                items: ret,
            }
        }
        &Integer(i) => TyInteger(Type::INT, i),
        &Float(i) => TyFloat(Type::FLOAT, i),
        &Block { ref stmts, ref ret } => {
            let module = tenv.module();
            tenv.push_scope(&module);
            let ret = TyBlock {
                stmts: Box::new(annotate(&stmts, tenv)),
                ret: Box::new(annotate(&ret, tenv)),
            };
            tenv.pop_scope(&module);
            ret
        }
        &List(ref stmts) => TyList(stmts.iter().map(|s| annotate(&s, tenv)).collect()),
        &Stmt { ref items } => TyStmt {
            items: Box::new(annotate(&items, tenv)),
        },
        &FieldAccess(ref f_a) => annotate_field_access(f_a, tenv),
        &None => TyNone,
        &Pipes(ref pipes) => annotate_pipes(pipes, tenv),
        _ => unimplemented!(),
    }
}

fn annotate_pipes(pipes: &[Term], tenv: &mut TypeEnv) -> TyTerm {
    let module = tenv.module();
    let mut it = pipes.iter();

    let p0 = it.next().unwrap();
    let mut term0 = annotate(p0, tenv);

    while let Some(t) = it.next() {
        let prev_arg = TyFnAppArg {
            name: Some(String::from("x")),
            ty: tenv.fresh_var(),
            arg: Box::new(term0),
        };
        let t = match t {
            // this may be `fc1`
            &Term::Ident(ref id) => TyTerm::TyFnApp(TyFnApp {
                mod_name: Some(tenv.resolve_type(&module, id).unwrap().as_str().to_owned()),
                orig_name: id.to_owned(),
                name: "self.forward".to_owned(),
                arg_ty: tenv.fresh_var(),
                args: vec![prev_arg],
                ret_ty: tenv.fresh_var(),
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
                    typed_fn_app.name = "self.forward".to_owned();
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
            &Term::ViewFn(ref v_f) => TyTerm::TyViewFn(annotate_view_fn(&v_f, prev_arg, tenv)),
            _ => unimplemented!(),
        };
        term0 = t.clone();
    }

    term0
}

fn annotate_view_fn(v_fn: &ViewFn, arg: TyFnAppArg, tenv: &mut TypeEnv) -> TyViewFn {
    let module = tenv.module();
    let tsr = tenv.create_tensor(&module, &v_fn.dims);
    TyViewFn { ty: tsr, arg }
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
            let ty_sig = annotate_fn_ty_sig(&decl.ty_sig, tenv);
            let mod_ty_sig = Type::Module(decl.name.clone(), Some(box ty_sig.clone()));

            // add current name into global scope
            tenv.add_type(&ModName::Global, &decl.name, mod_ty_sig.clone());

            // add "self" into module scope
            tenv.add_type(&module, "self", mod_ty_sig.clone());

            // // add "forward" function into module scope
            // tenv.add_type(&module, "self.forward", ty_sig.clone());

            TyDecl::TyNodeDecl(TyNodeDecl {
                name: decl.name.clone(),
                ty_sig,
            })
        }
        &WeightsDecl(ref decl) => {
            tenv.set_module(ModName::Named(decl.name.to_owned()));
            // TODO? also import global symbols into scope...
            TyDecl::TyWeightsDecl(TyWeightsDecl {
                name: decl.name.clone(),
                ty_sig: annotate_fn_ty_sig(&decl.ty_sig, tenv),
                inits: decl.inits
                    .iter()
                    .map(|t| annotate_weights_assign(t, tenv))
                    .collect(),
            })
        }
        &GraphDecl(ref decl) => {
            tenv.set_module(ModName::Named(decl.name.to_owned()));
            TyDecl::TyGraphDecl(TyGraphDecl {
                name: decl.name.clone(),
                ty_sig: annotate_fn_ty_sig(&decl.ty_sig, tenv),
                fns: decl.fns.iter().map(|f| annotate_fn_decl(f, tenv)).collect(),
            })
        }
        &UseStmt(ref decl) => {
            // in global scope
            // import names into scope
            // also import module and its associated functions
            for ref name in decl.imported_names.iter() {
                let ty = Type::Module(name.to_string(), None);
                tenv.add_type(&ModName::Global, name, ty);
                tenv.import_module(&decl.mod_name, &name);
            }

            TyDecl::TyUseStmt(TyUseStmt {
                mod_name: decl.mod_name.clone(),
                imported_names: decl.imported_names.clone(), // ...
            })
        }
    };
    tenv.set_module(ModName::Global);
    ret
}

fn annotate_fn_ty_sig(sig: &FnTySig, tenv: &mut TypeEnv) -> Type {
    Type::FUN(
        Box::new(annotate_tensor_ty_sig(&sig.from, tenv)),
        Box::new(annotate_tensor_ty_sig(&sig.to, tenv)),
    )
}

fn annotate_tensor_ty_sig(sig: &TensorTy, tenv: &mut TypeEnv) -> Type {
    use self::TensorTy::*;
    let module = tenv.module();
    match sig {
        &Generic(ref dims) => tenv.create_tensor(&module, dims),
        &TyAlias(ref als) => tenv.resolve_type(&module, als).unwrap().clone(),
    }
}

fn annotate_weights_assign(w_assign: &WeightsAssign, tenv: &mut TypeEnv) -> TyWeightsAssign {
    let name = w_assign.name.clone();
    let fn_ty = w_assign
        .clone()
        .mod_sig
        .map(|sig| Box::new(annotate_fn_ty_sig(&sig, tenv)));

    let module = tenv.module();
    tenv.add_type(
        &module,
        &name,
        Type::Module(w_assign.mod_name.to_owned(), fn_ty.clone()),
    );

    let fn_args: Vec<TyFnAppArg> = w_assign
        .fn_args
        .iter()
        .map(|a| annotate_fn_app_arg(a, tenv))
        .collect();
    
    tenv.add_init(&module, &name, fn_args.clone());

    TyWeightsAssign {
        name: name,
        ty: tenv.fresh_var(),
        mod_name: w_assign.mod_name.clone(),
        fn_name: w_assign.fn_name.clone(),
        arg_ty: fn_args.to_ty(),
        fn_args: fn_args,
    }
}

fn annotate_fn_app_arg(call: &FnAppArg, tenv: &mut TypeEnv) -> TyFnAppArg {
    let tyterm = Box::new(annotate(&call.arg, tenv));
    // println!("{}", tyterm);
    TyFnAppArg {
        name: Some(call.name.clone()),
        ty: tyterm.ty(),
        arg: tyterm,
    }
}

fn annotate_fn_app(fn_app: &FnApp, tenv: &mut TypeEnv) -> TyFnApp {
    let FnApp { ref name, ref args } = fn_app;
    let t_args: Vec<TyFnAppArg> = args.iter().map(|a| annotate_fn_app_arg(&a, tenv)).collect();
    let arg_ty = t_args.to_ty();
    TyFnApp {
        mod_name: None,
        orig_name: name.to_owned(),
        name: name.to_owned(),
        arg_ty: arg_ty,
        args: t_args,
        ret_ty: tenv.fresh_var(),
    }
}

fn annotate_fn_decl(f: &FnDecl, tenv: &mut TypeEnv) -> TyFnDecl {
    let module = tenv.module();
    tenv.push_scope(&module);
    let mod_ty = tenv.resolve_type(&ModName::Global, &module.as_str())
        .unwrap()
        .clone();

    let mut decl = TyFnDecl {
        name: f.name.clone(),
        fn_params: f.fn_params
            .iter()
            .map(|p| annotate_fn_decl_param(p, tenv))
            .collect(),
        fn_ty: tenv.fresh_var(),
        param_ty: tenv.fresh_var(),
        return_ty: Type::Unit,                // put a placeholder here
        func_block: Box::new(TyTerm::TyNone), // same here
    };

    match f.name.as_str() {
        // special function signatures
        "new" => {
            decl.return_ty = mod_ty;
            decl.fn_ty = tenv.resolve_type(&module, "self").unwrap().clone();
        }
        "forward" => {
            if decl.fn_params.len() == 0 {
                panic!("self.forward(x) must have at least 1 param");
            }
            let name0 = decl.fn_params[0].name.clone();
            if let Type::Module(_, Some(box Type::FUN(ref p, ref r))) = mod_ty.clone() {
                let ty_sig = *p.clone();
                // type the first parameter
                decl.fn_params = vec![
                    TyFnDeclParam {
                        name: String::from(name0),
                        ty_sig,
                    },
                ];
                // type the function return parameter
                decl.return_ty = *r.clone();
                // type the function itself...
                decl.fn_ty = Type::FUN(p.clone(), r.clone());
            } else {
                panic!("Signature of module is incorrect!");
            };
        }
        _ => {
            // ... todo: add parse argument type...
            decl.return_ty = tenv.resolve_tensor(&module, &f.return_ty);
        }
    };

    decl.func_block = Box::new(annotate(&f.func_block, tenv));

    tenv.pop_scope(&module);

    // insert this function into typeenv
    tenv.add_type(&module, &format!("self.{}", f.name), decl.fn_ty.clone());

    decl
}

fn annotate_fn_decl_param(p: &FnDeclParam, tenv: &mut TypeEnv) -> TyFnDeclParam {
    let module = tenv.module();
    let name = p.name.clone();
    let ty = tenv.fresh_var(); // ... todo!!!!
    tenv.add_type(&module, &name, ty.clone());
    let ret = TyFnDeclParam {
        name: name,
        ty_sig: ty,
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
                ty: tenv.fresh_var(),
            }),
            Some(ref v) => {
                let args: Vec<TyFnAppArg> =
                    v.iter().map(|arg| annotate_fn_app_arg(arg, tenv)).collect();
                let arg_ty = args.iter()
                    .map(|t_arg| Type::FnArg(t_arg.name.clone(), box t_arg.ty.clone()))
                    .collect();
                TyTerm::TyFnApp(TyFnApp {
                    mod_name: Some(f_a.mod_name.clone()),
                    orig_name: f_a.mod_name.clone(),
                    name: format!("self.{}", f_a.field_name),
                    arg_ty: Type::FnArgs(arg_ty),
                    args,
                    ret_ty: tenv.fresh_var(),
                })
            }
        },
    }
}
