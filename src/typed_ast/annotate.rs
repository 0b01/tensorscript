use parser::term::{Term, Decl, FnTySig, TensorTy, WeightsAssign, FnAppArg, FnDecl, FnDeclParam, FieldAccess, FnApp, ViewFn};
use typed_ast::typed_term::{TyTerm, TyDecl, TyNodeDecl, TyWeightsDecl, TyWeightsAssign, TyFnAppArg, TyGraphDecl, TyFnDecl, TyFnDeclParam, TyFieldAccess, TyFnApp, TyViewFn, TyUseStmt};
use typed_ast::type_env::{ModName, TypeEnv};
use typed_ast::Type;

pub fn annotate(term: &Term, tenv: &mut TypeEnv) -> TyTerm {
    use self::Term::*;
    use self::TyTerm::*;
    // println!("{:#?}", term);
    let module = tenv.module().clone();
    match term {
        Ident(ref id) => TyTerm::TyIdent(tenv.resolve_alias(&module, id).unwrap(), id.to_owned()),
        &Program(ref decls) => TyProgram(
            decls
                .iter()
                .map(|d| annotate_decl(d, tenv))
                .collect()
        ),
        &Expr { ref items } => TyExpr {
            items: Box::new(annotate(&items, tenv)),
            ty: tenv.fresh_var(),
        },
        &Integer(i) => TyInteger(tenv.fresh_var(), i),
        &Float(i) => TyFloat(tenv.fresh_var(), i),
        &Block {ref stmts, ref ret } => {
            let module = tenv.module().clone();
            tenv.push_scope(&module);
            let ret = TyBlock {
                stmts: Box::new(annotate(&stmts, tenv)),
                ret: Box::new(annotate(&ret, tenv))
            };
            tenv.pop_scope(&module);
            ret
        },
        &List(ref stmts) => TyList(stmts.iter()
            .map(|s|annotate(&s, tenv))
            .collect()),
        &Stmt{ref items} => TyStmt {
            items: Box::new(annotate(&items, tenv))
        },
        &FieldAccess(ref f_a) => annotate_field_access(f_a, tenv),
        &None => TyNone,
        &Pipes(ref pipes) => annotate_pipes(pipes, tenv),
        _ => unimplemented!(),
    }
}

fn annotate_pipes(pipes: &[Term], tenv: &mut TypeEnv) -> TyTerm {
    let mut it = pipes.iter();
    
    let p0 = it.next().unwrap();
    let mut term0 = annotate(p0, tenv);

    while let Some(t) = it.next() {
        let prev_arg = TyFnAppArg { name: String::from("x"), arg: Box::new(term0) };
        let t = match t {
            &Term::Ident(ref id) => {
                TyTerm::TyFnApp(TyFnApp {
                    mod_name: None,
                    name: id.to_owned(),
                    args: vec![prev_arg],
                    ret_ty: tenv.fresh_var(),
                })
            },
            &Term::FnApp(ref fn_app) => {
                let mut typed_fn_app = annotate_fn_app(&fn_app, tenv);
                TyTerm::TyFnApp(typed_fn_app.extend_arg(prev_arg))
            },
            &Term::FieldAccess(ref f_a) => {
                let typed_f_a = annotate_field_access(&f_a, tenv);
                match typed_f_a {
                    TyTerm::TyFnApp(ref fn_app) => TyTerm::TyFnApp(fn_app.clone().extend_arg(prev_arg)),
                    _ => panic!("Error: for field access in a pipeline, use parenthesis: f()"),
                }
            },
            &Term::ViewFn(ref v_f) => TyTerm::TyViewFn(annotate_view_fn(&v_f, prev_arg, tenv)),
            _ => unimplemented!(),
        };
        term0 = t.clone();
    }

    term0
}

fn annotate_view_fn(v_fn: &ViewFn, arg: TyFnAppArg, tenv: &mut TypeEnv) -> TyViewFn {
    let module = tenv.module().clone();
    let tsr = tenv.create_tensor(&module, &v_fn.dims);
    TyViewFn {
        ty: tsr,
        arg,
    }
}

fn annotate_decl(decl: &Decl, tenv: &mut TypeEnv) -> TyDecl {
    use self::Decl::*;
    let ret = match decl {
        &NodeDecl(ref decl) => {
            let module = ModName::Named(decl.name.to_owned());
            tenv.set_module(module.clone());
            let assigns = &decl.defs;
            assigns.into_iter()
                .map(|a| tenv.import_node_assign(&module, a))
                .collect::<Vec<()>>();
            let ty_sig = annotate_fn_ty_sig(&decl.ty_sig, tenv);
            tenv.add_alias(&ModName::Global, &decl.name, ty_sig.clone());
            TyDecl::TyNodeDecl(TyNodeDecl {
                name: decl.name.clone(),
                ty_sig,
            })
        },
        &WeightsDecl(ref decl) => {
            tenv.set_module(ModName::Named(decl.name.to_owned()));
            TyDecl::TyWeightsDecl(TyWeightsDecl {
                name: decl.name.clone(),
                ty_sig: annotate_fn_ty_sig(&decl.ty_sig, tenv),
                inits: decl.inits.iter()
                    .map(|t| annotate_weights_assign(t, tenv))
                    .collect()
            })
        },
        &GraphDecl(ref decl) => {
            tenv.set_module(ModName::Named(decl.name.to_owned()));
            TyDecl::TyGraphDecl(TyGraphDecl {
                name: decl.name.clone(),
                ty_sig: annotate_fn_ty_sig(&decl.ty_sig, tenv),
                fns: decl.fns.iter()
                    .map(|f| annotate_fn_decl(f, tenv))
                    .collect(),
            })

        },
        &UseStmt(ref decl) => {
            // in global scope
            // import names into scope
            for ref name in decl.imported_names.iter() {
                let ty = tenv.fresh_var(); // ...
                tenv.add_alias(&ModName::Global, &name, ty);
            }

            TyDecl::TyUseStmt(TyUseStmt {
                mod_name: decl.mod_name.clone(),
                imported_names: decl.imported_names.clone(), // ...
            })
        },
    };
    tenv.set_module(ModName::Global);
    ret
}

fn annotate_fn_ty_sig(sig: &FnTySig, tenv: &mut TypeEnv) -> Type {
    Type::Fun {
        param_ty: Box::new(annotate_tensor_ty_sig(&sig.from, tenv)),
        return_ty: Box::new(annotate_tensor_ty_sig(&sig.to, tenv)),
    }
}

fn annotate_tensor_ty_sig(sig: &TensorTy, tenv: &mut TypeEnv) -> Type {
    use self::TensorTy::*;
    let module = tenv.module().clone();
    match sig {
        &Generic(ref dims) => tenv.create_tensor(&module, dims),
        &TyAlias(ref als) => tenv.resolve_alias(&module, als).unwrap().clone()
    }
}

fn annotate_weights_assign(w_assign: &WeightsAssign, tenv: &mut TypeEnv) -> TyWeightsAssign {
    let name = w_assign.name.clone();
    let fn_ty = tenv.fresh_var();
    let module = tenv.module().clone();
    tenv.add_alias(&module, &name, fn_ty.clone());

    TyWeightsAssign {
        name: name,
        ty: tenv.fresh_var(),
        mod_name: w_assign.mod_name.clone(),
        fn_ty: fn_ty,
        fn_name: w_assign.fn_name.clone(),
        fn_args: w_assign.fn_args.iter()
            .map(|a| annotate_fn_app_arg(a, tenv))
            .collect(),
    }
}

fn annotate_fn_app_arg(call: &FnAppArg, tenv: &mut TypeEnv) -> TyFnAppArg {
    TyFnAppArg {
        name: call.name.clone(),
        arg: Box::new(annotate(&call.arg, tenv)),
    }
}

fn annotate_fn_app(fn_app: &FnApp, tenv: &mut TypeEnv) -> TyFnApp {
    let FnApp {ref name, ref args } = fn_app;
    TyFnApp {
        mod_name: None,
        name: name.to_owned(),
        args: args.iter()
            .map(|a| annotate_fn_app_arg(&a, tenv))
            .collect(),
        ret_ty: tenv.fresh_var(),
    }
}


fn annotate_fn_decl(f: &FnDecl, tenv: &mut TypeEnv) -> TyFnDecl {
    let module = tenv.module().clone();
    tenv.push_scope(&module);
    let mod_ty = tenv.resolve_alias(&ModName::Global, &module.as_str()).unwrap().clone();

    let mut decl = TyFnDecl {
        name: f.name.clone(),
        fn_params: f.fn_params.iter()
            .map(|p| annotate_fn_decl_param(p, tenv))
            .collect(),
        return_ty: Type::Unit, // put a placeholder here
        func_block: Box::new(TyTerm::TyNone), // same here
    };

    match f.name.as_str() {
        // special function signatures
        "new" => {
            decl.return_ty = mod_ty;
        },
        "forward" => {
            if decl.fn_params.len() == 0 { panic!("Forward must have at least 1 param"); }
            let name0 = decl.fn_params[0].name.clone();

            if let Type::Fun { ref param_ty, ref return_ty } = mod_ty {
                let ty_sig = *param_ty.clone();
                decl.fn_params = vec![TyFnDeclParam {name: String::from(name0), ty_sig }];
                decl.return_ty = *return_ty.clone();
            } else {
                panic!("Signature of module is incorrect!");
            };

        },
        _ => {
            decl.return_ty = tenv.resolve_tensor(&module, &f.return_ty);
        }
    };

    decl.func_block = Box::new(annotate(&f.func_block, tenv));

    tenv.pop_scope(&module);
    decl
}

fn annotate_fn_decl_param(p: &FnDeclParam, tenv: &mut TypeEnv) -> TyFnDeclParam {
    let module = tenv.module().clone();
    let name = p.name.clone();
    let ty = tenv.fresh_var();
    tenv.add_alias(&module, &name, ty.clone());
    let ret = TyFnDeclParam {
        name: name,
        ty_sig: ty,
    };
    ret
}

fn annotate_field_access(f_a: &FieldAccess, tenv: &mut TypeEnv) -> TyTerm {
    let module = tenv.module().clone();
    match module {
        ModName::Global =>
            panic!("Cannot use field access in global scope"),
        ModName::Named(ref _mod_name) => {
            match f_a.func_call {
                None => {
                    TyTerm::TyFieldAccess(TyFieldAccess {
                        mod_name: f_a.mod_name.clone(),
                        field_name: f_a.field_name.clone(),
                        ty: tenv.fresh_var(),
                    })
                },
                Some(ref v) => {
                    let args = v.iter()
                            .map(|arg| annotate_fn_app_arg(arg, tenv))
                            .collect();
                    TyTerm::TyFnApp(TyFnApp {
                        mod_name: Some(f_a.mod_name.clone()),
                        name: f_a.field_name.clone(),
                        args,
                        ret_ty: tenv.fresh_var(),
                    })
                }
            }
        }
    }
}