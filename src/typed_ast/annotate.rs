use parser::term::{Term, Decl, FnTySig, TensorTy, WeightsAssign, FnAppArg, FnDecl, FnDeclParam, FieldAccess, FnApp, ViewFn};
use typed_ast::typed_term::{TypedTerm, TypedDecl, TypedNodeDecl, TypedWeightsDecl, TypedWeightsAssign, TypedFnAppArg, TypedGraphDecl, TypedFnDecl, TypedFnDeclParam, TypedFieldAccess, TypedFnApp, TypedViewFn};
use typed_ast::type_env::{ModName, TypeEnv};
use typed_ast::Type;

pub fn annotate(term: &Term, tenv: &mut TypeEnv) -> TypedTerm {
    use self::Term::*;
    use self::TypedTerm::*;
    // println!("{:#?}", term);
    let module = tenv.module().clone();
    match term {
        Ident(ref id) => TypedTerm::TypedIdent(tenv.resolve_alias(&module, id).unwrap(), id.to_owned()),
        &Program(ref decls) => TypedProgram(
            decls
                .iter()
                .map(|d| annotate_decl(d, tenv))
                .collect()
        ),
        &Expr { ref items } => TypedExpr {
            items: Box::new(annotate(&items, tenv)),
            ty: tenv.fresh_var(),
        },
        &Integer(i) => TypedInteger(tenv.fresh_var(), i),
        &Float(i) => TypedFloat(tenv.fresh_var(), i),
        &Block {ref stmts, ref ret } => {
            let module = tenv.module().clone();
            tenv.push_scope(&module);
            let ret = TypedBlock {
                stmts: Box::new(annotate(&stmts, tenv)),
                ret: Box::new(annotate(&ret, tenv))
            };
            tenv.pop_scope(&module);
            ret
        },
        &List(ref stmts) => TypedList(stmts.iter()
            .map(|s|annotate(&s, tenv))
            .collect()),
        &Stmt{ref items} => TypedStmt {
            items: Box::new(annotate(&items, tenv))
        },
        &FieldAccess(ref f_a) => annotate_field_access(f_a, tenv),
        &None => TypedNone,
        &Pipes(ref pipes) => annotate_pipes(pipes, tenv),
        _ => unimplemented!(),
    }
}

fn annotate_pipes(pipes: &[Term], tenv: &mut TypeEnv) -> TypedTerm {
    let mut it = pipes.iter();
    
    let p0 = it.next().unwrap();
    let mut term0 = annotate(p0, tenv);

    while let Some(t) = it.next() {
        let prev_arg = TypedFnAppArg { name: String::from("x"), arg: Box::new(term0) };
        let t = match t {
            &Term::Ident(ref id) => {
                TypedTerm::TypedFnApp(TypedFnApp {
                    mod_name: None,
                    name: id.to_owned(),
                    args: vec![prev_arg],
                    ret_ty: tenv.fresh_var(),
                })
            },
            &Term::FnApp(ref fn_app) => {
                let mut typed_fn_app = annotate_fn_call(&fn_app, tenv);
                TypedTerm::TypedFnApp(typed_fn_app.extend_arg(prev_arg))
            },
            &Term::FieldAccess(ref f_a) => {
                let typed_f_a = annotate_field_access(&f_a, tenv);
                match typed_f_a {
                    TypedTerm::TypedFnApp(ref fn_app) => TypedTerm::TypedFnApp(fn_app.clone().extend_arg(prev_arg)),
                    _ => panic!("Error: for field access in a pipeline, use parenthesis: f()"),
                }
            },
            &Term::ViewFn(ref v_f) => TypedTerm::TypedViewFn(annotate_view_fn(&v_f, prev_arg, tenv)),
            _ => unimplemented!(),
        };
        term0 = t.clone();
    }

    term0
}

fn annotate_view_fn(v_fn: &ViewFn, arg: TypedFnAppArg, tenv: &mut TypeEnv) -> TypedViewFn {
    let module = tenv.module().clone();
    let tsr = tenv.make_tensor(&module, &v_fn.dims);
    TypedViewFn {
        ty: tsr,
        arg,
    }
}

fn annotate_decl(decl: &Decl, tenv: &mut TypeEnv) -> TypedDecl {
    use self::Decl::*;
    let ret = match decl {
        &NodeDecl(ref decl) => {
            let module = ModName::Named(decl.name.to_owned());
            tenv.set_module(module.clone());
            let assigns = &decl.defs;
            assigns.into_iter()
                .map(|a| tenv.import_node_assign(&module, a))
                .collect::<Vec<()>>();

            TypedDecl::TypedNodeDecl(TypedNodeDecl {
                name: decl.name.clone(),
                ty_sig: annotate_fn_ty_sig(&decl.ty_sig, tenv),
            })
        },
        &WeightsDecl(ref decl) => {
            tenv.set_module(ModName::Named(decl.name.to_owned()));
            TypedDecl::TypedWeightsDecl(TypedWeightsDecl {
                name: decl.name.clone(),
                ty_sig: annotate_fn_ty_sig(&decl.ty_sig, tenv),
                inits: decl.inits.iter()
                    .map(|t| annotate_weights_assign(t, tenv))
                    .collect()
            })
        },
        &GraphDecl(ref decl) => {
            tenv.set_module(ModName::Named(decl.name.to_owned()));
            TypedDecl::TypedGraphDecl(TypedGraphDecl {
                name: decl.name.clone(),
                ty_sig: annotate_fn_ty_sig(&decl.ty_sig, tenv),
                fns: decl.fns.iter()
                    .map(|f| annotate_fn_decl(f, tenv))
                    .collect(),
            })

        },
        // &UseStmt(decl) => {

        // },
        _ => unimplemented!(),
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
        &Generic(ref dims) => tenv.make_tensor(&module, dims),
        &TyAlias(ref als) => tenv.resolve_alias(&module, als).unwrap().clone()
    }
}

fn annotate_weights_assign(w_assign: &WeightsAssign, tenv: &mut TypeEnv) -> TypedWeightsAssign {
    let name = w_assign.name.clone();
    let fn_ty = tenv.fresh_var();
    let module = tenv.module().clone();
    tenv.add_alias(&module, &name, fn_ty.clone());

    TypedWeightsAssign {
        name: name,
        ty: tenv.fresh_var(),
        mod_name: w_assign.mod_name.clone(),
        fn_ty: fn_ty,
        fn_name: w_assign.fn_name.clone(),
        fn_args: w_assign.fn_args.iter()
            .map(|a| annotate_fn_call_arg(a, tenv))
            .collect(),
    }
}

fn annotate_fn_call_arg(call: &FnAppArg, tenv: &mut TypeEnv) -> TypedFnAppArg {
    TypedFnAppArg {
        name: call.name.clone(),
        arg: Box::new(annotate(&call.arg, tenv)),
    }
}

fn annotate_fn_call(fn_app: &FnApp, tenv: &mut TypeEnv) -> TypedFnApp {
    let FnApp {ref name, ref args } = fn_app;
    TypedFnApp {
        mod_name: None,
        name: name.to_owned(),
        args: args.iter()
            .map(|a| annotate_fn_call_arg(&a, tenv))
            .collect(),
        ret_ty: tenv.fresh_var(),
    }
}


fn annotate_fn_decl(f: &FnDecl, tenv: &mut TypeEnv) -> TypedFnDecl {
    let ret = TypedFnDecl {
        name: f.name.clone(),
        fn_params: f.fn_params.iter()
            .map(|p| annotate_fn_decl_param(p, tenv))
            .collect(),
        return_ty: tenv.fresh_var(),
        func_block: Box::new(annotate(&f.func_block, tenv)),
    };
    ret
}

fn annotate_fn_decl_param(p: &FnDeclParam, tenv: &mut TypeEnv) -> TypedFnDeclParam {
    let module = tenv.module().clone();
    let name = p.name.clone();
    let ty = tenv.fresh_var();
    tenv.add_alias(&module, &name, ty.clone());
    TypedFnDeclParam {
        name: name,
        ty_sig: ty,
    }
}

fn annotate_field_access(f_a: &FieldAccess, tenv: &mut TypeEnv) -> TypedTerm {
    let module = tenv.module().clone();
    match module {
        ModName::Global =>
            panic!("Cannot use field access in global scope"),
        ModName::Named(ref _node_name) => {
            match f_a.func_call {
                None => {
                    TypedTerm::TypedFieldAccess(TypedFieldAccess {
                        mod_name: f_a.mod_name.clone(),
                        field_name: f_a.field_name.clone(),
                        ty: tenv.fresh_var(),
                    })
                },
                Some(ref v) => {
                    let args = v.iter()
                            .map(|arg| annotate_fn_call_arg(arg, tenv))
                            .collect();
                    TypedTerm::TypedFnApp(TypedFnApp {
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