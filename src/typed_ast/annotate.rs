use parser::term::{Term, Decl, FnTySig, TensorTy, WeightsAssign, FnCallArg};
use typed_ast::typed_term::{TypedTerm, TypedDecl, TypedNodeDecl, TypedWeightsDecl, TypedWeightsAssign, TypedFnCallArg};
use typed_ast::type_env::TypeEnv;
use typed_ast::Type;

pub fn annotate(term: Term, tenv: &mut TypeEnv) -> TypedTerm {
    use self::Term::*;
    use self::TypedTerm::*;
    match term {
        Program(decls) => TypedProgram(
            decls
                .iter()
                .map(|d| annotate_decl(d, tenv))
                .collect()
        ),
        Expr { items } => TypedExpr {
            items: Box::new(annotate(*items, tenv)),
            ty: tenv.fresh_var(),
        },
        Integer(i) => TypedInteger(tenv.fresh_var(), i),
        Float(i) => TypedFloat(tenv.fresh_var(), i),
        _ => unimplemented!(),
    }
}

fn annotate_decl(decl: &Decl, tenv: &mut TypeEnv) -> TypedDecl {
    use self::Decl::*;
    println!("{:?}", decl);
    match decl {
        &NodeDecl(ref decl) => {
            let assigns = &decl.defs;
            assigns.into_iter()
                .map(|a| tenv.import_node_assign(&decl.name, a))
                .collect::<Vec<()>>();

            TypedDecl::TypedNodeDecl(TypedNodeDecl {
                name: decl.name.clone(),
                ty_sig: annotate_fn_ty_sig(&decl.name, &decl.ty_sig, tenv),
            })
        },
        &WeightsDecl(ref decl) => {
            TypedDecl::TypedWeightsDecl(TypedWeightsDecl {
                name: decl.name.clone(),
                ty_sig: annotate_fn_ty_sig(&decl.name, &decl.ty_sig, tenv),
                inits: decl.inits.iter()
                    .map(|t| annotate_weights_assign(&decl.name, t, tenv))
                    .collect()
            })
        },
        // &GraphDecl(decl) => {

        // },
        // &UseStmt(decl) => {

        // },
        _ => unimplemented!(),
    }
}

fn annotate_fn_ty_sig(node_name: &str, sig: &FnTySig, tenv: &mut TypeEnv) -> Type {
    Type::Fun {
        param_ty: Box::new(annotate_tensor_ty_sig(node_name, &sig.from, tenv)),
        return_ty: Box::new(annotate_tensor_ty_sig(node_name, &sig.to, tenv)),
    }
}

fn annotate_tensor_ty_sig(node_name: &str, sig: &TensorTy, tenv: &mut TypeEnv) -> Type {
    use self::TensorTy::*;
    match sig {
        &Generic(ref dims) => {
            tenv.make_tensor(node_name, dims)
        }
        &TyAlias(ref als) => {
            tenv.resolve_alias(node_name, als).unwrap()
        }
    }
}

fn annotate_weights_assign(node_name: &str, w_assign: &WeightsAssign, tenv: &mut TypeEnv) -> TypedWeightsAssign {
    TypedWeightsAssign {
        name: w_assign.name.clone(),
        ty: tenv.fresh_var(),
        mod_name: w_assign.mod_name.clone(),
        fn_ty: tenv.fresh_var(),
        fn_name: w_assign.fn_name.clone(),
        fn_args: w_assign.fn_args.iter().map(|a| annotate_fn_call_arg(node_name, a, tenv)).collect(),
    }
}

fn annotate_fn_call_arg(node_name: &str, call: &FnCallArg, tenv: &mut TypeEnv) -> TypedFnCallArg {
    TypedFnCallArg {
        name: call.name.clone(),
        ty: tenv.fresh_var(),
        arg: Box::new(annotate(*(call.arg).clone(), tenv)),
    }
}

// fn annotate_fn_call(node_name: &str, w_assign: &FnCall, tenv: &mut TypeEnv) -> TypedFnCall {
// }
