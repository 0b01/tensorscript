use parser::term::{Term, Decl, FnTySig, TensorTy, NodeAssign};
use typed_ast::typed_term::{TypedTerm, TypedDecl, TypedNodeDecl};
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
        _ => unimplemented!(),
    }
}

fn annotate_decl(decl: &Decl, tenv: &mut TypeEnv) -> TypedDecl {
    use self::Decl::*;
    match decl {
        &NodeDecl(ref decl) => {
            let assigns = &decl.defs;
            for a in assigns.into_iter() {
                match a {
                    &NodeAssign::TyAlias {
                        ident: ref id,
                        rhs: TensorTy::Generic(ref tys)
                    } => {
                        tenv.add_tsr_alias(&decl.name, id, tys);
                    },
                    &NodeAssign::ValueAlias {
                        ident: ref id,
                        rhs: Term::Integer(_)
                    } => {
                        tenv.add_dim_alias(&decl.name, id);
                    },
                    _ => unimplemented!(),
                }
            }

            TypedDecl::TypedNodeDecl(TypedNodeDecl {
                name: decl.name.clone(),
                ty_sig: annotate_fn_ty_sig(&decl.name, &decl.ty_sig, tenv),
            })
        },
        // &WeightsDecl(decl) => {

        // },
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
        },
        &TyAlias(ref als) => tenv.resolve_alias(node_name, als).unwrap(),
    }
}