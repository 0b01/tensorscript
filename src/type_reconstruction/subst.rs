/// Hindley-Milner type inference for type reconstruction
///
/// This consists of three substeps:
///
/// 1. Collect constraints. (handled in constraint.rs)
///     In this step, traverse typed ast and collect types of adjacent nodes that should
///     be equivalent. This generates a Constraint struct which is just a thin wrapper
///     around a hashset of (Type, Type) tuple.
///
/// 2. Unify constraints by generating substitutions.
///     This is a variant to Algorithm W in H-M type inference. Bascially, unify_one
///     function tries to replace 1 type var with a concrete type. The parent function, unify,
///     then uses that substitution on the rest of the constraints, thus eliminating the type
///     variable from the constraint set. The process is iterated until one of these conditions are met:
///     a) all type variable are exhausted. b) equivalence that can never happen. c) circular
///     type dependence (handled by occurs check).
///
/// 3. Generate Substitutions
///     Now after the unification is complete, the function returns a list of substitutions that
///     should remove all type variables from the typed AST.
///
use std::collections::HashMap;
use type_reconstruction::constraint::{Constraints, Equals};
use typed_ast::type_env::TypeEnv;
use typed_ast::type_env::TypeId;
use typed_ast::Type;

use codespan_reporting::termcolor::StandardStream;
use codespan_reporting::{emit, ColorArg, Diagnostic, Label, Severity};

use span::CSpan;

#[derive(Debug)]
pub struct Substitution(pub HashMap<Type, Type>);

impl Substitution {
    /// returns an empty substitution
    pub fn new() -> Substitution {
        Substitution::empty()
    }

    /// apply substitution to a set of constraints
    pub fn apply(&mut self, cs: &Constraints) -> Constraints {
        Constraints(
            cs.0
                .iter()
                .map(|Equals(a, b)| Equals(self.apply_ty(a), self.apply_ty(b)))
                .collect(),
        )
    }

    pub fn apply_ty(&mut self, ty: &Type) -> Type {
        self.0.iter().fold(ty.clone(), |result, solution| {
            let (ty, solution_type) = solution;
            if let Type::VAR(ref tvar, ref span) = ty {
                substitute_tvar(result, tvar, &solution_type.with_span(span))
            } else {
                panic!();
                // substitute_ty(result, ty, solution_type)
            }
        })
    }

    pub fn compose(&mut self, mut other: Substitution) -> Substitution {
        let mut self_substituded: HashMap<Type, Type> = self.0
            .clone()
            .into_iter()
            .map(|(k, s)| (k, other.apply_ty(&s)))
            .collect();
        self_substituded.extend(other.0);
        Substitution(self_substituded)
    }

    fn empty() -> Substitution {
        Substitution(HashMap::new())
    }
}

fn occurs(tvar: TypeId, ty: &Type) -> bool {
    use self::Type::*;
    match ty {
        &FUN(ref p, ref r, _) => occurs(tvar, &p) | occurs(tvar, &r),
        &VAR(ref tvar2, _) => tvar == *tvar2,
        _ => false,
    }
}

/// replace tvar with replacement in ty
fn substitute_tvar(ty: Type, tvar: &TypeId, replacement: &Type) -> Type {
    use self::Type::*;
    // println!("\nTVAR:::\n{:?}, \n'{:?}, \n{:?}\n", ty, tvar, replacement);
    match ty {
        UnresolvedModuleFun(_, _, _, _) => ty,
        Unit(_) => ty,
        INT(_) => ty,
        BOOL(_) => ty,
        FLOAT(_) => ty,
        ResolvedDim(_, _) => ty,
        VAR(tvar2, span) => {
            if tvar.clone() == tvar2 {
                replacement.with_span(&span)
            } else {
                ty
            }
        }
        DIM(tvar2, span) => {
            if tvar.clone() == tvar2 {
                replacement.with_span(&span)
            } else {
                ty
            }
        }
        FnArgs(args, span) => FnArgs(
            args.into_iter()
                .map(|ty| match ty {
                    FnArg(name, a, s) => FnArg(name, box substitute_tvar(*a, tvar, replacement), s),
                    _ => panic!(ty),
                })
                .collect(),
            span,
        ),
        FUN(p, r, s) => FUN(
            box substitute_tvar(*p, tvar, &replacement),
            box substitute_tvar(*r, tvar, &replacement),
            s,
        ),
        TSR(_, _) => ty,

        Module(n, Some(box ty), s) => {
            Module(n, Some(box substitute_tvar(ty, tvar, replacement)), s)
        }

        Module(_, None, _) => ty,
        FnArg(name, box ty, s) => FnArg(name, box substitute_tvar(ty, tvar, replacement), s),
        _ => {
            panic!("{:?}", ty);
        }
    }
}

pub fn unify(constraints: Constraints, tenv: &mut TypeEnv) -> Substitution {
    if constraints.0.is_empty() {
        Substitution::new()
    } else {
        let mut it = constraints.0.into_iter();
        let mut subst = unify_one(it.next().unwrap(), tenv);
        let subst_tail = subst.apply(&Constraints(it.collect()));
        let subst_tail: Substitution = unify(subst_tail, tenv);
        subst.compose(subst_tail)
    }
}

fn unify_one(cs: Equals, tenv: &mut TypeEnv) -> Substitution {
    use self::Type::*;
    // println!("{:?}", cs);
    match cs {
        Equals(Unit(_), Unit(_)) => Substitution::empty(),
        Equals(INT(_), INT(_)) => Substitution::empty(),
        Equals(FLOAT(_), FLOAT(_)) => Substitution::empty(),
        Equals(BOOL(_), BOOL(_)) => Substitution::empty(),

        Equals(INT(_), ResolvedDim(_, _)) => Substitution::empty(),
        Equals(ResolvedDim(_, _), INT(_)) => Substitution::empty(),

        Equals(a @ ResolvedDim(_, _), b @ ResolvedDim(_, _)) => {
            if a.as_num() == b.as_num() {
                Substitution::empty()
            } else {
                match (a, b) {
                    (ResolvedDim(v1, s1), ResolvedDim(v2, s2)) => {
                        // let error = Diagnostic::new(Severity::Error, "Unexpected type in `+` application")
                        //     .with_label(
                        //         Label::new_primary(s1))
                        //     .with_message("Dimension mismatch.");

                        // let writer = StandardStream::stderr(opts.color.into());
                        // emit(&mut writer.lock(), &code_map, &error).unwrap();
                        // println!();

                        panic!("Dimension mismatch! {:?} != {:?} ({}/{})", v1, v2, s1, s2);
                    }
                    _ => unimplemented!(),
                }
            }
        }

        Equals(VAR(tvar, _), ty) => unify_var(tvar, ty),
        Equals(ty, VAR(tvar, _)) => unify_var(tvar, ty),

        Equals(DIM(tvar, _), ty) => unify_var(tvar, ty),
        Equals(ty, DIM(tvar, _)) => unify_var(tvar, ty),

        Equals(FnArgs(v1, _), FnArgs(v2, _)) => unify(
            Constraints(v1.into_iter().zip(v2).map(|(i, j)| Equals(i, j)).collect()),
            tenv,
        ),

        Equals(FnArg(Some(a), ty1, _), FnArg(Some(b), ty2, _)) => {
            if a == b {
                unify(
                    Constraints(hashset!{
                        Equals(*ty1, *ty2),
                    }),
                    tenv,
                )
            } else {
                panic!("supplied parameter is incorrect! {} != {}", a, b);
            }
        }

        Equals(FUN(p1, r1, _), FUN(p2, r2, _)) => unify(
            Constraints(hashset!{
                Equals(*p1, *p2),
                Equals(*r1, *r2),
            }),
            tenv,
        ),
        Equals(TSR(dims1, _), TSR(dims2, _)) => unify(
            Constraints({
                dims1
                    .into_iter()
                    .zip(dims2)
                    .map(|(i, j)| Equals(i, j))
                    .collect()
            }),
            tenv,
        ),

        Equals(Module(n1, Some(box ty1), _), Module(n2, Some(box ty2), _)) => unify(
            Constraints(hashset!{
                if n1 == n2 {
                    Equals(ty1, ty2)
                } else {
                    panic!();
                }
            }),
            tenv,
        ),

        Equals(UnresolvedModuleFun(_, _, _, _), _) => Substitution::empty(),

        _ => {
            panic!("{:#?}", cs);
        }
    }
}

fn unify_var(tvar: TypeId, ty: Type) -> Substitution {
    use self::Type::*;

    let span = CSpan::fresh_span();
    match ty.clone() {
        VAR(tvar2, _) => {
            if tvar == tvar2 {
                Substitution::empty()
            } else {
                Substitution(hashmap!{ VAR(tvar, span) => ty })
            }
        }
        DIM(tvar2, _) => {
            if tvar == tvar2 {
                Substitution::empty()
            } else {
                Substitution(hashmap!{ VAR(tvar, span) => ty })
            }
        }
        _ => if occurs(tvar, &ty) {
            panic!("circular type")
        } else {
            Substitution(hashmap!{ VAR(tvar, span) => ty })
        },
    }
}
