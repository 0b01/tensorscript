use std::collections::BTreeMap;
use typing::constraint::{Constraints, Equals};
use typing::type_env::TypeId;
use typing::Type;

#[derive(Debug, PartialEq)]
pub struct Substitution(pub BTreeMap<Type, Type>);

impl Substitution {

    /// apply substitution to a set of constraints
    pub fn apply(&mut self, cs: &Constraints) -> Constraints {
        Constraints {
            set: cs.set
                .iter()
                .map(|Equals(a, b)| Equals(self.apply_ty(a), self.apply_ty(b)))
                .collect(),
            emitter: cs.emitter.clone(),
        }
    }

    pub fn apply_ty(&mut self, ty: &Type) -> Type {
        self.0.iter().fold(ty.clone(), |result, solution| {
            let (ty, solution_type) = solution;
            if let Type::VAR(ref tvar, ref span) = ty {
                substitute_tvar(result, tvar, &solution_type.with_span(span))
            } else {
                substitute_ty(result, ty, solution_type)
            }
        })
    }

    pub fn compose(&mut self, mut other: Substitution) -> Substitution {
        let mut self_substituded: BTreeMap<Type, Type> = self.0
            .clone()
            .into_iter()
            .map(|(k, s)| (k, other.apply_ty(&s)))
            .collect();
        self_substituded.extend(other.0);
        Substitution(self_substituded)
    }

    pub fn empty() -> Substitution {
        Substitution(BTreeMap::new())
    }
}

/// magic...
///
/// basically, a workaround to replace UnresolvedModuleFunction...
/// for more info take a look at unify_one
///         Equals(u @ UnresolvedModuleFun(_, _, _, _), ty) => {..}
/// which is the only place that calls it
fn substitute_ty(ty: Type, _replaced: &Type, replacement: &Type) -> Type {
    if let Type::FUN(m1,n1,_,_,_) = ty.clone() {
    if let Type::FUN(m2,n2,_,_,_) = replacement.clone() {
        if (m1 == m2) && (n1 == n2) {
            // println!("----\n{:?}\n{:?}\n{:?}\n-----\n\n", ty, replaced, replacement);
            ty.clone()
        } else {ty}
    } else {ty}} else {
        ty
    }
}

/// replace tvar with replacement in ty
fn substitute_tvar(ty: Type, tvar: &TypeId, replacement: &Type) -> Type {
    use self::Type::*;
    // println!("\nTVAR:::\n{:?}, \n'{:?}, \n{:?}\n", ty, tvar, replacement);
    match ty {
        UnresolvedModuleFun(_, _, _, _) => {
            println!("{:?}, replacement: {:?}", ty, replacement);
            ty
        },
        Unit(_) => ty,
        INT(_) => ty,
        BOOL(_) => ty,
        FLOAT(_) => ty,
        ResolvedDim(_, _) => ty,
        VAR(tvar2, span) => {
            if *tvar == tvar2 {
                replacement.with_span(&span)
            } else {
                ty
            }
        }
        DIM(tvar2, span) => {
            if *tvar == tvar2 {
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
        Tuple(tys, s) => Tuple(tys.into_iter().map(|t| substitute_tvar(t, tvar, replacement)).collect(), s),
        FUN(module,name,p, r, s) => FUN(
            module,
            name,
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
    }
}
