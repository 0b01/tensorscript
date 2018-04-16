use std::collections::{HashSet, HashMap};
use type_reconstruction::constraint::{Constraints, Equals};
use typed_ast::Type;
use typed_ast::type_env::TypeId;

#[derive(Debug)]
pub struct Substitution(pub HashMap<TypeId, Type>);

impl Substitution {
    pub fn new() -> Substitution {
        Substitution(HashMap::new())
    }

    pub fn apply(&mut self, cs: &Constraints) -> Constraints {
        Constraints(cs.0.iter().map(|eq| self.apply_equals(eq)).collect())
    }

    pub fn apply_equals(&mut self, eq: &Equals) -> Equals {
        let Equals(a, b) = eq;
        Equals(self.apply_ty(a), self.apply_ty(b))
    }

    pub fn apply_ty(&mut self, ty: &Type) -> Type {
        self.0.iter().fold(ty.clone(), |result, solution| {
            let (tvar, solution_type) = solution;
            substitute(result, tvar, solution_type)
        })
    }

    pub fn compose(&mut self, mut other: Substitution) -> Substitution {
        let mut self_substituded: HashMap<TypeId, Type> = self.0
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

    //   def occurs(tvar: Type.Var, ty: Type): Boolean = {
    //     ty match {
    //       case Type.FUN(p, r) => occurs(tvar, p) || occurs(tvar, r)
    //       case Type.VAR(tvar2) => tvar == tvar2
    //       case _ => false
    //     }
    //   }
    // }

fn occurs(tvar: TypeId, ty: &Type) -> bool {
    use self::Type::*;
    match ty {
        &FUN(ref p,ref r) => occurs(tvar, &p) | occurs(tvar, &r),
        &VAR(ref tvar2) => tvar == *tvar2,
        _ => false
    }
}

fn substitute(ty: Type, tvar: &TypeId, replacement: &Type) -> Type {
    use self::Type::*;
    match ty {
        Unit => ty,
        VAR(tvar2) => if tvar.clone() == tvar2 {
            replacement.clone()
        } else {
            ty
        },
        DIM(tvar2) => if tvar.clone() == tvar2 {
            replacement.clone()
        } else {
            ty
        },
        FUN(p, r) => FUN(
            Box::new(substitute(*p, tvar, &replacement)),
            Box::new(substitute(*r, tvar, &replacement)),
        ),
        TSR(dims) => TSR(dims),
        _ => {
            println!(
                "ty: {:?}, tvar: {}, replacement: {:?}",
                ty, tvar, replacement
            );
            unimplemented!();
        }
    }
}

pub fn unify(constraints: Constraints) -> Substitution {
    if constraints.0.is_empty() {
        Substitution::new()
    } else {
        let mut it = constraints.0.into_iter();
        let mut subst = unify_one(it.next().unwrap());
        let subst_tail = subst.apply(&Constraints(it.collect()));
        let subst_tail: Substitution = unify(subst_tail);
        subst.compose(subst_tail)
    }
}

fn unify_one(cs: Equals) -> Substitution {
    use self::Type::*;
    match cs {
        Equals(VAR(tvar), ty) => unify_var(tvar, ty),
        Equals(ty, VAR(tvar)) => unify_var(tvar, ty),
        Equals(DIM(tvar), ty) => unify_var(tvar, ty),
        Equals(ty, DIM(tvar)) => unify_var(tvar, ty),
        Equals(FUN(p1,r1), FUN(p2,r2)) => {
            unify(Constraints({
                let mut hs = HashSet::new();
                hs.insert(Equals(*p1, *p2));
                hs.insert(Equals(*r1, *r2));
                hs
            }))
        },
        Equals(TSR(dims1), TSR(dims2)) => {
            unify(Constraints({
                let mut hs = HashSet::new();
                for (i, j) in dims1.into_iter().zip(dims2) {
                    hs.insert(Equals(i, j));
                }
                hs
            }))
        },
        _ => {
            println!("{:#?}", cs);
            unimplemented!();
        }
    }
}

//   def unifyOne(constraint: Constraint): Substitution = {
//     (constraint.a, constraint.b) match {
//       case (Type.INT, Type.INT) => Substitution.empty
//       case (Type.BOOL, Type.BOOL) => Substitution.empty
//       case (Type.FUN(param1, return1), Type.FUN(param2, return2)) =>
//         unify(Set(
//           Constraint(param1, param2),
//           Constraint(return1, return2)
//         ))
//       case (Type.VAR(tvar), ty) => unifyVar(tvar, ty)
//       case (ty, Type.VAR(tvar)) => unifyVar(tvar, ty)
//       case (a, b) => throw new RuntimeException(s"cannot unify $a with $b")
//     }
//   }

fn unify_var(tvar: TypeId, ty: Type) -> Substitution {
    use self::Type::*;
    if let VAR(tvar2) = ty {
        if tvar == tvar2 { // if they are the same, no substitution
            Substitution::empty()
        } else { // they must be equal
            Substitution({
                let mut hm = HashMap::new();
                hm.insert(tvar, ty);
                hm
            })
        }
    } else if occurs(tvar, &ty) {
        panic!("circular use: {} occurs in {:?}", tvar, ty);
    } else {
        Substitution({
            let mut hm = HashMap::new();
            hm.insert(tvar, ty);
            hm
        })
    }
}

//   def unifyVar(tvar: Type.Var, ty: Type): Substitution = {
//     ty match {
//       case Type.VAR(tvar2) if tvar == tvar2 => Substitution.empty
//       case Type.VAR(_) => Substitution.fromPair(tvar, ty)
//       case ty if occurs(tvar, ty) =>
//         throw new RuntimeException(s"circular use: $tvar occurs in $ty")
//       case ty => Substitution.fromPair(tvar, ty)
//     }
//   }