use std::collections::HashMap;
use typed_ast::type_env::TypeId;
use typed_ast::Type;
use type_reconstruction::constraint::{Constraints, Equals};

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
        Equals(self.apply_ty(a), self.apply_ty(b),)
    }

    pub fn apply_ty(&mut self, ty: &Type) -> Type {
        self.0.iter().fold(ty.clone(), |result, solution| {
            let (tvar, solution_type) = solution;
            substitute(result, tvar, solution_type)
        })
    }

    pub fn compose(&mut self, mut other: Substitution) -> Substitution {
        let mut self_substituded: HashMap<TypeId, Type> = self.0.clone().into_iter().map(|(k,s)| (k, other.apply_ty(&s))).collect();
        self_substituded.extend(other.0);
        Substitution(self_substituded)
    }

    pub fn unify(constraints: Constraints) -> Substitution {
        if constraints.0.is_empty() {
            Substitution::new()
        } else {
            let mut it = constraints.0.into_iter();
            let mut subst = Self::unify_one(it.next().unwrap());
            let subst_tail = subst.apply(&Constraints(it.collect()));
            let subst_tail: Substitution = Self::unify(subst_tail);
            subst.compose(subst_tail)
        }
    }
    
    fn unify_one(cs: Equals) -> Substitution {
        let Equals(a, b) = cs;
        match (a, b)  {
            _ => unimplemented!()
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

//   def unifyVar(tvar: Type.Var, ty: Type): Substitution = {
//     ty match {
//       case Type.VAR(tvar2) if tvar == tvar2 => Substitution.empty
//       case Type.VAR(_) => Substitution.fromPair(tvar, ty)
//       case ty if occurs(tvar, ty) =>
//         throw new RuntimeException(s"circular use: $tvar occurs in $ty")
//       case ty => Substitution.fromPair(tvar, ty)
//     }
//   }

//   def occurs(tvar: Type.Var, ty: Type): Boolean = {
//     ty match {
//       case Type.FUN(p, r) => occurs(tvar, p) || occurs(tvar, r)
//       case Type.VAR(tvar2) => tvar == tvar2
//       case _ => false
//     }
//   }
// }
}

fn substitute(ty: Type, tvar: &TypeId, replacement: &Type) -> Type {
    use self::Type::*;
    println!("{:?}", ty);
    match ty {
        Unit => ty,
        Var(tvar2) => if tvar.clone() == tvar2 { replacement.clone() } else { ty },
        Dim(tvar2) => if tvar.clone() == tvar2 { replacement.clone() } else { ty },
        Fun { param_ty, return_ty } => Type::Fun {
            param_ty: Box::new(substitute(*param_ty, tvar, &replacement)),
            return_ty: Box::new(substitute(*return_ty, tvar, &replacement)),
        },
        // Tensor {
        //     rank: usize,
        //     dims: Vec<Type>,
        // },
        _ => {
            println!("ty: {:?}, tvar: {}, replacement: {:?}", ty, tvar, replacement);
            unimplemented!();
        }
    }
}
