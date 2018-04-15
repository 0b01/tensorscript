use typed_ast::type_env::TypeId;
use std::fmt::{Debug, Formatter, Error};

#[derive(PartialEq, Clone)]
pub enum Type {
    Unit,
    Var(TypeId),
    Dim(TypeId),
    Fun { param_ty: Box<Type>, return_ty: Box<Type> },
    Tensor { rank: usize, dims: Vec<Type> },
}

impl Debug for Type {
    fn fmt(&self, f: &mut Formatter) -> Result<(), Error> {
        use self::Type::*;
        match self {
            Unit => write!(f, "()"),
            Var(ref t_id) => write!(f, "?{:?}", t_id),
            Dim(ref t_id) => write!(f, "!{:?}", t_id),
            Fun { ref param_ty, ref return_ty } => write!(f, "{:?} -> {:?}", param_ty, return_ty),
            Tensor { ref rank, ref dims } => {
                write!(f, "[")?;
                for i in dims[0..dims.len()-1].iter() {
                    write!(f, "{:?}, ", i)?;
                }
                write!(f, "{:?}]", dims[dims.len()-1])
            },
        }
    }
}