use std::fmt::{Debug, Error, Formatter};
use typed_ast::type_env::TypeId;

#[derive(PartialEq, Clone, Eq, Hash)]
pub enum Type {
    Unit,
    INT,
    BOOL,
    VAR(TypeId),
    DIM(TypeId),
    ResolvedDim(i64),
    FUN(Box<Type>, Box<Type>),
    TSR(Vec<Type>),
}

impl Debug for Type {
    fn fmt(&self, f: &mut Formatter) -> Result<(), Error> {
        use self::Type::*;
        match self {
            Unit => write!(f, "()"),
            INT => write!(f, "int"),
            BOOL => write!(f, "bool"),
            VAR(ref t_id) => write!(f, "'{:?}", t_id),
            DIM(ref t_id) => write!(f, "!{:?}", t_id),
            ResolvedDim(ref d) => write!(f, "<{}>", d),
            FUN(ref p, ref r) => write!(f, "({:?} -> {:?})", p, r),
            TSR(ref dims) => {
                if dims.len() > 0 {
                    write!(f, "[")?;
                    for i in dims[0..dims.len() - 1].iter() {
                        write!(f, "{:?}, ", i)?;
                    }
                    write!(f, "{:?}]", dims[dims.len() - 1])
                } else {
                    write!(f, "[]")
                }
            }
        }
    }
}
