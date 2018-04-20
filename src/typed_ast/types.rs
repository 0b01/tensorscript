/// Types for typed AST
use std::fmt::{Debug, Error, Formatter};
use typed_ast::type_env::TypeId;

#[derive(PartialEq, Clone, Eq, Hash)]
pub enum Type {
    // literals
    Unit,
    INT,
    FLOAT,
    BOOL,
    UnresolvedModuleFun(&'static str, &'static str, &'static str),
    // type variables that need to be resolved
    VAR(TypeId),
    DIM(TypeId),

    // recursive types
    Module(String, Option<Box<Type>>),
    FnArgs(Vec<Type>),
    FnArg(Option<String>, Box<Type>),
    ResolvedDim(i64),
    FUN(Box<Type>, Box<Type>),
    TSR(Vec<Type>),
}

impl Type {
    pub fn as_str(&self) -> &str {
        use self::Type::*;
        match self {
            Module(ref n, _) => n,
            _ => unimplemented!(),
        }
    }

    pub fn as_num(&self) -> i64 {
        use self::Type::*;
        match self {
            ResolvedDim(ref i) => *i,
            _ => unimplemented!(),
        }
    }

    // pub fn is_resolved(&self) -> bool {
    //     use self::Type::*;
    //     match self {
    //         Unit => true,
    //         INT => true,
    //         FLOAT => true,
    //         BOOL => true,
    //         UnresolvedModuleFun(_,_,_) => false,

    //         VAR(_) => false,
    //         DIM(_) => false,

    //         Module(_, Some(i)) => i.is_resolved(),
    //         Module(_, None) => false,
    //         FnArgs(ts) => ts.iter().map(|t| t.is_resolved()).all(|t| t),
    //         FnArg(_, t) => t.is_resolved(),
    //         ResolvedDim(_) => true,
    //         FUN(p, r) => Type::is_resolved(p) && r.is_resolved(),
    //         TSR(ts) => ts.iter().map(|t| t.is_resolved()).all(|t|t),
    //         MismatchedDim(_,_) => true,
    //         _ => unimplemented!(),
    //     }
    // }

}

impl Debug for Type {
    fn fmt(&self, f: &mut Formatter) -> Result<(), Error> {
        use self::Type::*;
        match self {
            Unit => write!(f, "()"),
            INT => write!(f, "int"),
            FLOAT => write!(f, "float"),
            BOOL => write!(f, "bool"),
            UnresolvedModuleFun(ref a, ref b, ref c) => write!(f, "UNRESOLVED({}::{}::{})", a, b ,c),
            VAR(ref t_id) => write!(f, "'{:?}", t_id),
            DIM(ref t_id) => write!(f, "!{:?}", t_id),
            FnArgs(ref args) => write!(f, "FnArgs({:?})", args),
            FnArg(ref name, ref ty) => write!(f, "ARG({:?}={:?})", name, ty),
            ResolvedDim(ref d) => write!(f, "<{}>", d),
            Module(ref s, ref ty) => write!(f, "MODULE({}, {:?})", s, ty),
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

macro_rules! args {
    ( $( $x:expr ),* ) => {
        {
            Type::FnArgs(vec![$($x),*])
        }
    };
}

macro_rules! arg {
    ($name:expr, $ty:expr) => {
        Type::FnArg(Some($name.to_owned()), box $ty)
    };
}

macro_rules! fun {
    ($e1:expr, $e2:expr) => {
        Type::FUN(box $e1, box $e2)
    };
}

macro_rules! module {
    ($e1:expr) => {
        Type::Module($e1.to_owned(), None)
    };
}
