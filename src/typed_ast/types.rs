/// Types for typed AST
use std::hash::{Hash, Hasher};
use std::fmt::{Debug, Error, Formatter};
use typed_ast::type_env::TypeId;
use codespan::{Span, ByteIndex};

#[derive(PartialEq, Clone, Eq)]
pub enum Type {
    // literals
    Unit(Span<ByteIndex>),
    INT(Span<ByteIndex>),
    FLOAT(Span<ByteIndex>),
    BOOL(Span<ByteIndex>),
    UnresolvedModuleFun(&'static str, &'static str, &'static str, Span<ByteIndex>),
    // type variables that need to be resolved
    VAR(TypeId, Span<ByteIndex>),
    DIM(TypeId, Span<ByteIndex>),

    // recursive types
    Module(String, Option<Box<Type>>, Span<ByteIndex>),
    FnArgs(Vec<Type>, Span<ByteIndex>),
    FnArg(Option<String>, Box<Type>, Span<ByteIndex>),
    ResolvedDim(i64, Span<ByteIndex>),
    FUN(Box<Type>, Box<Type>, Span<ByteIndex>),
    TSR(Vec<Type>, Span<ByteIndex>),
}

impl Hash for Type {
    fn hash<H: Hasher>(&self, state: &mut H) {
        use self::Type::*;
        match self {
            Unit(_) => ().hash(state),
            INT(_) => 0.hash(state),
            FLOAT(_) => 1.hash(state),
            BOOL(_) => 2.hash(state),
            // UnresolvedModuleFun(_,_,_) => false,

            VAR(a, _) => {3.hash(state); a.hash(state)},
            DIM(b, _) => {4.hash(state); b.hash(state)},

            Module(a, b, _) => {5.hash(state);a.hash(state);b.hash(state);},
            FnArgs(ts, _) => {6.hash(state); ts.hash(state)},
            FnArg(n, t, _) => {7.hash(state); n.hash(state); t.hash(state);}
            ResolvedDim(a, _) => {8.hash(state); a.hash(state)},
            FUN(p, r, _) => {9.hash(state); p.hash(state); r.hash(state);},
            TSR(ts, _) => {10.hash(state); ts.hash(state);}
            UnresolvedModuleFun(a,b,c,_) => {11.hash(state);a.hash(state);b.hash(state);c.hash(state);},
            // MismatchedDim(_,_) => true,
            _ => {
                panic!("{:?}", self);
            }
        }
    }
}

impl Type {

    pub fn with_span(&self, sp: &Span<ByteIndex>) -> Type {
        use self::Type::*;
        match self {
            Unit(_) => Unit(sp.clone()),
            VAR(ref a,_) => VAR(*a, sp.clone()),
            DIM(ref a,_) => DIM(*a, sp.clone()),
            INT(_) => INT(sp.clone()),
            FLOAT(_) => FLOAT(sp.clone()),
            BOOL(_) => BOOL(sp.clone()),
            UnresolvedModuleFun(ref a, ref b, ref c,_) => UnresolvedModuleFun(a,b,c,sp.clone()),
            FnArgs(ref args,_) => FnArgs(args.clone(), sp.clone()),
            FnArg(ref name, ref ty,_) => FnArg(name.clone(), ty.clone(), sp.clone()),
            ResolvedDim(ref d,_) => ResolvedDim(d.clone(), sp.clone()),
            Module(ref s, ref ty,_) => Module(s.clone(),ty.clone(),sp.clone()),
            FUN(ref p, ref r,_) => FUN(p.clone(), r.clone(),sp.clone()),
            TSR(ref dims,_) => TSR(dims.clone(),sp.clone()),
            _ => panic!("{:?}", self),
            
        }
    }

    pub fn as_str(&self) -> &str {
        use self::Type::*;
        match self {
            Module(ref n, _,_) => n,
            _ => unimplemented!(),
        }
    }

    pub fn as_num(&self) -> i64 {
        use self::Type::*;
        match self {
            ResolvedDim(ref i, _) => *i,
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
            Unit(_) => write!(f, "()"),
            INT(_) => write!(f, "int"),
            FLOAT(_) => write!(f, "float"),
            BOOL(_) => write!(f, "bool"),
            UnresolvedModuleFun(ref a, ref b, ref c,_) => write!(f, "UNRESOLVED({}::{}::{})", a, b ,c),
            VAR(ref t_id,_) => write!(f, "'{:?}", t_id),
            DIM(ref t_id,_) => write!(f, "!{:?}", t_id),
            FnArgs(ref args,_) => write!(f, "FnArgs({:?})", args),
            FnArg(ref name, ref ty,_) => write!(f, "ARG({:?}={:?})", name, ty),
            ResolvedDim(ref d, ref s) => write!(f, "<{}({})>", d, s),
            Module(ref s, ref ty,_) => write!(f, "MODULE({}, {:?})", s, ty),
            FUN(ref p, ref r,_) => write!(f, "({:?} -> {:?})", p, r),
            TSR(ref dims,_) => {
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
            Type::FnArgs(vec![$($x),*], CSpan::fresh_span())
        }
    };
}

macro_rules! arg {
    ($name:expr, $ty:expr) => {
        Type::FnArg(Some($name.to_owned()), box $ty, CSpan::fresh_span())
    };
}

macro_rules! fun {
    ($e1:expr, $e2:expr) => {
        Type::FUN(box $e1, box $e2, CSpan::fresh_span())
    };
}

macro_rules! float {
    () => {
        Type::FLOAT(CSpan::fresh_span())
    };
}

macro_rules! unit {
    () => {
        Type::Unit(CSpan::fresh_span())
    };
}

macro_rules! int {
    () => {
        Type::INT(CSpan::fresh_span())
    };
}

macro_rules! module {
    ($e1:expr) => {
        Type::Module($e1.to_owned(), None, CSpan::fresh_span())
    };
}
