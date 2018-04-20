use codespan::ByteSpan;
/// Data structures for untyped AST.
///
use std::fmt::{Display, Error, Formatter};

type Expression = Box<Term>;
type Statements = Box<Term>;

#[derive(Debug, PartialEq, Clone)]
pub enum Term {
    None,
    /// a vector of decls
    Program(Vec<Decl>),
    Integer(i64, ByteSpan),
    Float(f64, ByteSpan),
    List(Vec<Term>),
    Ident(String, ByteSpan),
    ViewFn(ViewFn),
    FieldAccess(FieldAccess),
    FnApp(FnApp),
    Block {
        stmts: Statements,
        ret: Expression,
        span: ByteSpan,
    },
    Expr {
        items: Box<Term>,
        span: ByteSpan,
    },
    Stmt {
        items: Box<Term>,
        span: ByteSpan,
    },
    Pipes(Vec<Term>),
}

// impl Term {
//     pub fn span(&self) -> ByteSpan {
//         use self::Term::*;
//         match self {
//             None,
//             /// a vector of decls
//             Program(Vec<Decl>),
//             Integer(i64),
//             Float(f64),
//             List(Vec<Term>),
//             Ident(String, ByteSpan),
//             ViewFn(ViewFn),
//             FieldAccess(FieldAccess),
//             FnApp(FnApp),
//             Block {
//                 stmts: Statements,
//                 ret: Expression,
//                 span: ByteSpan,
//             },
//             Expr {
//                 items: Box<Term>,
//                 span: ByteSpan,
//             },
//             Stmt {
//                 items: Box<Term>,
//                 span: ByteSpan,
//             },
//             Pipes(Vec<Term>, ),
//             _ => unimplemented!(),
//         }
//     }
// }

#[derive(Debug, PartialEq, Clone)]
pub enum Decl {
    NodeDecl(NodeDecl),
    WeightsDecl(WeightsDecl),
    GraphDecl(GraphDecl),
    UseStmt(UseStmt),
}

#[derive(Debug, PartialEq, Clone)]
pub struct UseStmt {
    pub mod_name: String,
    pub imported_names: Vec<String>,
    pub span: ByteSpan,
}

#[derive(Debug, PartialEq, Clone)]
pub struct NodeDecl {
    pub name: String,
    pub ty_sig: FnTySig,
    pub defs: Vec<NodeAssign>,
    pub span: ByteSpan,
}

#[derive(Debug, PartialEq, Clone)]
pub struct GraphDecl {
    pub name: String,
    pub ty_sig: FnTySig,
    pub fns: Vec<FnDecl>,
    pub span: ByteSpan,
}

#[derive(Debug, PartialEq, Clone)]
pub struct WeightsDecl {
    pub name: String,
    pub ty_sig: FnTySig,
    pub inits: Vec<WeightsAssign>,
    pub span: ByteSpan,
}

#[derive(Debug, PartialEq, Clone)]
pub struct FnDeclParam {
    pub name: String,
    pub ty_sig: TensorTy,
    pub span: ByteSpan,
}

#[derive(Debug, PartialEq, Clone)]
pub struct FieldAccess {
    pub mod_name: String,
    pub field_name: String,
    pub func_call: Option<Vec<FnAppArg>>,
    pub span: ByteSpan,
}

#[derive(Debug, PartialEq, Clone)]
pub struct FnApp {
    pub name: String,
    pub args: Vec<FnAppArg>,
    pub span: ByteSpan,
}

#[derive(Debug, PartialEq, Clone)]
pub struct FnAppArg {
    pub name: String,
    pub arg: Box<Term>,
    pub span: ByteSpan,
}

#[derive(Debug, PartialEq, Clone)]
pub struct WeightsAssign {
    pub name: String,
    pub mod_name: String,
    pub fn_name: String,
    pub mod_sig: Option<FnTySig>,
    pub fn_args: Vec<FnAppArg>,
    pub span: ByteSpan,
}

#[derive(Debug, PartialEq, Clone)]
pub struct FnTySig {
    pub from: TensorTy,
    pub to: TensorTy,
}

#[derive(Debug, PartialEq, Clone)]
pub struct FnDecl {
    pub name: String,
    pub fn_params: Vec<FnDeclParam>,
    pub return_ty: TensorTy,
    pub func_block: Box<Term>,
    pub span: ByteSpan,
}

#[derive(Debug, PartialEq, Clone)]
pub enum NodeAssign {
    ValueAlias {
        ident: String,
        rhs: Term,
        span: ByteSpan,
    },
    TyAlias {
        ident: String,
        rhs: TensorTy,
        span: ByteSpan,
    },
}

#[derive(Debug, PartialEq, Clone)]
pub enum TensorTy {
    TyAlias(String, ByteSpan),
    Generic(Vec<String>, ByteSpan),
}

#[derive(Debug, PartialEq, Clone)]
pub struct ViewFn {
    pub dims: Vec<String>,
    pub span: ByteSpan,
}

impl Term {
    // pub fn is(&self, var: &Self) -> bool {
    //   ::std::mem::discriminant(self) == ::std::mem::discriminant(var)
    // }

    // pub fn is_UseStmt(&self) -> bool {
    //   self.is(&Term::UseStmt {
    //     mod_name: format!(""),
    //     imported_names: vec![],
    //   })
    // }

    // pub fn to_list(&self) -> Option<Vec<Term>> {
    //   if let &Term::List(ref vs) = self {
    //     Some(vs.to_vec())
    //   } else {
    //     None
    //   }
    // }

    // /// args is List(Arg)
    // pub fn extend_arg_list(func: FnApp, init: Term) -> Vec<FnAppArg> {
    //     let mut new_arg_vec = vec![
    //         FnAppArg {
    //             name: format!("x"),
    //             arg: Box::new(init),
    //         },
    //     ];
    //     new_arg_vec.extend(func.args);
    //     new_arg_vec
    // }
}

impl Display for Term {
    fn fmt(&self, f: &mut Formatter) -> Result<(), Error> {
        write!(f, "{:#?}", self)
    }
}

// #[derive(Debug, PartialEq, Clone)]
// pub enum Op {
//   Expo,
//   Mult,
//   Div,
//   Mod,
//   Add,
//   Sub,
//   ShL,
//   ShR,
//   BAnd,
//   BOr,
//   BXor,
//   Lt,
//   LtE,
//   Gt,
//   GtE,
//   Eq,
//   NotEq,
//   And,
//   Or,
//   Assign,
// }
