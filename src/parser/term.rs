use std::fmt::{Display, Error, Formatter};

#[derive(Debug, PartialEq, Clone)]
pub enum Term {
    None,
    /// a vector of decls
    Program(Vec<Decl>),
    Integer(i64),
    Float(f64),
    List(Vec<Term>),
    Ident(String),
    FieldAccess(FieldAccess),
    FnApp(FnApp),
    Block { stmts: Box<Term>, ret: Box<Term> },
    Expr { items: Box<Term> },
    Stmt { items: Box<Term> },
    Pipes(Vec<Term>),
}

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
}

#[derive(Debug, PartialEq, Clone)]
pub struct NodeDecl {
    pub name: String,
    pub ty_sig: FnTySig,
    pub defs: Vec<NodeAssign>,
}

#[derive(Debug, PartialEq, Clone)]
pub struct GraphDecl {
    pub name: String,
    pub ty_sig: FnTySig,
    pub fns: Vec<FnDecl>,
}

#[derive(Debug, PartialEq, Clone)]
pub struct WeightsDecl {
    pub name: String,
    pub ty_sig: FnTySig,
    pub inits: Vec<WeightsAssign>,
}

#[derive(Debug, PartialEq, Clone)]
pub struct FnDeclArg {
    pub name: String,
    pub ty_sig: TensorTy,
}

#[derive(Debug, PartialEq, Clone)]
pub struct FieldAccess {
    pub var_name: String,
    pub field_name: String,
    pub func_call: Option<Vec<FnAppArg>>,
}

#[derive(Debug, PartialEq, Clone)]
pub struct FnApp {
    pub name: String,
    pub args: Vec<FnAppArg>,
}

#[derive(Debug, PartialEq, Clone)]
pub struct FnAppArg {
    pub name: String,
    pub arg: Box<Term>,
}

#[derive(Debug, PartialEq, Clone)]
pub struct WeightsAssign {
    pub name: String,
    pub mod_name: String,
    pub fn_name: String,
    pub mod_sig: FnTySig,
    pub fn_args: Vec<FnAppArg>,
}

#[derive(Debug, PartialEq, Clone)]
pub struct FnTySig {
    pub from: TensorTy,
    pub to: TensorTy,
}

#[derive(Debug, PartialEq, Clone)]
pub struct FnDecl {
    pub name: String,
    pub fn_params: Vec<FnDeclArg>,
    pub return_ty: TensorTy,
    pub func_block: Box<Term>,
}

#[derive(Debug, PartialEq, Clone)]
pub enum NodeAssign {
    ValueAlias { ident: String, rhs: Term },
    TyAlias { ident: String, rhs: TensorTy },
}

#[derive(Debug, PartialEq, Clone)]
pub enum TensorTy {
    TyAlias(String),
    Generic(Vec<String>),
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

    /// args is List(Arg)
    pub fn extend_arg_list(func: FnApp, init: Term) -> Vec<FnAppArg> {
        let mut new_arg_vec = vec![
            FnAppArg {
                name: format!("x"),
                arg: Box::new(init),
            },
        ];
        new_arg_vec.extend(func.args);
        new_arg_vec
    }
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
