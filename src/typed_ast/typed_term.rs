use std::fmt::{Display, Formatter, Error};
use typed_ast::Type;

pub trait Typed {
    fn ty(&self) -> Type;
}

#[derive(Debug, PartialEq, Clone)]
pub enum TypedTerm {
    None,
    TypedProgram(Vec<TypedDecl>),
    TypedInteger(i64),
    TypedFloat(f64),
    TypedList(Vec<TypedTerm>),
    TypedIdent(String),
    // TypedFieldAccess(FieldAccess),
    // TypedFnCall(FnCall),
    TypedBlock { stmts: Box<TypedTerm>, ret: Box<TypedTerm> },
    TypedExpr { items: Box<TypedTerm> },
    TypedStmt { items: Box<TypedTerm> },
    TypedPipes(Vec<TypedTerm>),
}

impl Display for TypedTerm {
    fn fmt(&self, f: &mut Formatter) -> Result<(), Error> {
        write!(f, "{:#?}", self)
    }
}

#[derive(Debug, PartialEq, Clone)]
pub enum TypedDecl {
    TypedNodeDecl(TypedNodeDecl),
    TypedWeightsDecl(TypedWeightsDecl),
    TypedGraphDecl(TypedGraphDecl),
    TypedUseStmt(TypedUseStmt),
}

#[derive(Debug, PartialEq, Clone)]
pub struct TypedUseStmt {
    pub mod_name: String,
    pub imported_names: Vec<String>,
}

#[derive(Debug, PartialEq, Clone)]
pub struct TypedNodeDecl {
    pub name: String,
    pub ty_sig: Type,
}

#[derive(Debug, PartialEq, Clone)]
pub struct TypedGraphDecl {
    pub name: String,
    pub ty_sig: Type,
    pub fns: Vec<TypedFnDecl>,
}

#[derive(Debug, PartialEq, Clone)]
pub struct TypedWeightsDecl {
    pub name: String,
    pub ty_sig: Type,
    pub initialization: Vec<TypedWeightsAssign>,
}

#[derive(Debug, PartialEq, Clone)]
pub struct TypedWeightsAssign {
    pub name: String,
    pub mod_name: String,
    pub mod_sig: Type,
    pub func: TypedFnCall,
}

#[derive(Debug, PartialEq, Clone)]
pub struct TypedFnCall {
    pub name: String,
    pub args: Vec<TypedFnCallArg>,
}

#[derive(Debug, PartialEq, Clone)]
pub struct TypedFnCallArg {
    pub name: String,
    pub arg: Box<TypedTerm>,
}

#[derive(Debug, PartialEq, Clone)]
pub enum TypedNodeAssign {
    ValueAlias { ident: String, rhs: Box<TypedTerm> },
    TyAlias { ident: String, rhs: Type },
}

#[derive(Debug, PartialEq, Clone)]
pub struct TypedFnDecl {
    pub name: String,
    pub fn_params: Vec<TypedFnDeclArg>,
    pub return_ty: Type,
    pub func_block: Box<TypedTerm>,
}

#[derive(Debug, PartialEq, Clone)]
pub struct TypedFnDeclArg {
    pub name: String,
    pub ty_sig: Type,
}