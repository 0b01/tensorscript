use std::fmt::{Display, Formatter, Error};
use typed_ast::Type;

pub trait Typed {
    fn ty(&self) -> Type;
}

impl Typed for TypedTerm {
    fn ty(&self) -> Type {
        use self::TypedTerm::*;
        use self::Type::*;
        match self {
            &TypedNone => Unit,
            &TypedProgram(_) => Unit,
            &TypedInteger(ref t, _) => t.clone(),
            &TypedFloat(ref t, _) => t.clone(),
            &TypedList(_) => Unit,
            &TypedIdent(_, _) => Unit,
            &TypedFieldAccess(ref f_a) => f_a.ty(),
            &TypedFnApp(ref f_a) => f_a.ty(),
            &TypedBlock { stmts: _, ref ret } => ret.ty(),
            &TypedExpr { items: _, ref ty } => ty.clone(),
            &TypedStmt { items: _ } => Unit,
            &TypedViewFn(ref view_fn) => view_fn.ty(),
        }
    }
}

impl Typed for TypedFieldAccess {
    fn ty(&self) -> Type {
        self.ty.clone()
    }
}

impl Typed for TypedFnApp {
    fn ty(&self) -> Type {
        self.ret_ty.clone()
    }
}

impl Typed for TypedViewFn {
    fn ty(&self) -> Type {
        self.ty.clone()
    }
}

#[derive(Debug, PartialEq, Clone)]
pub enum TypedTerm {
    TypedNone,
    TypedProgram(Vec<TypedDecl>),
    TypedInteger(Type, i64),
    TypedFloat(Type, f64),
    TypedList(Vec<TypedTerm>),
    TypedIdent(Type, String),
    TypedFieldAccess(TypedFieldAccess),
    TypedFnApp(TypedFnApp),
    TypedBlock { stmts: Box<TypedTerm>, ret: Box<TypedTerm> },
    TypedExpr { items: Box<TypedTerm>, ty: Type },
    TypedStmt { items: Box<TypedTerm> },
    TypedViewFn(TypedViewFn),
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
    pub inits: Vec<TypedWeightsAssign>,
}

#[derive(Debug, PartialEq, Clone)]
pub struct TypedWeightsAssign {
    pub name: String,
    pub ty: Type,
    pub mod_name: String,
    pub fn_ty: Type,
    pub fn_name: String,
    pub fn_args: Vec<TypedFnAppArg>,
}

#[derive(Debug, PartialEq, Clone)]
pub struct TypedFnApp {
    pub mod_name: Option<String>,
    pub name: String,
    pub args: Vec<TypedFnAppArg>,
    pub ret_ty: Type,
}

impl TypedFnApp {
    pub fn extend_arg(mut self, arg: TypedFnAppArg) -> TypedFnApp {
        self.args.insert(0, arg);
        TypedFnApp {
            mod_name: self.mod_name,
            name: self.name,
            args: self.args,
            ret_ty: self.ret_ty,
        }
    }
}

#[derive(Debug, PartialEq, Clone)]
pub struct TypedFnAppArg {
    pub name: String,
    pub arg: Box<TypedTerm>,
}

#[derive(Debug, PartialEq, Clone)]
pub struct TypedViewFn {
    pub ty: Type,
    pub arg: TypedFnAppArg,
}


// #[derive(Debug, PartialEq, Clone)]
// pub enum TypedNodeAssign {
//     ValueAlias { ident: String, rhs: Box<TypedTerm> },
//     TyAlias { ident: String, rhs: Type },
// }

#[derive(Debug, PartialEq, Clone)]
pub struct TypedFnDecl {
    pub name: String,
    pub fn_params: Vec<TypedFnDeclParam>,
    pub return_ty: Type,
    pub func_block: Box<TypedTerm>,
}

#[derive(Debug, PartialEq, Clone)]
pub struct TypedFnDeclParam {
    pub name: String,
    pub ty_sig: Type,
}

#[derive(Debug, PartialEq, Clone)]
pub struct TypedFieldAccess {
    pub mod_name: String,
    pub field_name: String,
    pub ty: Type,
}
