use std::fmt::{Display, Error, Formatter};
use typed_ast::Type;

pub trait Ty {
    fn ty(&self) -> Type;
}

impl Ty for TyTerm {
    fn ty(&self) -> Type {
        use self::TyTerm::*;
        use self::Type::*;
        match self {
            &TyNone => Unit,
            &TyProgram(_) => Unit,
            &TyInteger(ref t, _) => t.clone(),
            &TyFloat(ref t, _) => t.clone(),
            &TyList(_) => Unit,
            &TyIdent(ref t, _) => t.clone(),
            &TyFieldAccess(ref f_a) => f_a.ty(),
            &TyFnApp(ref f_a) => f_a.ty(),
            &TyBlock { stmts: _, ref ret } => ret.ty(),
            &TyExpr { items: _, ref ty } => ty.clone(),
            &TyStmt { items: _ } => Unit,
            &TyViewFn(ref view_fn) => view_fn.ty(),
        }
    }
}

impl Ty for TyFieldAccess {
    fn ty(&self) -> Type {
        self.ty.clone()
    }
}

impl Ty for TyFnApp {
    fn ty(&self) -> Type {
        self.ret_ty.clone()
    }
}

impl Ty for TyViewFn {
    fn ty(&self) -> Type {
        self.ty.clone()
    }
}

#[derive(Debug, PartialEq, Clone)]
pub enum TyTerm {
    TyNone,
    TyProgram(Vec<TyDecl>),
    TyInteger(Type, i64),
    TyFloat(Type, f64),
    TyList(Vec<TyTerm>),
    TyIdent(Type, String),
    TyFieldAccess(TyFieldAccess),
    TyFnApp(TyFnApp),
    TyBlock {
        stmts: Box<TyTerm>,
        ret: Box<TyTerm>,
    },
    TyExpr {
        items: Box<TyTerm>,
        ty: Type,
    },
    TyStmt {
        items: Box<TyTerm>,
    },
    TyViewFn(TyViewFn),
}

impl Display for TyTerm {
    fn fmt(&self, f: &mut Formatter) -> Result<(), Error> {
        write!(f, "{:#?}", self)
    }
}

#[derive(Debug, PartialEq, Clone)]
pub enum TyDecl {
    TyNodeDecl(TyNodeDecl),
    TyWeightsDecl(TyWeightsDecl),
    TyGraphDecl(TyGraphDecl),
    TyUseStmt(TyUseStmt),
}

#[derive(Debug, PartialEq, Clone)]
pub struct TyUseStmt {
    pub mod_name: String,
    pub imported_names: Vec<String>,
}

#[derive(Debug, PartialEq, Clone)]
pub struct TyNodeDecl {
    pub name: String,
    pub ty_sig: Type,
}

#[derive(Debug, PartialEq, Clone)]
pub struct TyGraphDecl {
    pub name: String,
    pub ty_sig: Type,
    pub fns: Vec<TyFnDecl>,
}

#[derive(Debug, PartialEq, Clone)]
pub struct TyWeightsDecl {
    pub name: String,
    pub ty_sig: Type,
    pub inits: Vec<TyWeightsAssign>,
}

#[derive(Debug, PartialEq, Clone)]
pub struct TyWeightsAssign {
    pub name: String,
    pub ty: Type,
    pub mod_name: String,
    pub fn_name: String,
    pub fn_args: Vec<TyFnAppArg>,
}

#[derive(Debug, PartialEq, Clone)]
pub struct TyFnApp {
    pub mod_name: Option<String>,
    pub name: String,
    pub arg_ty: Type,
    pub ret_ty: Type,
    pub args: Vec<TyFnAppArg>,
}

impl TyFnApp {
    pub fn extend_arg(&mut self, arg: TyFnAppArg) {
        self.args.insert(0, arg.clone());
        match &mut self.arg_ty {
            &mut Type::FN_ARGS(ref mut args) => args.insert(0, Type::FN_ARG(arg.name.clone(), box arg.ty)),
            &mut Type::VAR(_) => (),
            _ => unimplemented!(),
        };
    }
}

#[derive(Debug, PartialEq, Clone)]
pub struct TyFnAppArg {
    pub name: Option<String>,
    pub ty: Type,
    pub arg: Box<TyTerm>,
}

#[derive(Debug, PartialEq, Clone)]
pub struct TyViewFn {
    pub ty: Type,
    pub arg: TyFnAppArg,
}

// #[derive(Debug, PartialEq, Clone)]
// pub enum TyNodeAssign {
//     ValueAlias { ident: String, rhs: Box<TyTerm> },
//     TyAlias { ident: String, rhs: Type },
// }

#[derive(Debug, PartialEq, Clone)]
pub struct TyFnDecl {
    pub name: String,
    pub fn_params: Vec<TyFnDeclParam>,
    pub fn_ty: Type,
    pub param_ty: Type,
    pub return_ty: Type,
    pub func_block: Box<TyTerm>,
}

#[derive(Debug, PartialEq, Clone)]
pub struct TyFnDeclParam {
    pub name: String,
    pub ty_sig: Type,
}

#[derive(Debug, PartialEq, Clone)]
pub struct TyFieldAccess {
    pub mod_name: String,
    pub field_name: String,
    pub ty: Type,
}
