use codespan::ByteSpan;
use span::CSpan;
use std::collections::BTreeMap;
/// Data structures for Typed AST
///
///
use std::fmt::{Display, Error, Formatter};
use typed_ast::type_env::Alias;
use typed_ast::Type;

pub trait Ty {
    fn ty(&self) -> Type;
    fn span(&self) -> ByteSpan;
    fn int(&self) -> Option<i64> {
        None
    }
}

impl Ty for TyTerm {
    fn span(&self) -> ByteSpan {
        use self::TyTerm::*;
        match self {
            &TyNone => CSpan::fresh_span(),
            &TyProgram(_) => CSpan::fresh_span(),
            &TyInteger(_, _, ref s) => s.clone(),
            &TyFloat(_, _, ref s) => s.clone(),
            // &TyList(_) => Unit(CSpan::fresh_span()),
            &TyIdent(_, _, ref s) => s.clone(),
            &TyFieldAccess(ref f_a) => f_a.span(),
            &TyFnApp(ref f_a) => f_a.span(),
            &TyBlock {
                stmts: _,
                ret: _,
                ref span,
            } => span.clone(),
            &TyExpr {
                items: _,
                ty: _,
                ref span,
            } => span.clone(),
            &TyStmt { items: _, ref span } => span.clone(),
            &TyViewFn(ref view_fn) => view_fn.span(),
            _ => panic!("{:?}", self),
        }
    }

    fn ty(&self) -> Type {
        use self::TyTerm::*;
        use self::Type::*;
        match self {
            &TyNone => Unit(CSpan::fresh_span()),
            &TyProgram(_) => Unit(CSpan::fresh_span()),
            &TyInteger(ref t, _, _) => t.clone(),
            &TyFloat(ref t, _, _) => t.clone(),
            &TyList(_) => Unit(CSpan::fresh_span()),
            &TyIdent(ref t, _, _) => t.clone(),
            &TyFieldAccess(ref f_a) => f_a.ty(),
            &TyFnApp(ref f_a) => f_a.ty(),
            &TyBlock {
                stmts: _,
                ref ret,
                span: _,
            } => ret.ty(),
            &TyExpr {
                items: _,
                ref ty,
                span: _,
            } => ty.clone(),
            &TyStmt { items: _, span: _ } => Unit(CSpan::fresh_span()),
            &TyViewFn(ref view_fn) => view_fn.ty(),
        }
    }

    fn int(&self) -> Option<i64> {
        match self {
            &TyTerm::TyInteger(_, i, _) => Some(i),
            _ => None,
        }
    }
}

impl Ty for TyFieldAccess {
    fn span(&self) -> ByteSpan {
        self.span.clone()
    }
    fn ty(&self) -> Type {
        self.ty.clone()
    }
}

impl Ty for TyFnApp {
    fn span(&self) -> ByteSpan {
        self.span.clone()
    }
    fn ty(&self) -> Type {
        self.ret_ty.clone()
    }
}

impl Ty for TyViewFn {
    fn span(&self) -> ByteSpan {
        self.span.clone()
    }
    fn ty(&self) -> Type {
        self.ty.clone()
    }
}

#[derive(Debug, PartialEq, Clone)]
pub enum TyTerm {
    TyNone,
    TyProgram(Vec<TyDecl>),
    TyInteger(Type, i64, ByteSpan),
    TyFloat(Type, f64, ByteSpan),
    TyList(Vec<TyTerm>),
    TyIdent(Type, Alias, ByteSpan),
    TyFieldAccess(TyFieldAccess),
    TyFnApp(TyFnApp),
    TyBlock {
        stmts: Box<TyTerm>,
        ret: Box<TyTerm>,
        span: ByteSpan,
    },
    TyExpr {
        items: Box<TyTerm>,
        ty: Type,
        span: ByteSpan,
    },
    TyStmt {
        items: Box<TyTerm>,
        span: ByteSpan,
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
    pub span: ByteSpan,
}

#[derive(Debug, PartialEq, Clone)]
pub struct TyNodeDecl {
    pub name: String,
    pub ty_sig: Type,
    pub span: ByteSpan,
}

#[derive(Debug, PartialEq, Clone)]
pub struct TyGraphDecl {
    pub name: String,
    pub ty_sig: Type,
    pub fns: Vec<TyFnDecl>,
    pub span: ByteSpan,
}

#[derive(Debug, PartialEq, Clone)]
pub struct TyWeightsDecl {
    pub name: String,
    pub ty_sig: Type,
    pub inits: Vec<TyWeightsAssign>,
    pub span: ByteSpan,
}

#[derive(Debug, PartialEq, Clone)]
pub struct TyWeightsAssign {
    pub name: String,
    pub ty: Type,
    pub mod_name: String,
    pub fn_name: String,
    pub arg_ty: Type,
    pub fn_args: Vec<TyFnAppArg>,
    pub span: ByteSpan,
}

#[derive(Debug, PartialEq, Clone)]
pub struct TyFnApp {
    pub mod_name: Option<String>,
    pub orig_name: String,
    pub name: Alias,
    pub arg_ty: Type,
    pub ret_ty: Type,
    pub args: Vec<TyFnAppArg>,
    pub span: ByteSpan,
}

impl TyFnApp {
    pub fn extend_arg(&mut self, arg: TyFnAppArg) {
        self.args.insert(0, arg.clone());
        let new_args_ty = self.args.to_ty(&self.span);
        // self.fn_ty = match &self.fn_ty {
        // Type::FUN(_, box r, span) => Type::FUN(box new_args_ty, box r.clone(), span.clone()),
        //     _ => unimplemented!(),
        // };
        self.arg_ty = new_args_ty;
    }
}

#[derive(Debug, PartialEq, Clone)]
pub struct TyFnAppArg {
    pub name: Option<String>,
    pub arg: Box<TyTerm>,
    pub span: ByteSpan,
}

pub trait ArgsVecInto {
    fn to_ty(&self, span: &ByteSpan) -> Type;
    fn to_btreemap(&self) -> Option<BTreeMap<String, Box<TyTerm>>>;
}

impl ArgsVecInto for [TyFnAppArg] {
    fn to_ty(&self, span: &ByteSpan) -> Type {
        Type::FnArgs(
            self.iter()
                .map(|t_arg| {
                    Type::FnArg(
                        t_arg.name.clone(),
                        box t_arg.arg.ty().clone(),
                        t_arg.span.clone(),
                    )
                })
                .collect(),
            span.clone(),
        )
    }
    fn to_btreemap(&self) -> Option<BTreeMap<String, Box<TyTerm>>> {
        Some(
            self.iter()
                .filter_map(|a| {
                    if a.name.is_some() {
                        Some((a.name.clone().unwrap(), a.arg.clone()))
                    } else {
                        None
                    }
                })
                .collect(),
        )
    }
}

impl ArgsVecInto for [TyFnDeclParam] {
    fn to_ty(&self, span: &ByteSpan) -> Type {
        Type::FnArgs(
            self.iter()
                .map(|t_arg| {
                    Type::FnArg(
                        Some(t_arg.name.clone()),
                        box t_arg.ty.clone(),
                        t_arg.span.clone(),
                    )
                })
                .collect(),
            span.clone(),
        )
    }
    fn to_btreemap(&self) -> Option<BTreeMap<String, Box<TyTerm>>> {
        None
        // self.iter().filter_map(|a|
        //     Some((a.name.clone(), box a.clone()))
        // ).collect()
    }
}

#[derive(Debug, PartialEq, Clone)]
pub struct TyViewFn {
    pub ty: Type,
    pub arg: TyFnAppArg,
    pub span: ByteSpan,
}

#[derive(Debug, PartialEq, Clone)]
pub struct TyFnDecl {
    pub name: Alias,
    pub fn_params: Vec<TyFnDeclParam>,
    pub fn_ty: Type,
    pub func_block: Box<TyTerm>,
    pub span: ByteSpan,
}

#[derive(Debug, PartialEq, Clone)]
pub struct TyFnDeclParam {
    pub name: String,
    pub ty: Type,
    pub span: ByteSpan,
}

#[derive(Debug, PartialEq, Clone)]
pub struct TyFieldAccess {
    pub mod_name: String,
    pub field_name: String,
    pub ty: Type,
    pub span: ByteSpan,
}
