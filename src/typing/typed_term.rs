/// Data structures for Typed AST
///
use codespan::ByteSpan;
use span::CSpan;
use std::collections::BTreeMap;
use typing::type_env::Alias;
use typing::Type;
use std::fmt::Write;

/// typed AST nodes
#[derive(Debug, PartialEq, Clone)]
pub enum TyTerm {
    TyNone,
    TyProgram(Vec<TyDecl>),
    TyInteger(Type, i64, ByteSpan),
    TyFloat(Type, f64, ByteSpan),
    TyList(Vec<TyTerm>),
    TyIdent(Type, Alias, ByteSpan),
    TyFieldAccess(TyFieldAccess),
    TyFnApp(Box<TyFnApp>),
    TyTuple(Type, Vec<TyTerm>, ByteSpan),
    TyBlock {
        stmts: Box<TyTerm>,
        ret: Box<TyTerm>,
        span: ByteSpan,
    },
    TyExpr(Box<TyTerm>, Type, ByteSpan),
    TyStmt(Box<TyTerm>, ByteSpan),
}

/// convenience functions and getters for a bunch of attributes
pub trait Conversion {
    /// get the type of AST node
    fn ty(&self) -> Type;
    /// get the span of AST node
    fn span(&self) -> ByteSpan;
    /// get integer value of Integer node
    fn as_num(&self) -> Option<i64> {
        None
    }
    /// convert ty term to string
    fn as_str(&self) -> Option<String> {
        None
    }
}

impl Conversion for TyTerm {
    fn span(&self) -> ByteSpan {
        use self::TyTerm::*;
        match self {
            TyNone => CSpan::fresh_span(),
            TyProgram(_) => CSpan::fresh_span(),
            TyInteger(_, _, ref s) => *s,
            TyFloat(_, _, ref s) => *s,
            TyIdent(_, _, ref s) => *s,
            TyFieldAccess(ref f_a) => f_a.span(),
            TyFnApp(ref f_a) => f_a.span(),
            TyBlock {ref span, ..} => *span,
            TyExpr(_, _, ref span) => *span,
            TyStmt(_, ref span) => *span,
            _ => panic!("{:?}", self),
        }
    }

    fn ty(&self) -> Type {
        use self::TyTerm::*;
        use self::Type::*;
        match self {
            TyNone => Unit(CSpan::fresh_span()),
            TyProgram(_) => Unit(CSpan::fresh_span()),
            TyInteger(ref t, _, _) => t.clone(),
            TyFloat(ref t, _, _) => t.clone(),
            TyList(_) => Unit(CSpan::fresh_span()),
            TyIdent(ref t, _, _) => t.clone(),
            TyFieldAccess(ref f_a) => f_a.ty(),
            TyFnApp(ref f_a) => f_a.ty(),
            TyBlock {ref ret, ..} => ret.ty(),
            TyExpr(_,ref ty, _) => ty.clone(),
            TyStmt(..) => Unit(CSpan::fresh_span()),
            TyTuple(ref t, ..) => t.clone(),
        }
    }

    fn as_num(&self) -> Option<i64> {
        use self::TyTerm::*;
        match self {
            TyInteger(_, i, _) => Some(*i),
            TyExpr(ref items, ..) => items.as_num(),
            TyIdent(ref t, ..) => t.as_num(),
            _ => None,
        }
    }

    fn as_str(&self) -> Option<String> {
        use self::TyTerm::*;
        let mut s = String::new();
        match self {
            TyInteger(..) => write!(s, "{}", self.as_num()?),
            TyExpr(ref items, ..) => write!(s, "{}", items.as_str()?),
            TyIdent(ref t, ..) => write!(s, "{}", t.as_string()),
            TyFloat(_, f, ..) => write!(s, "{}", f),
            TyTuple(_, ref ts, _) => {
                write!(s, "(");
                write!(s, "{}", ts
                    .iter()
                    .map(|t| t.as_str())
                    .collect::<Option<Vec<_>>>()?.join(", ")
                );
                write!(s, ")")
            }
            _ => panic!("{:?}", self),
        };
        Some(s)
    }
}

impl Conversion for TyFieldAccess {
    fn span(&self) -> ByteSpan {
        self.span
    }
    fn ty(&self) -> Type {
        self.ty.clone()
    }
}

impl Conversion for TyFnApp {
    fn span(&self) -> ByteSpan {
        self.span
    }
    fn ty(&self) -> Type {
        self.ret_ty.clone()
    }
}

#[derive(Debug, PartialEq, Clone)]
pub enum TyDecl {
    TyNodeDecl(TyNodeDecl),
    TyWeightsDecl(TyWeightsDecl),
    TyGraphDecl(TyGraphDecl),
    TyUseStmt(TyUseStmt),
    TyAliasAssign(TyAliasAssign),
}

#[derive(Debug, PartialEq, Clone)]
pub enum TyAliasAssign {
    Placeholder,
    // Dimension {
    //     ident: String,
    //     span: ByteSpan,
    // },
    // Tensor {
    //     ident: String,
    //     span: ByteSpan,
    // },
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
    pub mod_name: String,
    pub fn_name: String,
    pub arg_ty: Type,
    pub fn_args: Vec<TyFnAppArg>,
    pub span: ByteSpan,
}

#[derive(Debug, PartialEq, Clone)]
pub struct TyFnApp {
    pub mod_name: Option<String>,
    pub orig_name: Option<String>,
    pub name: Alias,
    pub arg_ty: Type,
    pub ret_ty: Type,
    pub args: Vec<TyFnAppArg>,
    pub span: ByteSpan,
}

impl TyFnApp {
    pub fn extend_arg(&mut self, arg: &TyFnAppArg) {
        self.args.insert(0, arg.clone());
        let new_args_ty = self.args.to_ty(&self.span);
        // self.fn_ty = match &self.fn_ty {
        // Type::FUN(_, box r, span) => Type::FUN(box new_args_ty, box r.clone(), span),
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
                        t_arg.span,
                    )
                })
                .collect(),
            *span,
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
                        t_arg.span,
                    )
                })
                .collect(),
            *span,
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
pub struct TyFnDecl {
    pub name: Alias,
    pub fn_params: Vec<TyFnDeclParam>,
    pub arg_ty: Type, // args!()
    pub ret_ty: Type, // any type
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
