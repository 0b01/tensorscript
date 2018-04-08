use std::fmt::{ Formatter, Display, Error };

// struct Program {
//   module: Module
// }

// struct Module {
//   decls: Vec<Decl>
// }

// struct Decl {
//   decl_type: DeclType
// }

// enum DeclType {
//   Node,
//   Weights,
//   Graph
// }



#[derive(Debug, PartialEq, Clone)]
pub enum AST {
  None,
  Integer(i64),
  Float(f64),
  String(String),
  Atom(String),
  True,
  False,
  Braced(Box<AST>),
  List(Vec<AST>),
  Ident(String),
  FieldAccess(FieldAccess),
  Block {
    stmts: Box<AST>,
    ret: Box<AST>,
  },
  Expr {
    items: Box<AST>,
  },
  Stmt {
    items: Box<AST>,
  },
  FnCall(FnCall),
  NodeDecl {
    name: String,
    type_sig: FnTypeSig,
    initialization: Vec<MacroAssign>,
  },
  GraphDecl {
    name: String,
    type_sig: FnTypeSig,
    fns: Vec<FnDecl>,
  },
  WeightsDecl {
    name: String,
    type_sig: FnTypeSig,
    initialization: Vec<WeightsAssign>,
  },
  Pipes(Vec<AST>),
  UseStmt {
    mod_name: String,
    imported_names: Vec<String>
  },
}

#[derive(Debug, PartialEq, Clone)]
pub struct FnDeclArg {
  pub name: String,
  pub type_sig: Vec<String>,
}

#[derive(Debug, PartialEq, Clone)]
pub struct FieldAccess {
    pub var_name: String,
    pub field_name: String,
    pub func_call: Option<Vec<FnCallArg>>,
}

#[derive(Debug, PartialEq, Clone)]
pub struct FnCall {
  pub name: String,
  pub args: Vec<FnCallArg>,
}

#[derive(Debug, PartialEq, Clone)]
pub struct FnCallArg {
  pub name: String,
  pub arg: Box<AST>,
}

#[derive(Debug, PartialEq, Clone)]
pub struct WeightsAssign {
  pub name: String,
  pub mod_name: String,
  pub mod_sig: FnTypeSig,
  pub func: FnCall,
}

#[derive(Debug, PartialEq, Clone)]
pub struct FnTypeSig{
  pub from: Vec<String>,
  pub to: Vec<String>,
}

#[derive(Debug, PartialEq, Clone)]
pub struct FnDecl {
  pub name: String,
  pub fn_params: Vec<FnDeclArg>,
  pub return_type: Vec<String>,
  pub func_block: Box<AST>,
}

#[derive(Debug, PartialEq, Clone)]
pub struct MacroAssign {
  pub ident: String,
  pub rhs: Box<AST>
}

impl AST {

  pub fn is(&self, var: &Self) -> bool {
    ::std::mem::discriminant(self) == ::std::mem::discriminant(var)
  }

  pub fn is_UseStmt(&self) -> bool {
    self.is(&AST::UseStmt {
      mod_name: format!(""),
      imported_names: vec![],
    })
  }

  pub fn to_list(&self) -> Option<Vec<AST>> {
    if let &AST::List(ref vs) = self {
      Some(vs.to_vec())
    } else {
      None
    }
  }

  /// args is List(Arg)
  pub fn extend_arg_list(func: FnCall, init: AST) -> Vec<FnCallArg> {
    let mut new_arg_vec = vec![FnCallArg {
            name: format!("x"),
            arg: Box::new(init),
        }];
    new_arg_vec.extend(func.args);
    new_arg_vec
  }
}

impl Display for AST {
  fn fmt(&self, f: &mut Formatter) -> Result<(), Error> {
    let txt = format!("{:?}", self);
    write!(f, "{}", txt)
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
