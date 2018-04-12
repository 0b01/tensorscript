use std::fmt::{ Formatter, Display, Error };

#[derive(Debug, PartialEq, Clone)]
pub struct Program {
  pub module: Module,
}

#[derive(Debug, PartialEq, Clone)]
pub struct Module {
  pub decls: Vec<Decl>
}

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
  List(Vec<AST>),
  Ident(String),
  FieldAccess(FieldAccess),
  FnCall(FnCall),
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
  Pipes(Vec<AST>),
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
  pub type_sig: FnTypeSig,
  pub initialization: Vec<MacroAssign>,
}

#[derive(Debug, PartialEq, Clone)]
pub struct GraphDecl {
  pub name: String,
  pub type_sig: FnTypeSig,
  pub fns: Vec<FnDecl>,
}

#[derive(Debug, PartialEq, Clone)]
pub struct WeightsDecl {
  pub name: String,
  pub type_sig: FnTypeSig,
  pub initialization: Vec<WeightsAssign>,
}

#[derive(Debug, PartialEq, Clone)]
pub struct FnDeclArg {
  pub name: String,
  pub type_sig: TensorType,
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
  pub from: TensorType,
  pub to: TensorType,
}

#[derive(Debug, PartialEq, Clone)]
pub struct FnDecl {
  pub name: String,
  pub fn_params: Vec<FnDeclArg>,
  pub return_type: TensorType,
  pub func_block: Box<AST>,
}


#[derive(Debug, PartialEq, Clone)]
pub enum MacroAssign {
  ValueAlias {
    ident: String,
    rhs: Box<AST>
  },

  TypeAlias {
    ident: String,
    rhs: TensorType,
  },
}

#[derive(Debug, PartialEq, Clone)]
pub enum TensorType {
  TypeAlias(String),
  Generic(Vec<String>),
}

impl AST {

  // pub fn is(&self, var: &Self) -> bool {
  //   ::std::mem::discriminant(self) == ::std::mem::discriminant(var)
  // }

  // pub fn is_UseStmt(&self) -> bool {
  //   self.is(&AST::UseStmt {
  //     mod_name: format!(""),
  //     imported_names: vec![],
  //   })
  // }

  // pub fn to_list(&self) -> Option<Vec<AST>> {
  //   if let &AST::List(ref vs) = self {
  //     Some(vs.to_vec())
  //   } else {
  //     None
  //   }
  // }

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
