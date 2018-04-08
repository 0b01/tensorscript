use std::fmt::{ Formatter, Display, Error };

#[derive(Debug, PartialEq, Clone)]
pub enum AST {
  None,
  Integer(i64),
  Float(f64),
  String(String),
  Atom(String),
  True,
  False,
  UnaryNot(Box<AST>),
  UnaryComplement(Box<AST>),
  UnaryPlus(Box<AST>),
  UnaryMinus(Box<AST>),
  Braced(Box<AST>),
  Constant(String),
  LocalAccess(String),
  CallWithImplicitSelf(Box<AST>, Vec<(AST, AST)>),
  List(Vec<AST>),
  Function(Vec<AST>),
  Clause(Vec<(AST, AST)>, Box<AST>),

  Ident(String),
  Block {
    stmts: Box<AST>,
    ret: Box<AST>,
  },
  WeightsDecl {
    name: String,
    type_sig: Box<AST>,
    initialization: Box<AST>,
  },
  Expr {
    items: Box<AST>,
  },
  Stmt {
    items: Box<AST>,
  },
  FieldAccess {
      var_name: String,
      field_name: String,
      func_call: Box<AST>,
  },
  NodeDecl {
    name: String,
    type_sig: Box<AST>,
    initialization: Box<AST>,
  },
  GraphDecl {
    name: String,
    type_sig: Box<AST>,
    fns: Box<AST>,
  },
  WeightsAssign {
    name: String,
    mod_name: String,
    mod_sig: Box<AST>,
    func: Box<AST>,
  },
  FnCall {
    name: String,
    args: Box<AST>,
  },
  FnCallArg {
    name: String,
    arg: Box<AST>,
  },
  FnDecl {
    name: String,
    fn_params: Box<AST>,
    return_type: Vec<String>,
    func_block: Box<AST>,
  },
  FnDeclArg {
    name: String,
    type_sig: Vec<String>,
  },
  UseStmt {
    mod_name: String,
    imported_names: Vec<String>
  },
  Start,
  FnTypeSig(Vec<String>, Vec<String>),
  MacroAssign(String, Box<AST>),
}

impl AST {
  /// args is List(Arg)
  pub fn extend_arg_list(args: Box<AST>, init: AST) -> Box<AST> {
      if let AST::List(vec) = *args {
          let mut new_arg_vec = vec![AST::FnCallArg {
                  name: format!("x"),
                  arg: Box::new(init),
              }];
          new_arg_vec.extend(vec);

          Box::new(AST::List(new_arg_vec))
      } else { 
        println!("{:?}", args);
        unimplemented!();
      }
  }
}

impl Display for AST {
  fn fmt(&self, f: &mut Formatter) -> Result<(), Error> {
    let txt = format!("{:?}", self);
    write!(f, "{}", txt)
  }
}

#[derive(Debug, PartialEq, Clone)]
pub enum Op {
  Expo,
  Mult,
  Div,
  Mod,
  Add,
  Sub,
  ShL,
  ShR,
  BAnd,
  BOr,
  BXor,
  Lt,
  LtE,
  Gt,
  GtE,
  Eq,
  NotEq,
  And,
  Or,
  Assign,
}
