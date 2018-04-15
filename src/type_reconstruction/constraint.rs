use std::collections::HashSet;

use typed_ast::typed_term::*;
use typed_ast::type_env::TypeEnv;
use typed_ast::Type;

#[derive(Debug, Hash, Eq, PartialEq)]
pub struct Equals(Type, Type);

#[derive(Debug)]
pub struct Constraints {
    constraints: HashSet<Equals>,
}

impl Constraints {
    pub fn new() -> Self {
        Constraints {
            constraints: HashSet::new(),
        }
    }

    fn add(&mut self, a: Type, b: Type) {
        self.constraints.insert(Equals(a, b));
    }


    pub fn collect(&mut self, typed_ast: &TyTerm, tenv: &TypeEnv) {
        use self::TyTerm::*;
        match typed_ast {
            &TyProgram(ref decls) => decls.iter().map(|decl| collect_ty_decl(self, &decl, tenv)).collect::<()>(),
            // &TyInteger(Type, i64),
            // &TyFloat(Type, f64),
            // &TyList(Vec<TyTerm>),
            // &TyIdent(Type, String),
            // &TyFieldAccess(TyFieldAccess),
            // &TyFnApp(TyFnApp),
            // &TyBlock { stmts: Box<TyTerm>, ret: Box<TyTerm> },
            // &TyExpr { items: Box<TyTerm>, ty: Type },
            // &TyStmt { items: Box<TyTerm> },
            // &TyViewFn(TyViewFn),
            _ => unimplemented!(),
        }
    }
}

fn collect_ty_decl(cs: &mut Constraints, decl: &TyDecl, tenv: &TypeEnv) {
    use self::TyDecl::*;
    match decl {
        TyGraphDecl(d) => collect_graph_decl(cs, d, tenv),
        TyNodeDecl(d) => collect_node_decl(cs, d, tenv),
        TyUseStmt(d) => collect_use_stmt(cs, d, tenv),
        TyWeightsDecl(d) => collect_weights_decl(cs, d, tenv),
    }
}

fn collect_graph_decl(cs: &mut Constraints, decl: &TyGraphDecl, tenv: &TypeEnv) {
}

fn collect_node_decl(_cs: &mut Constraints, _decl: &TyNodeDecl, _tenv: &TypeEnv) {
    ()
}

fn collect_weights_decl(cs: &mut Constraints, decl: &TyWeightsDecl, tenv: &TypeEnv) {
    decl.inits.iter().map(|w| collect_weights_assign(cs, &w, tenv)).collect::<Vec<()>>();
}

fn collect_use_stmt(_cs: &mut Constraints, _decl: &TyUseStmt, _tenv: &TypeEnv) {
    ()
}

fn collect_weights_assign(cs: &mut Constraints, w_a: &TyWeightsAssign, tenv: &TypeEnv) {
    // w_a.fn_ty

    // ...
}