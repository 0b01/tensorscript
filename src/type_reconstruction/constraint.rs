use std::collections::HashSet;

use typed_ast::Type;
use typed_ast::type_env::{ModName, TypeEnv};
use typed_ast::typed_term::*;

#[derive(Debug, Hash, Eq, PartialEq, Clone)]
pub struct Equals(pub Type, pub Type);

#[derive(Debug, Clone)]
pub struct Constraints(pub HashSet<Equals>);

impl Constraints {
    pub fn new() -> Self {
        Constraints(HashSet::new())
    }

    fn add(&mut self, a: Type, b: Type) {
        self.0.insert(Equals(a, b));
    }

    pub fn collect(&mut self, typed_term: &TyTerm, tenv: &mut TypeEnv) {
        use self::TyTerm::*;
        let module = tenv.module().clone();
        // println!("{}", typed_term);
        match typed_term {
            &TyProgram(ref decls) => decls
                .iter()
                .map(|decl| collect_ty_decl(self, &decl, tenv))
                .collect(),
            // &TyInteger(Type, i64),
            // &TyFloat(Type, f64),
            &TyList(ref terms) => terms.iter().map(|t| self.collect(&t, tenv)).collect(),
            &TyIdent(ref t, ref name) => self.add(
                t.clone(),
                tenv.resolve_alias(&module, name.as_str()).unwrap().clone(),
            ),
            // &TyFieldAccess(TyFieldAccess),
            &TyFnApp(ref fn_app) => collect_fn_app(self, &fn_app, tenv),
            &TyBlock { ref stmts, ref ret } => {
                self.collect(&stmts, tenv);
                self.collect(&ret, tenv);
            }
            &TyExpr { ref items, ty: _ } => self.collect(&items, tenv), // ... need to use ty?
            &TyStmt { ref items } => self.collect(&items, tenv),
            // &TyViewFn(TyViewFn),
            _ => unimplemented!(),
        }
    }
}

fn collect_ty_decl(cs: &mut Constraints, decl: &TyDecl, tenv: &mut TypeEnv) {
    use self::TyDecl::*;
    match decl {
        TyGraphDecl(d) => collect_graph_decl(cs, d, tenv),
        TyNodeDecl(d) => collect_node_decl(cs, d, tenv),
        TyUseStmt(d) => collect_use_stmt(cs, d, tenv),
        TyWeightsDecl(d) => collect_weights_decl(cs, d, tenv),
    }
}

fn collect_graph_decl(cs: &mut Constraints, decl: &TyGraphDecl, tenv: &mut TypeEnv) {
    tenv.set_module(ModName::Named(decl.name.clone()));
    // type decl should be the same
    let graph_ty_sig = tenv.resolve_alias(&ModName::Global, decl.name.as_str())
        .unwrap()
        .clone();
    cs.add(decl.ty_sig.clone(), graph_ty_sig);
    // collect fn_decls
    decl.fns
        .iter()
        .map(|f| collect_fn_decl(cs, &f, tenv))
        .collect::<Vec<_>>();
    tenv.set_module(ModName::Global);
}

fn collect_fn_decl(cs: &mut Constraints, decl: &TyFnDecl, tenv: &mut TypeEnv) {
    cs.collect(&decl.func_block, tenv);
    cs.add(decl.return_ty.clone(), decl.func_block.ty());
    cs.add(
        decl.fn_ty.clone(),
        Type::Fun {
            param_ty: Box::new(decl.param_ty.clone()),
            return_ty: Box::new(decl.return_ty.clone()),
        },
    )
    // ...
}

fn collect_node_decl(cs: &mut Constraints, decl: &TyNodeDecl, tenv: &TypeEnv) {
    // type decl should be the same
    let graph_ty_sig = tenv.resolve_alias(&ModName::Global, decl.name.as_str())
        .unwrap()
        .clone();
    cs.add(decl.ty_sig.clone(), graph_ty_sig);
}

fn collect_weights_decl(cs: &mut Constraints, decl: &TyWeightsDecl, tenv: &TypeEnv) {
    // type decl should be the same
    let graph_ty_sig = tenv.resolve_alias(&ModName::Global, decl.name.as_str())
        .unwrap()
        .clone();
    cs.add(decl.ty_sig.clone(), graph_ty_sig);

    // collect weight assigns
    decl.inits
        .iter()
        .map(|w| collect_weights_assign(cs, &w, tenv))
        .collect::<Vec<_>>();
}

fn collect_use_stmt(_cs: &mut Constraints, _decl: &TyUseStmt, _tenv: &TypeEnv) {
    ()
}

fn collect_weights_assign(_cs: &mut Constraints, _w_a: &TyWeightsAssign, _tenv: &TypeEnv) {
    // w_a.fn_ty
    // ... need to somehow collect_fn_app
    ()
}

fn collect_fn_app(_cs: &mut Constraints, _fn_app: &TyFnApp, _tenv: &TypeEnv) {
    // ...
}
