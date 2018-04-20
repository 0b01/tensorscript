use typed_ast::typed_term::*;
use typed_ast::typed_term;
use type_reconstruction::subst::Substitution;
use self::TyTerm::*;

pub fn subs(typed_term: &TyTerm, s: &mut Substitution) -> TyTerm {
    // println!("{}", typed_term);
    match typed_term {
        &TyProgram(ref decls) => TyProgram(decls
            .iter()
            .map(|decl| subs_decl(&decl, s))
            .collect()),
        &TyInteger(ref ty, ref a, ref sp) => TyInteger(s.apply_ty(&ty), *a, sp.clone()),
        &TyFloat(ref ty, ref a, ref sp) => TyFloat(s.apply_ty(&ty), *a, sp.clone()),
        &TyList(ref terms) => TyList(terms.iter().map(|t| subs(&t, s)).collect()),
        &TyIdent(ref t, ref name, ref span) => TyIdent(s.apply_ty(t), name.clone(), span.clone()),
        // // &TyFieldAccess(TyFieldAccess),
        &TyFnApp(ref fn_app) => TyFnApp(subs_fn_app(&fn_app, s)),
        &TyBlock { ref stmts, ref ret, ref span } => {
            TyBlock {
                stmts: box subs(&stmts, s),
                ret: box subs(&ret, s),
                span: span.clone(),
            }
        }
        &TyExpr { ref items, ref ty , ref span} => TyExpr { items: box subs(&items, s), ty: s.apply_ty(ty), span: span.clone()},
        &TyStmt { ref items, ref span } => TyStmt{ items: box subs(&items, s), span: span.clone() },
        &TyViewFn(ref view_fn) => TyViewFn(typed_term::TyViewFn {
            ty: s.apply_ty(&view_fn.ty),
            arg: subs_fn_app_arg(&view_fn.arg, s),
            span: view_fn.span.clone(),
        }),
        _ => {
            panic!("{:#?}", typed_term);
        }
    }
}

fn subs_decl(decl: &TyDecl, s: &mut Substitution) -> TyDecl {
    use self::TyDecl::*;
    match decl {
        TyGraphDecl(d) => TyGraphDecl(subs_graph_decl(d, s)),
        TyNodeDecl(d) => TyNodeDecl(subs_node_decl(d, s)),
        TyUseStmt(d) => TyUseStmt(subs_use_stmt(d, s)),
        TyWeightsDecl(d) => TyWeightsDecl(subs_weights_decl(d, s)),
    }
}

fn subs_graph_decl(decl: &TyGraphDecl, s: &mut Substitution) -> TyGraphDecl {
    TyGraphDecl {
        name: decl.name.clone(),
        ty_sig: s.apply_ty(&decl.ty_sig),
        fns: decl.fns.iter().map(|f| subs_fn_decl(f, s)).collect(),
        span: decl.span.clone(),
    }
}

fn subs_fn_decl(decl: &TyFnDecl, s: &mut Substitution) -> TyFnDecl {
    let mut c = decl.clone();
    c.param_ty = s.apply_ty(&c.param_ty);
    c.return_ty = s.apply_ty(&c.return_ty);
    c.func_block = box subs(&c.func_block, s);
    c
}

fn subs_node_decl(decl: &TyNodeDecl, s: &mut Substitution) -> TyNodeDecl {
    TyNodeDecl {
        name: decl.name.clone(),
        ty_sig: s.apply_ty(&decl.ty_sig),
        span: decl.span.clone(),
    }
}

fn subs_weights_decl(decl: &TyWeightsDecl, s: &mut Substitution) -> TyWeightsDecl {
    let mut c = decl.clone();
    c.ty_sig = s.apply_ty(&c.ty_sig);
    c.inits = c.inits.iter().map(|w_a| subs_weights_assign(w_a, s)).collect::<Vec<_>>();
    c
}

fn subs_use_stmt(decl: &TyUseStmt, _tenv: &mut Substitution) -> TyUseStmt {
    decl.clone()
}

fn subs_weights_assign(w_a: &TyWeightsAssign, s: &mut Substitution) -> TyWeightsAssign {
    let mut c = w_a.clone();
    c.ty = s.apply_ty(&c.ty);
    c.arg_ty = s.apply_ty(&c.arg_ty);
    c.fn_args = c.fn_args.iter().map(|a| subs_fn_app_arg(a, s)).collect();
    c
}

fn subs_fn_app(fn_app: &typed_term::TyFnApp, s: &mut Substitution) -> typed_term::TyFnApp {
    let mut c = fn_app.clone();
    c.arg_ty = s.apply_ty(&c.arg_ty);
    c.ret_ty = s.apply_ty(&c.ret_ty);
    c.args = c.args.iter().map(|a| subs_fn_app_arg(&a, s) ).collect();
    c
}

fn subs_fn_app_arg(a: &TyFnAppArg, s: &mut Substitution) -> TyFnAppArg {
    TyFnAppArg {
        name: a.name.clone(),
        arg: box subs(&a.arg, s),
        span: a.span.clone(),
    }
}