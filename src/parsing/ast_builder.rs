/// builds untyped AST from token tree.
///
use codespan::{ByteSpan, Span, ByteIndex};
use parsing::grammar::Rule::*;
use parsing::grammar::{Rule, TensorScriptParser};
use parsing::term::{Decl, FieldAccess, FnApp, FnAppArg, FnDecl, FnDeclParam, FnTySig, GraphDecl,
                   NodeAssign, NodeDecl, TensorTy, Term, UseStmt, ViewFn, WeightsAssign,
                   WeightsDecl};
use pest::iterators::Pair;
use pest::Parser;
use pest::Error as PestError;
use span::CSpan;
use errors::{Diag, Emitter};
use std::rc::Rc;
use std::cell::RefCell;
use std::process::exit;

pub struct ASTBuilder {
    emitter: Rc<RefCell<Emitter>>,
    cspan: CSpan,
}

impl ASTBuilder {
    pub fn new(emitter: Rc<RefCell<Emitter>>, cspan: CSpan) -> Self {
        Self {
            emitter,
            cspan,
        }
    }

    pub fn parse_str(&self, source: &str) -> Option<Term> {
        // let program = TensorScriptParser::parse(dim_assign, "dim T = 1;");
        // println!("{}", program.unwrap());
        // panic!("test...");

        let parser = TensorScriptParser::parse(Rule::input, source);
        if parser.is_err() {
            let e = parser.err().unwrap();
            if let PestError::ParsingError{ ref positives, ref pos, .. } = e.clone() {
                let e0 = &positives[0];
                let err = match e0 {
                    semicolon => {
                        Diag::ParseError("Missing semicolon".to_owned(),
                            Span::new(ByteIndex(pos.pos() as u32 - 1) , ByteIndex(pos.pos() as u32 - 1 ))
                        )
                    }
                    _ => panic!("{}", e),
                };
                self.emitter.borrow_mut().add(err);
                None
            } else { unimplemented!() }
        } else {
            let decls = parser.unwrap();
            let terms = decls
                .map(|pair| match pair.as_rule() {
                    use_stmt => self.build_use_stmt(pair),
                    weights_decl => self.build_weights_decl(pair),
                    graph_decl => self.build_graph_decl(pair),
                    node_decl => self.build_node_decl(pair),
                    _ => panic!("Only node, graph, weights, use supported at top level"),
                })
                .collect::<Result<_,_>>();
            match terms {
                Ok(t) => Some(Term::Program(t)),
                Err(e) => {
                    let mut em = self.emitter.borrow_mut();
                    em.add(e);
                    em.print_errs();
                    exit(-1);
                    None
                }
            }
        }
    }

    fn consume(&self, pair: Pair<Rule>) -> Result<Term, Diag> {
        // println!("{}", pair);
        match pair.as_rule() {
            // node_decl_body => build_node_decl_body(pair),
            int_lit => self.build_int_lit(pair),
            float_lit => self.build_float_lit(pair),
            graph_decl_body => self.build_graph_decl_body(pair),

            fn_decls => self.build_fn_decls(pair),
            stmts => self.build_stmts(pair),

            stmt => self.build_stmt(pair),
            expr => self.build_expr(pair),
            tuple => self.build_tuple(pair),
            block => self.build_block(pair),
            pipes => self.build_pipes(pair),
            semicolon => Ok(Term::None),
            _ => unexpected_token(&pair),
        }
    }

    fn build_tuple(&self, pair: Pair<Rule>) -> Result<Term, Diag> {
        let sp = self.cspan.convert_span(&pair.clone().into_span());
        let tokens = pair.into_inner();
        let res = tokens
            .map(|i| self.consume(i).unwrap())
            .collect::<Vec<_>>();

        Ok(Term::Tuple(res, sp))
    }

    fn build_block(&self, pair: Pair<Rule>) -> Result<Term, Diag> {
        let sp = self.cspan.convert_span(&pair.clone().into_span());
        let mut tokens = pair.into_inner();
        let statements = eat!(tokens, stmts, "Cannot parse statements", sp);
        let possible_expr = eat!(tokens, expr, "Does not have a dangling expr", sp);
        let ret = if possible_expr.is_err() {
            Term::None
        } else {
            self.consume(possible_expr?)?
        };

        Ok(Term::Block {
            stmts: Box::new(self.consume(statements?)?),
            ret: Box::new(ret),
            span: sp,
        })
    }

    fn build_fn_app_param(&self, pair: Pair<Rule>) -> Result<Vec<FnAppArg>, Diag> {
        let sp = self.cspan.convert_span(&pair.clone().into_span());
        let mut tokens = pair.into_inner();
        let args = eat!(tokens, fn_app_args, "Does not have args", sp);
        if args.is_err() {
            Ok(vec![])
        } else {
            self.build_fn_app_args(args?)
        }
    }

    fn build_field_access(&self, pair: Pair<Rule>) -> Result<FieldAccess, Diag> {
        let sp = self.cspan.convert_span(&pair.clone().into_span());
        let mut tokens = pair.into_inner();
        let var_name = eat!(tokens, ident, "Failed to parse variable name", sp)?;
        let field_name = eat!(tokens, ident, "Failed to parse field name", sp)?;
        let func_call = eat!(tokens, fn_app_param, "Is not a function call", sp);
        let func_call = if func_call.is_err() {
            None
        } else {
            Some(self.build_fn_app_param(func_call?)?)
        };

        Ok(FieldAccess {
            mod_name: var_name.as_str().to_owned(),
            field_name: field_name.as_str().to_owned(),
            func_call,
            span: sp,
        })
    }

    fn build_expr(&self, pair: Pair<Rule>) -> Result<Term, Diag> {
        let sp = self.cspan.convert_span(&pair.clone().into_span());
        let mut tokens = pair.into_inner();
        let p = tokens.next().unwrap();
        assert!(tokens.next().is_none());
        let val = match p.as_rule() {
            field_access => Term::FieldAccess(self.build_field_access(p).unwrap()),
            fn_app => Term::FnApp(self.build_fn_app(p).unwrap()),
            ident => {
                let sp = self.cspan.convert_span(&p.clone().into_span());
                Term::Ident(p.as_str().to_owned(), sp)
            }
            _ => self.consume(p).unwrap(),
        };
        Ok(Term::Expr(Box::new(val), sp))
    }

    fn build_stmt(&self, pair: Pair<Rule>) -> Result<Term, Diag> {
        let sp = self.cspan.convert_span(&pair.clone().into_span());
        let tokens = pair.into_inner();
        let vals = tokens.map(|p| self.consume(p).unwrap()).collect();
        Ok(Term::Stmt(Box::new(Term::List(vals)), sp))
    }

    fn build_fn_decl(&self, pair: Pair<Rule>) -> Result<FnDecl, Diag> {
        let sp = self.cspan.convert_span(&pair.clone().into_span());
        let mut tokens = pair.into_inner();
        let mut head = eat!(tokens, fn_decl_head, "Failed to parse fn_decl_head", sp)?.into_inner();
        let name = eat!(head, ident, "Failed to parse fn_decl_head ident", sp)?;
        let fn_sig = head.next();
        let func_block = eat!(tokens, block, "Failed to parse function block", sp)?;

        let (fn_params, return_ty) = match fn_sig {
            Some(fn_sig) => {
                let mut fn_sig_tokens = fn_sig.into_inner();
                let param = eat!(fn_sig_tokens, fn_decl_param, "Failed to parse fn_decl_param", sp)?;
                let return_ty = {
                    let temp = eat!(fn_sig_tokens, ty_sig, "Function does not have a type signature", sp);
                    if temp.is_err() {
                        TensorTy::Generic(vec![], sp)
                    } else {
                        let temp = temp?.into_inner().next().unwrap();
                        if temp.as_rule() == cap_ident {
                            TensorTy::Tensor(temp.as_str().to_owned(), sp)
                        } else {
                            TensorTy::Generic(to_idents!(temp), sp)
                        }
                    }
                };

                let params = if let Some(args) = param.into_inner().next() {
                    self.build_fn_decl_params(args)?
                } else {
                    vec![]
                };
                (Some(params), Some(return_ty))
            }
            None => {
                (None, None)
            }
        };

        Ok(FnDecl {
            name: name.as_str().to_owned(),
            fn_params,
            return_ty,
            func_block: Box::new(self.consume(func_block)?),
            span: sp,
        })
    }

    // fn build_fn_decl_sig(pair: Pair<Rule>) -> Result<Term, Diag> {
    //     unimplemented!()
    // }

    fn build_stmts(&self, pair: Pair<Rule>) -> Result<Term, Diag> {
        let tokens = pair.into_inner();
        let vals = tokens.map(|p| self.consume(p).unwrap()).collect();
        Ok(Term::List(vals))
    }

    fn build_fn_decls(&self, pair: Pair<Rule>) -> Result<Term, Diag> {
        let tokens = pair.into_inner();
        let vals = tokens.map(|p| self.consume(p).unwrap()).collect();
        Ok(Term::List(vals))
    }

    fn build_fn_decl_param(&self, pair: Pair<Rule>) -> Result<FnDeclParam, Diag> {
        let sp = self.cspan.convert_span(&pair.clone().into_span());
        let mut tokens = pair.into_inner();
        let param = eat!(tokens, ident, "Failed to parse function parameter", sp)?;
        let typ = eat!(tokens, ty_sig, "Failed to parse type signature", sp);
        let typ = if typ.is_err() {
            TensorTy::Generic(vec![], sp)
        } else {
            TensorTy::Generic(to_idents!(typ?.into_inner().next().unwrap()), sp)
        };

        Ok(FnDeclParam {
            name: param.as_str().to_owned(),
            ty_sig: typ,
            span: sp,
        })
    }

    fn build_fn_decl_params(&self, pair: Pair<Rule>) -> Result<Vec<FnDeclParam>, Diag> {
        let tokens = pair.into_inner();
        let vals = tokens.map(|p| self.build_fn_decl_param(p).unwrap()).collect();
        Ok(vals)
    }

    fn build_view_fn(&self, pair: Pair<Rule>) -> Result<ViewFn, Diag> {
        let sp = self.cspan.convert_span(&pair.clone().into_span());
        let tokens = pair.into_inner();
        let dims = tokens.map(|p| String::from(p.as_str())).collect();
        let span = sp;
        Ok(ViewFn { dims, span })
    }

    fn build_fn_app(&self, pair: Pair<Rule>) -> Result<FnApp, Diag> {
        let sp = self.cspan.convert_span(&pair.clone().into_span());
        let mut tokens = pair.into_inner();
        let name = eat!(tokens, ident, "Cannot parse function call identifier", sp)?;
        let args = if let Some(args) = tokens.next() {
            self.build_fn_app_args(args)?
        } else {
            vec![]
        };

        Ok(FnApp {
            name: name.as_str().to_owned(),
            args,
            span: sp,
        })
    }

    fn build_fn_app_arg(&self, pair: Pair<Rule>) -> Result<FnAppArg, Diag> {
        let sp = self.cspan.convert_span(&pair.clone().into_span());
        let mut tokens = pair.into_inner();
        let param = eat!(tokens, ident, "Failed to parse function call argument", sp)?;
        let param_val = eat!(tokens, expr, "Failed to parse function call parameter", sp)?;

        Ok(FnAppArg {
            name: param.as_str().to_owned(),
            arg: Box::new(self.consume(param_val)?),
            span: sp,
        })
    }

    fn build_fn_app_args(&self, pair: Pair<Rule>) -> Result<Vec<FnAppArg>, Diag> {
        let tokens = pair.into_inner();
        let vals = tokens.map(|p| self.build_fn_app_arg(p).unwrap()).collect();
        Ok(vals)
    }

    fn build_weights_decl(&self, pair: Pair<Rule>) -> Result<Decl, Diag> {
        let sp = self.cspan.convert_span(&pair.clone().into_span());
        let mut tokens = pair.into_inner();
        let mut head = eat!(tokens, weights_decl_head, "Parsing `weight_head` error", sp)?.into_inner();
        let weights_name = eat!(head, cap_ident, "Does not have a weight name", sp)?.as_str();
        let ty_decl = eat!(head, fn_ty_sig, "Failed to parse `fn_ty_sig`", sp)?;
        let weights_body = eat!(
            tokens,
            weights_decl_body,
            "Failed to parse `weights_decl_body`", sp
        )?.into_inner()
            .map(|p| self.build_weights_assign(p).unwrap())
            .collect();

        Ok(Decl::WeightsDecl(WeightsDecl {
            name: weights_name.to_owned(),
            ty_sig: self.build_fn_ty_sig(ty_decl)?,
            inits: weights_body,
            span: sp,
        }))
    }

    fn build_weights_assign(&self, body: Pair<Rule>) -> Result<WeightsAssign, Diag> {
        let sp = self.cspan.convert_span(&body.clone().into_span());
        let mut tokens = body.into_inner();
        let name = eat!(tokens, ident, "Failed to parse ident", sp)?;
        let _assign = eat!(tokens, op_assign, "Failed to parse `=`", sp)?;
        let mod_name = eat!(tokens, cap_ident, "Failed to parse `mod_name`", sp)?;
        let nxt = tokens.next().unwrap();
        if nxt.as_rule() == fn_ty_sig {
            let func = eat!(tokens, fn_app, "Failed to parse `fn_app`", sp)?;
            let fncall = self.build_fn_app(func)?;
            Ok(WeightsAssign {
                name: name.as_str().to_owned(),
                mod_name: mod_name.as_str().to_owned(),
                fn_name: fncall.name,
                mod_sig: Some(self.build_fn_ty_sig(nxt).expect("Cannot parse function type signature!")),
                fn_args: fncall.args,
                span: sp,
            })
        } else {
            let fncall = self.build_fn_app(nxt)?;
            Ok(WeightsAssign {
                name: name.as_str().to_owned(),
                mod_name: mod_name.as_str().to_owned(),
                fn_name: fncall.name,
                mod_sig: None,
                fn_args: fncall.args,
                span: sp,
            })
        }
    }

    fn _process_level(&self, curr: Pair<Rule>) -> Term {
        if curr.as_rule() == fn_app {
            Term::FnApp(self.build_fn_app(curr).unwrap())
        } else if curr.as_rule() == field_access {
            Term::FieldAccess(self.build_field_access(curr).unwrap())
        } else if curr.as_rule() == ident {
            let span = self.cspan.convert_span(&curr.clone().into_span());
            Term::Ident(curr.as_str().to_owned(), span)
        } else if curr.as_rule() == view_fn {
            Term::ViewFn(self.build_view_fn(curr).unwrap())
        } else {
            panic!("{:?}", curr.as_rule());
        }
    }

    fn build_pipes(&self, pair: Pair<Rule>) -> Result<Term, Diag> {
        // linearizes from tree
        let mut exprs = vec![];
        let mut tokens = pair.into_inner(); // [ident, expr]
        loop {
            let curr = tokens.next();
            if curr.is_none() {
                break;
            }
            let curr = curr.unwrap();
            if curr.as_rule() == expr {
                // { expr { pipe | ident | fn_app } }
                let temp = curr.into_inner().next();
                if temp.is_none() {
                    break;
                }
                let temp = temp.unwrap();
                if temp.as_rule() == pipes {
                    tokens = temp.into_inner(); // { pipe | ident | fn_app }
                    continue;
                }
                exprs.push(self._process_level(temp.clone()));
            } else {
                exprs.push(self._process_level(curr).clone());
            }
        }

        // // construct a deep fn_app recursively
        // let mut iter = exprs.iter();
        // let mut init = iter.next().unwrap().to_owned();

        // while let Some(node) = iter.next() {
        //     init = match node {
        //         &Term::Ident(ref name) => Term::FnApp {
        //             name: name.clone(),
        //             args: Box::new(Term::List(vec![
        //                 Term::FnAppArg {
        //                     name: format!("x"),
        //                     arg: Box::new(init),
        //                 }
        //             ]))
        //         },
        //         &Term::FnApp{ref name, ref args} => Term::FnApp {
        //             name: name.clone(),
        //             args: Term::extend_arg_list(args.clone(), init),
        //         },
        //         _ => {
        //             println!("{:?}", node);
        //             unimplemented!()
        //         }
        //     };
        // }

        Ok(Term::Pipes(exprs))
    }

    fn build_graph_decl(&self, pair: Pair<Rule>) -> Result<Decl, Diag> {
        let sp = self.cspan.convert_span(&pair.clone().into_span());
        let mut tokens = pair.into_inner();
        let mut head = eat!(tokens, graph_decl_head, "Parsing `graph_head` error", sp)?.into_inner();
        let node_name = eat!(head, cap_ident, "Does not have a graph name", sp)?.as_str();
        let ty_decl = eat!(head, fn_ty_sig, "Failed to parse `fn_ty_sig`", sp)?;
        let graph_body = eat!(tokens, graph_decl_body, "Failed to parse `graph_decl_body`", sp)?;
        let func_decls = graph_body
            .into_inner()
            .next()
            .unwrap()
            .into_inner()
            .map(|f| self.build_fn_decl(f))
            .collect::<Result<_,_>>()?;

        Ok(Decl::GraphDecl(GraphDecl {
            name: node_name.to_owned(),
            ty_sig: self.build_fn_ty_sig(ty_decl)?,
            fns: func_decls,
            span: sp,
        }))
    }

    fn build_graph_decl_body(&self, pair: Pair<Rule>) -> Result<Term, Diag> {
        let sp = self.cspan.convert_span(&pair.clone().into_span());
        let mut tokens = pair.into_inner();
        let fns = eat!(tokens, fn_decls, "Failed to parse `fn_decls`", sp)?;
        let vals = fns.into_inner().map(|p| self.consume(p).unwrap()).collect();
        Ok(Term::List(vals))
    }

    fn build_node_decl(&self, pair: Pair<Rule>) -> Result<Decl, Diag> {
        let sp = self.cspan.convert_span(&pair.clone().into_span());
        let mut tokens = pair.into_inner();
        let mut head = eat!(tokens, node_decl_head, "Parsing `node_head` error", sp)?.into_inner();
        let node_name = eat!(head, cap_ident, "Does not have a node name", sp)?.as_str();
        let ty_decl = eat!(head, fn_ty_sig, "Failed to parse `fn_ty_sig`", sp)?;
        let node_body = eat!(tokens, node_decl_body, "Failed to parse `node_decl_body`", sp)?;

        let ty_signature = self.build_fn_ty_sig(ty_decl)?;

        let macros = node_body.into_inner();
        let macros = macros.map(|p| self.build_node_assign(p).unwrap()).collect();

        Ok(Decl::NodeDecl(NodeDecl {
            name: node_name.to_owned(),
            ty_sig: ty_signature,
            defs: macros,
            span: sp,
        }))
    }

    // fn build_node_decl_body(body: Pair<Rule>) -> Result<Term, Diag> {
    //     let tokens = body.into_inner();
    //     let vals = tokens.map(|p| build_node_assign(p).unwrap()).collect();

    //     Ok(Term::List(vals))
    // }

    fn build_node_assign(&self, pair: Pair<Rule>) -> Result<NodeAssign, Diag> {
        if pair.as_rule() != node_assign {
            let errmsg = format!("ty mismatch: {:?}", node_assign);
            return Err(err!(errmsg, self.cspan.convert_span(&pair.clone().into_span())));
        }
        let sp = self.cspan.convert_span(&pair.clone().into_span());
        let mut tokens = pair.into_inner();
        let identifier = eat!(tokens, ident, "Failed to parse `ident`", sp)?;
        let _assign = eat!(tokens, op_assign, "Cannot parse `=`", sp)?;

        let identifier = identifier.as_str().to_owned();

        let handle_lit = move |token: Pair<Rule>, id: String, sp: ByteSpan| {
            let lit = self.consume(token)?;
            Ok(NodeAssign::Dimension {
                ident: id,
                rhs: lit,
                span: sp,
            })
        };

        let handle_ty = move |ty: Pair<Rule>, id: String, sp: ByteSpan| {
            let ty = to_idents!(ty);
            Ok(NodeAssign::Tensor {
                ident: id,
                rhs: TensorTy::Generic(ty, sp),
                span: sp,
            })
        };

        let tok = tokens.next().unwrap();
        match tok.as_rule() {
            int_lit => handle_lit(tok, identifier, sp),
            float_lit => handle_lit(tok, identifier, sp),
            ty_ident_list => handle_ty(tok, identifier, sp),
            _ => unimplemented!(),
        }
    }

    fn build_float_lit(&self, pair: Pair<Rule>) -> Result<Term, Diag> {
        let ret = pair.as_str().parse().unwrap();
        let span = self.cspan.convert_span(&pair.into_span());
        Ok(Term::Float(ret, span))
    }

    fn build_int_lit(&self, pair: Pair<Rule>) -> Result<Term, Diag> {
        let ret = pair.as_str().parse().unwrap();
        let span = self.cspan.convert_span(&pair.into_span());
        Ok(Term::Integer(ret, span))
    }

    fn build_fn_ty_sig(&self, pair: Pair<Rule>) -> Result<FnTySig, Diag> {
        let sp = self.cspan.convert_span(&pair.clone().into_span());
        let mut tokens = pair.into_inner();

        let handle_tensor_ty = |token: Pair<Rule>| TensorTy::Generic(to_idents!(token), sp);
        let handle_alias = |token: Pair<Rule>| TensorTy::Tensor(token.as_str().to_owned(), sp);
        let handle = |tok: Pair<Rule>| match tok.as_rule() {
            ty_ident_list => handle_tensor_ty(tok),
            ident => handle_alias(tok),
            _ => unimplemented!(),
        };

        let tok = tokens.next().unwrap();
        let from_ty = handle(tok);

        let tok = tokens.next().unwrap();
        let to_ty = handle(tok);

        Ok(FnTySig {
            from: from_ty,
            to: to_ty,
        })
    }

    fn build_use_stmt(&self, pair: Pair<Rule>) -> Result<Decl, Diag> {
        let sp = self.cspan.convert_span(&pair.clone().into_span());
        let mut tokens = pair.into_inner();
        let _use_lit = eat!(tokens, use_lit, "Parsing `use` error", sp)?;
        let module_name = eat!(tokens, ident, "module name not defined", sp)?.as_str();
        let imported = eat!(tokens, "no imported modules", sp)?;

        let mut imported_tokens = vec![];
        match imported.as_rule() {
            Rule::ident_list => imported
                .into_inner()
                .map(|tok| imported_tokens.push(tok.as_str().to_owned()))
                .collect(),
            Rule::ident => imported_tokens.push(imported.as_str().to_owned()),
            _ => unexpected_token(&imported),
        };

        Ok(Decl::UseStmt(UseStmt {
            mod_name: module_name.to_owned(),
            imported_names: imported_tokens,
            span: sp,
        }))
    }

}

fn unexpected_token(pair: &Pair<Rule>) -> ! {
    let message = format!("Unexpected token: {:#}", pair);
    panic!(message);
}
