macro_rules! err {
    ($msg:expr) => {
        TSSParseError {
            msg: $msg.to_owned(),
        }
    };
}

macro_rules! eat {
    ($tokens:expr, $err:expr) => {
        $tokens.next()
            .ok_or(err!($err))
    };


    ($tokens:expr, $rule:ident, $err:expr) => {
        $tokens.next()
            .ok_or(err!($err))
            .and_then(|val| {
                if Rule::$rule != val.as_rule() {
                    Err(err!(&format!("Type is not {:?}", $rule)))
                } else {
                    Ok(val)
                }
            })
    };

    ($tokens:expr, [$( $rule:ident ),+], $err:expr) => {
        $tokens.next()
            .ok_or(err!($err))
            .and_then(|val| {
                $(
                    if Rule::$rule == val.as_rule() {
                        return Ok(val);
                    }
                )*
                return Err(err!("Type is wrong"))
            })
    };
}

macro_rules! to_idents {
    ($ident_list: expr) => {
        $ident_list.into_inner()
            .map(|id| id.as_str())
            .map(String::from)
            .collect()
    };
}

use parser::ast::{Decl, FieldAccess, FnCall, FnCallArg, FnDecl, FnDeclArg, FnTySig, GraphDecl,
          MacroAssign, Module, NodeDecl, Program, TensorTy, UseStmt, WeightsAssign, WeightsDecl,
          AST};
use parser::grammar::{Rule, TensorScriptParser};
use parser::grammar::Rule::*;
use pest::iterators::{Pair};
use pest::Parser;

#[derive(Debug)]
pub struct TSSParseError {
    msg: String,
}


pub fn parse_str(source: &str) -> Result<Program, TSSParseError> {
    // let program = TensorScriptParser::parse(Rule::fn_ty_sig, "<image->labels>");
    // println!("{}", program.unwrap());
    // unimplemented!();

    let parser = TensorScriptParser::parse(Rule::input, source);
    if parser.is_err() {
        panic!(format!("{:#}", parser.err().unwrap()));
    }

    let decls = parser
        .unwrap()
        .map(|pair| match pair.as_rule() {
            use_stmt => build_use_stmt(pair).unwrap(),
            weights_decl => build_weights_decl(pair).unwrap(),
            graph_decl => build_graph_decl(pair).unwrap(),
            node_decl => build_node_decl(pair).unwrap(),
            _ => panic!("Only node, graph, weights, use supported at top level"),
        })
        .collect();
    Ok(Program {
        module: Module { decls: decls },
    })
}

fn consume(pair: Pair<Rule>) -> Result<AST, TSSParseError> {
    // println!("{}", pair);
    match pair.as_rule() {
        // node_decl_body => build_node_decl_body(pair),
        int_lit => build_int_lit(pair),
        float_lit => build_float_lit(pair),
        graph_decl_body => build_graph_decl_body(pair),

        fn_decls => build_fn_decls(pair),
        fn_decl_sig => build_fn_decl_sig(pair),
        stmts => build_stmts(pair),

        stmt => build_stmt(pair),
        expr => build_expr(pair),
        block => build_block(pair),
        pipes => build_pipes(pair),

        // Rule::statements                  => build_block(pair),
        // Rule::integer_zero_literal        => integer!(0),
        // Rule::integer_binary_literal      => build_integer(pair, 2),
        // Rule::integer_octal_literal       => build_integer(pair, 8),
        // Rule::integer_decimal_literal     => build_integer(pair, 10),
        // Rule::integer_hexadecimal_literal => build_integer(pair, 16),
        // Rule::float_literal               => build_float(pair),
        // Rule::atom_literal                => build_atom(pair),
        // Rule::string_literal              => build_string(pair),
        // Rule::bool_true                   => boolean!(true),
        // Rule::bool_false                  => boolean!(false),
        // Rule::unary_not                   => unary_not!(consume(pair.into_inner().next().unwrap())),
        // Rule::unary_complement            => unary_complement!(consume(pair.into_inner().next().unwrap())),
        // Rule::unary_plus                  => unary_plus!(consume(pair.into_inner().next().unwrap())),
        // Rule::unary_minus                 => unary_minus!(consume(pair.into_inner().next().unwrap())),
        // Rule::braced_expression           => braced!(consume(pair.into_inner().next().unwrap())),
        // Rule::const_literal               => build_const(pair),
        // Rule::local_var_access            => build_lvar_access(pair),
        // Rule::call_with_implicit_receiver => build_implicit_call(pair.into_inner().next().unwrap()),
        // Rule::call_with_explicit_receiver => build_explicit_call(pair),
        // Rule::list_literal                => build_list(pair),
        // Rule::index                       => build_index(pair),
        // Rule::function_literal            => build_function(pair),
        // Rule::block                       => build_block(pair),
        // Rule::map_literal                 => build_map(pair),
        _ => unexpected_token(pair),
    }
}

fn build_block(pair: Pair<Rule>) -> Result<AST, TSSParseError> {
    let mut tokens = pair.into_inner();
    let statements = eat!(tokens, stmts, "Cannot parse statements");
    let possible_expr = eat!(tokens, expr, "Does not have a dangling expr");
    let ret = if possible_expr.is_err() {
        AST::None
    } else {
        consume(possible_expr?)?
    };

    Ok(AST::Block {
        stmts: Box::new(consume(statements?)?),
        ret: Box::new(ret),
    })
}

fn build_fn_app_param(pair: Pair<Rule>) -> Result<Vec<FnCallArg>, TSSParseError> {
    let mut tokens = pair.into_inner();
    let args = eat!(tokens, fn_app_args, "Does not have args");
    if args.is_err() {
        Ok(vec![])
    } else {
        build_fn_app_args(args?)
    }
}

fn build_field_access(pair: Pair<Rule>) -> Result<FieldAccess, TSSParseError> {
    let mut tokens = pair.into_inner();
    let var_name = eat!(tokens, ident, "Failed to parse variable name")?;
    let field_name = eat!(tokens, ident, "Failed to parse field name")?;
    let func_call = eat!(tokens, fn_app_param, "Is not a function call");
    let func_call = if func_call.is_err() {
        None
    } else {
        Some(build_fn_app_param(func_call?)?)
    };

    Ok(FieldAccess {
        var_name: var_name.as_str().to_owned(),
        field_name: field_name.as_str().to_owned(),
        func_call: func_call,
    })
}

fn build_expr(pair: Pair<Rule>) -> Result<AST, TSSParseError> {
    let tokens = pair.into_inner();
    let vals = tokens
        .map(|p| match p.as_rule() {
            field_access => AST::FieldAccess(build_field_access(p).unwrap()),
            fn_app => AST::FnCall(build_fn_app(p).unwrap()),
            _ => consume(p).unwrap(),
        })
        .collect();
    Ok(AST::Expr {
        items: Box::new(AST::List(vals)),
    })
}

fn build_stmt(pair: Pair<Rule>) -> Result<AST, TSSParseError> {
    let tokens = pair.into_inner();
    let vals = tokens.map(|p| consume(p).unwrap()).collect();
    Ok(AST::Stmt {
        items: Box::new(AST::List(vals)),
    })
}

fn build_fn_decl(pair: Pair<Rule>) -> Result<FnDecl, TSSParseError> {
    let mut tokens = pair.into_inner();
    let mut head = eat!(tokens, fn_decl_head, "Failed to parse fn_decl_head")?.into_inner();
    let name = eat!(head, ident, "Failed to parse fn_decl_head ident")?;
    let fn_sig = eat!(head, fn_decl_sig, "Failed to parse fn decl signature")?;
    let func_block = eat!(tokens, block, "Failed to parse function block")?;

    let mut tokens = fn_sig.into_inner();
    let param = eat!(tokens, fn_decl_param, "Failed to parse fn_decl_param")?;
    let return_ty = {
        let temp = eat!(tokens, ty_sig, "Function does not have a type signature");
        if temp.is_err() {
            TensorTy::Generic(vec![])
        } else {
            TensorTy::Generic(to_idents!(temp?))
        }
    };

    let params = if let Some(args) = param.into_inner().next() {
        build_fn_decl_args(args)?
    } else {
        vec![]
    };

    Ok(FnDecl {
        name: name.as_str().to_owned(),
        fn_params: params,
        return_ty: return_ty,
        func_block: Box::new(consume(func_block)?),
    })
}

fn build_fn_decl_sig(pair: Pair<Rule>) -> Result<AST, TSSParseError> {
    unimplemented!()
}

fn build_stmts(pair: Pair<Rule>) -> Result<AST, TSSParseError> {
    let tokens = pair.into_inner();
    let vals = tokens.map(|p| consume(p).unwrap()).collect();
    Ok(AST::List(vals))
}

fn build_fn_decls(pair: Pair<Rule>) -> Result<AST, TSSParseError> {
    let tokens = pair.into_inner();
    let vals = tokens.map(|p| consume(p).unwrap()).collect();
    Ok(AST::List(vals))
}

fn build_fn_decl_arg(pair: Pair<Rule>) -> Result<FnDeclArg, TSSParseError> {
    let mut tokens = pair.into_inner();
    let param = eat!(tokens, ident, "Failed to parse function parameter")?;
    let typ = eat!(tokens, ty_sig, "Failed to parse type signature");
    let typ = if typ.is_err() {
        TensorTy::Generic(vec![])
    } else {
        TensorTy::Generic(to_idents!(typ?))
    };

    Ok(FnDeclArg {
        name: param.as_str().to_owned(),
        ty_sig: typ,
    })
}

fn build_fn_decl_args(pair: Pair<Rule>) -> Result<Vec<FnDeclArg>, TSSParseError> {
    let tokens = pair.into_inner();
    let vals = tokens.map(|p| build_fn_decl_arg(p).unwrap()).collect();
    Ok(vals)
}

fn build_fn_app(pair: Pair<Rule>) -> Result<FnCall, TSSParseError> {
    let mut tokens = pair.into_inner();
    let name = eat!(tokens, ident, "Cannot parse function call identifier")?;
    let args = if let Some(args) = tokens.next() {
        build_fn_app_args(args)?
    } else {
        vec![]
    };

    Ok(FnCall {
        name: name.as_str().to_owned(),
        args: args,
    })
}

fn build_fn_app_arg(pair: Pair<Rule>) -> Result<FnCallArg, TSSParseError> {
    let mut tokens = pair.into_inner();
    let param = eat!(tokens, ident, "Failed to parse function call argument")?;
    let param_val = eat!(tokens, expr, "Failed to parse function call parameter")?;

    Ok(FnCallArg {
        name: param.as_str().to_owned(),
        arg: Box::new(consume(param_val)?),
    })
}

fn build_fn_app_args(pair: Pair<Rule>) -> Result<Vec<FnCallArg>, TSSParseError> {
    let tokens = pair.into_inner();
    let vals = tokens.map(|p| build_fn_app_arg(p).unwrap()).collect();
    Ok(vals)
}

fn build_weights_decl(pair: Pair<Rule>) -> Result<Decl, TSSParseError> {
    let mut tokens = pair.into_inner();
    let mut head = eat!(tokens, weights_decl_head, "Parsing `weight_head` error")?.into_inner();
    let weights_name = eat!(head, cap_ident, "Does not have a weight name")?.as_str();
    let ty_decl = eat!(head, fn_ty_sig, "Failed to parse `fn_ty_sig`")?;
    let weights_body = eat!(
        tokens,
        weights_decl_body,
        "Failed to parse `weights_decl_body`"
    )?.into_inner()
        .map(|p| build_weights_assign(p).unwrap())
        .collect();

    Ok(Decl::WeightsDecl(WeightsDecl {
        name: weights_name.to_owned(),
        ty_sig: build_fn_ty_sig(ty_decl)?,
        initialization: weights_body,
    }))
}

fn build_weights_assign(body: Pair<Rule>) -> Result<WeightsAssign, TSSParseError> {
    let mut tokens = body.into_inner();
    let name = eat!(tokens, ident, "Failed to parse ident")?;
    let _assign = eat!(tokens, op_assign, "Failed to parse `=`")?;
    let mod_name = eat!(tokens, cap_ident, "Failed to parse `mod_name`")?;
    let fn_sig = eat!(tokens, fn_ty_sig, "Failed to parse `fn_sig`")?;
    let func = eat!(tokens, fn_app, "Failed to parse `fn_app`")?;

    Ok(WeightsAssign {
        name: name.as_str().to_owned(),
        mod_name: mod_name.as_str().to_owned(),
        mod_sig: build_fn_ty_sig(fn_sig)?,
        func: build_fn_app(func)?,
    })
}

fn _process_level(curr: Pair<Rule>) -> AST {
    if curr.as_rule() == fn_app {
        AST::FnCall(build_fn_app(curr).unwrap())
    } else if curr.as_rule() == field_access {
        AST::FieldAccess(build_field_access(curr).unwrap())
    } else if curr.as_rule() == ident {
        AST::Ident(curr.as_str().to_owned())
    } else {
        println!("{:?}", curr.as_rule());
        unimplemented!()
    }
}

fn build_pipes(pair: Pair<Rule>) -> Result<AST, TSSParseError> {
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
            exprs.push(_process_level(temp.clone()));
        } else {
            exprs.push(_process_level(curr).clone());
        }
    }

    // // construct a deep fn_app recursively
    // let mut iter = exprs.iter();
    // let mut init = iter.next().unwrap().to_owned();

    // while let Some(node) = iter.next() {
    //     init = match node {
    //         &AST::Ident(ref name) => AST::FnCall {
    //             name: name.clone(),
    //             args: Box::new(AST::List(vec![
    //                 AST::FnCallArg {
    //                     name: format!("x"),
    //                     arg: Box::new(init),
    //                 }
    //             ]))
    //         },
    //         &AST::FnCall{ref name, ref args} => AST::FnCall {
    //             name: name.clone(),
    //             args: AST::extend_arg_list(args.clone(), init),
    //         },
    //         _ => {
    //             println!("{:?}", node);
    //             unimplemented!()
    //         }
    //     };
    // }

    Ok(AST::Pipes(exprs))
}

fn build_graph_decl(pair: Pair<Rule>) -> Result<Decl, TSSParseError> {
    let mut tokens = pair.into_inner();
    let mut head = eat!(tokens, graph_decl_head, "Parsing `graph_head` error")?.into_inner();
    let node_name = eat!(head, cap_ident, "Does not have a graph name")?.as_str();
    let ty_decl = eat!(head, fn_ty_sig, "Failed to parse `fn_ty_sig`")?;
    let graph_body = eat!(tokens, graph_decl_body, "Failed to parse `graph_decl_body`")?;
    let func_decls = graph_body
        .into_inner()
        .next()
        .unwrap()
        .into_inner()
        .map(|f| build_fn_decl(f).unwrap())
        .collect();

    Ok(Decl::GraphDecl(GraphDecl {
        name: node_name.to_owned(),
        ty_sig: build_fn_ty_sig(ty_decl)?,
        fns: func_decls,
    }))
}

fn build_graph_decl_body(pair: Pair<Rule>) -> Result<AST, TSSParseError> {
    let mut tokens = pair.into_inner();
    let fns = eat!(tokens, fn_decls, "Failed to parse `fn_decls`")?;
    let vals = fns.into_inner().map(|p| consume(p).unwrap()).collect();
    Ok(AST::List(vals))
}

fn build_node_decl(pair: Pair<Rule>) -> Result<Decl, TSSParseError> {
    let mut tokens = pair.into_inner();
    let mut head = eat!(tokens, node_decl_head, "Parsing `node_head` error")?.into_inner();
    let node_name = eat!(head, cap_ident, "Does not have a node name")?.as_str();
    let ty_decl = eat!(head, fn_ty_sig, "Failed to parse `fn_ty_sig`")?;
    let node_body = eat!(tokens, node_decl_body, "Failed to parse `node_decl_body`")?;

    let ty_signature = build_fn_ty_sig(ty_decl)?;

    let macros = node_body.into_inner();
    let macros = macros
        .map(|p| build_node_macro_assign(p).unwrap())
        .collect();

    Ok(Decl::NodeDecl(NodeDecl {
        name: node_name.to_owned(),
        ty_sig: ty_signature,
        initialization: macros,
    }))
}

// fn build_node_decl_body(body: Pair<Rule>) -> Result<AST, TSSParseError> {
//     let tokens = body.into_inner();
//     let vals = tokens.map(|p| build_node_macro_assign(p).unwrap()).collect();

//     Ok(AST::List(vals))
// }

fn build_node_macro_assign(pair: Pair<Rule>) -> Result<MacroAssign, TSSParseError> {
    if pair.as_rule() != node_macro_assign {
        return Err(err!(format!("ty mismatch: {:?}", node_macro_assign)));
    }
    let mut tokens = pair.into_inner();
    let identifier = eat!(tokens, upper_ident, "Failed to parse `upper_ident`")?;
    let _assign = eat!(tokens, op_assign, "Cannot parse `=`")?;

    let identifier = identifier.as_str().to_owned();

    let handle_lit = move |token: Pair<Rule>, id: String| {
        let lit = consume(token)?;
        Ok(MacroAssign::ValueAlias {
            ident: id,
            rhs: Box::new(lit),
        })
    };

    let handle_ty = move |ty: Pair<Rule>, id: String| {
        let ty = to_idents!(ty);
        Ok(MacroAssign::TyAlias {
            ident: id,
            rhs: TensorTy::Generic(ty),
        })
    };

    let tok = tokens.next().unwrap();
    match tok.as_rule() {
        int_lit => handle_lit(tok, identifier),
        float_lit => handle_lit(tok, identifier),
        ty_ident_list => handle_ty(tok, identifier),
        _ => unimplemented!(),
    }
}

fn build_float_lit(pair: Pair<Rule>) -> Result<AST, TSSParseError> {
    let ret = pair.as_str().parse().unwrap();
    Ok(AST::Float(ret))
}

fn build_int_lit(pair: Pair<Rule>) -> Result<AST, TSSParseError> {
    let ret = pair.as_str().parse().unwrap();
    Ok(AST::Integer(ret))
}

fn build_fn_ty_sig(pair: Pair<Rule>) -> Result<FnTySig, TSSParseError> {
    let mut tokens = pair.into_inner();

    let handle_tensor_ty = |token: Pair<Rule>| TensorTy::Generic(to_idents!(token));
    let handle_alias = |token: Pair<Rule>| TensorTy::TyAlias(token.as_str().to_owned());
    let handle = |tok: Pair<Rule>| match tok.as_rule() {
        ty_ident_list => handle_tensor_ty(tok),
        cap_ident => handle_alias(tok),
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

fn build_use_stmt(pair: Pair<Rule>) -> Result<Decl, TSSParseError> {
    let mut tokens = pair.into_inner();
    let value = eat!(tokens, use_lit, "Parsing `use` error")?;
    let module_name = eat!(tokens, ident, "module name not defined")?.as_str();
    let imported = eat!(tokens, "no imported modules")?;

    let mut imported_tokens = vec![];
    match imported.as_rule() {
        Rule::ident_list => imported
            .into_inner()
            .map(|tok| imported_tokens.push(tok.as_str().to_owned()))
            .collect(),
        Rule::ident => imported_tokens.push(imported.as_str().to_owned()),
        _ => unexpected_token(imported),
    };

    Ok(Decl::UseStmt(UseStmt {
        mod_name: module_name.to_owned(),
        imported_names: imported_tokens,
    }))
}

fn unexpected_token(pair: Pair<Rule>) -> ! {
    let message = format!("Unexpected token: {:#}", pair);
    panic!(message);
}
