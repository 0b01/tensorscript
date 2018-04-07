use ast::AST;
use grammar::{TensorScriptParser, Rule};
use pest::iterators::Pair;
use pest::Parser;

#[derive(Debug)]
pub struct TSSParseError {
    msg: String,
}

use grammar::Rule::*;


pub fn parse_str(source: &str) -> Result<AST, TSSParseError> {
    let parser = TensorScriptParser::parse(Rule::input, source);
    if parser.is_err() { panic!(format!("{:#}", parser.err().unwrap())); }

    let stmts = parser.unwrap()
        .map(|pair|consume(pair).unwrap())
        .collect();
    Ok(AST::List(stmts))
}

pub fn consume(pair: Pair<Rule>) -> Result<AST, TSSParseError> {
    println!("{}", pair);
    match pair.as_rule() {
        use_stmt                        => build_use_stmt(pair),
        node_decl                       => build_node_decl(pair),
        fn_type_sig                     => build_type_sig(pair),
        node_decl_body                  => build_node_decl_body(pair),
        node_macro_assign               => build_node_macro_assign(pair),
        int_lit                         => build_int_lit(pair),
        float_lit                       => build_float_lit(pair),
        weights_decl                    => build_weights_decl(pair),
        weights_decl_body               => build_weights_decl_body(pair),
        weights_assign                  => build_weights_assign(pair),
        fn_call                         => build_fn_call(pair),
        fn_call_args                    => build_fn_call_args(pair),
        fn_call_arg                     => build_fn_call_arg(pair),
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

macro_rules! err {
    ($msg:expr) => {
        TSSParseError {
            msg: $msg.to_owned(),
        }
    };
}

macro_rules! eat {
    ($values:expr, $err:expr) => {
        $values.next()
            .ok_or(err!($err))
    };


    ($values:expr, $rule:ident, $err:expr) => {
        $values.next()
            .ok_or(err!($err))
            .and_then(|val| {
                if Rule::$rule != val.as_rule() {
                    Err(err!(&format!("Type is not {:?}", $rule)))
                } else {
                    Ok(val)
                }
            })
    };

    ($values:expr, [$( $rule:ident ),+], $err:expr) => {
        $values.next()
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
        $ident_list?.into_inner()
            .map(|id| id.as_str())
            .map(String::from)
            .collect()
    };
}

fn build_fn_call_arg(pair: Pair<Rule>) -> Result<AST, TSSParseError> {
    let mut values = pair.into_inner();
    let param = eat!(values, ident, "Failed to parse function call argument")?;
    let param_val = eat!(values, [int_lit, float_lit], "Failed to parse function call parameter")?;

    Ok(AST::FnCallArg {
        name: param.as_str().to_owned(),
        arg: Box::new(consume(param_val)?),
    })
}

fn build_fn_call(pair: Pair<Rule>) -> Result<AST, TSSParseError> {
    let mut values = pair.into_inner();
    let name = eat!(values, ident, "Cannot parse function call identifier")?;
    let args = if let Some(args) = values.next() {
        if args.as_rule() == fn_call_args {
            consume(args)?
        } else {
            return Err(err!("Unsupported function arguments"));
        }
    } else {
        AST::List(vec![])
    };

    Ok(AST::FnCall {
        name: name.as_str().to_owned(),
        args: Box::new(args),
    })
}

fn build_fn_call_args(pair: Pair<Rule>) -> Result<AST, TSSParseError> {
    let mut values = pair.into_inner();
    let vals = values.map(|p| {
            consume(p).unwrap()
        })
        .collect();

    Ok(AST::List(vals))
}

fn build_weights_decl(pair: Pair<Rule>) -> Result<AST, TSSParseError> {
    let mut values = pair.into_inner();
    let mut head = eat!(values, weights_decl_head, "Parsing `weight_head` error")?.into_inner();
    let weights_name = eat!(head, cap_ident, "Does not have a weight name")?.as_str();
    let type_decl = eat!(head, fn_type_sig, "Failed to parse `fn_type_sig`")?;
    let weights_body = eat!(values, weights_decl_body, "Failed to parse `weights_decl_body`")?;

    Ok(AST::WeightsDecl {
        name: weights_name.to_owned(),
        type_sig: Box::new(consume(type_decl)?),
        initialization: Box::new(consume(weights_body)?),
    })
}

fn build_weights_decl_body(body: Pair<Rule>) -> Result<AST, TSSParseError> {
    let values = body.into_inner();
    let vals = values.map(|p| {
        consume(p).unwrap()
    })
    .collect();

    Ok(AST::List(vals))
}

fn build_weights_assign(body: Pair<Rule>) -> Result<AST, TSSParseError> {
    let mut values = body.into_inner();
    let name = eat!(values, ident, "Failed to parse ident")?;
    let _assign = eat!(values, op_assign, "Failed to parse `=`")?;
    let mod_name = eat!(values, cap_ident, "Failed to parse `mod_name`")?;
    let fn_sig = eat!(values, fn_type_sig, "Failed to parse `fn_sig`")?;
    let func = eat!(values, fn_call, "Failed to parse `fn_call`")?;

    Ok(AST::WeightsAssign {
        name: name.as_str().to_owned(),
        mod_name: mod_name.as_str().to_owned(),
        mod_sig: Box::new(consume(fn_sig)?),
        func: Box::new(consume(func)?),
    })
}

fn build_node_decl(pair: Pair<Rule>) -> Result<AST, TSSParseError> {
    let mut values = pair.into_inner();
    let mut head = eat!(values, node_decl_head, "Parsing `node_head` error")?.into_inner();
    let node_name = eat!(head, cap_ident, "Does not have a node name")?.as_str();
    let type_decl = eat!(head, fn_type_sig, "Failed to parse `fn_type_sig`")?;
    let node_body = eat!(values, node_decl_body, "Failed to parse `node_decl_body`")?;

    Ok(AST::NodeDecl {
        name: node_name.to_owned(),
        type_sig: Box::new(consume(type_decl)?),
        initialization: Box::new(consume(node_body)?),
    })
}

fn build_node_decl_body(body: Pair<Rule>) -> Result<AST, TSSParseError> {
    let values = body.into_inner();
    let vals = values.map(|p| {
        consume(p).unwrap()
    })
    .collect();

    Ok(AST::List(vals))
}

fn build_node_macro_assign(pair: Pair<Rule>) -> Result<AST, TSSParseError> {
    if pair.as_rule() != node_macro_assign {
        return Err(err!(format!("Type mismatch: {:?}", node_macro_assign)));
    }
    let mut values = pair.into_inner();
    let identifier = eat!(values, upper_ident, "Failed to parse `upper_ident`")?;
    let _assign = eat!(values, op_assign, "Cannot parse `=`")?;
    let lit = eat!(values, [int_lit, float_lit], "Cannot parse literal")?;

    let identifier = identifier.as_str().to_owned();
    let lit = consume(lit)?;

    Ok(AST::MacroAssign(identifier, Box::new(lit)))
}

fn build_float_lit(pair: Pair<Rule>) -> Result<AST, TSSParseError> {
    let ret = pair.as_str().parse().unwrap();
    Ok(AST::Float(ret))
}

fn build_int_lit(pair: Pair<Rule>) -> Result<AST, TSSParseError> {
    let ret = pair.as_str().parse().unwrap();
    Ok(AST::Integer(ret))
}

fn build_type_sig(pair: Pair<Rule>) -> Result<AST, TSSParseError> {
    let mut values = pair.into_inner();
    let ident_list_from = eat!(values, type_ident_list, "Cannot parse type_ident_list");
    let ident_list_to = eat!(values, type_ident_list, "Cannot parse type_ident_list");
    let from_type = to_idents!(ident_list_from);
    let to_type = to_idents!(ident_list_to);

    Ok(AST::TypeSig(from_type, to_type))
}

fn build_use_stmt(pair: Pair<Rule>) -> Result<AST, TSSParseError> {
    let mut values = pair.into_inner();
    let value = eat!(values, use_lit, "Parsing `use` error")?;
    let module_name = eat!(values, ident, "module name not defined")?.as_str();
    let imported = eat!(values, "no imported modules")?;
    
    let mut imported_tokens = vec![];
    match imported.as_rule() {
        Rule::ident_list    => imported.into_inner().map(
                                    |tok| imported_tokens.push(tok.as_str().to_owned())
                                ).collect(),
        Rule::ident         => imported_tokens.push(imported.as_str().to_owned()),
        _ => unexpected_token(imported),
    };

    Ok(AST::UseStmt(module_name.to_owned(), imported_tokens))
}

fn unexpected_token(pair: Pair<Rule>) -> ! {
    let message = format!("Unexpected token: {:#}", pair);
    panic!(message);
}