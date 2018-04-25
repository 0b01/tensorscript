#![feature(iterator_flatten)]
#![feature(box_syntax)]
#![feature(box_patterns)]

/// How it works:
/// 1. PEG parser parses into token tree. The downside of PEG parser is that
/// it is mostly magic, which means either it works or not, very difficult
/// to debug or rigorously test other than trial and error. The Pest crate handles
/// lexing and parsing in this compiler.
///
/// 2. Parses token tree into untyped AST. This constructs a simple traversable tree
/// structure for quality of life. The typing step might as well be merged to this part.
///
/// 3. Annotate untyped AST into typed AST for type inference and reconstruction. The
/// idea is to annotate each tree node with a dummy type variable.
///
/// 4. Hindley-Milner type inference. This is consisted of a few substeps. See module
/// documentation for more details.
///
/// 5. ...
///
/// TODO:
/// 1. [*] implement module pattern matching
/// 2. [ ] type level computation (resolved tensor dimension)
/// 3. [*] BUG: dimension mismatch for mnist example
///             need to create fresh type variables for different static forward functions
/// 4. [*] BUG: non-determinism
/// 5. [*] BUG: impl Hash, Eq for Type
/// 6. [*] set up examples and tests
/// 7. [*] set up commandline
/// 8. [*] more examples
/// 9. [*] better errors in parser

extern crate pest;
#[macro_use]
mod typing;
#[macro_use]
extern crate pest_derive;
#[macro_use]
extern crate maplit;

extern crate codespan;
extern crate clap;
extern crate codespan_reporting;

mod core;
mod parser;
mod span;
mod errors;

use std::fs::File;
use std::io::Read;

use typing::constraint::Constraints;
use typing::unifier::Unifier;
use typing::annotate::annotate;
use typing::type_env::TypeEnv;
use typing::inferred_ast::subs;

use codespan::CodeMap;
use clap::{Arg, App, ArgMatches};

fn get_matches<'a>() -> ArgMatches<'a> {
    App::new("tsrc")
        .version("0.0")
        .author("Ricky Han <rickylqhan@gmail.com>")
        .about("Compiler for Tensorscript")
        .arg(Arg::with_name("input")
            .short("f")
            .long("in")
            .value_name("FILE")
            .help("Sets a custom config file")
            .takes_value(true)
            .required(true))
        .arg(Arg::with_name("v")
            .short("v")
            .multiple(true)
            .help("Sets the level of verbosity"))
        .get_matches()
}

fn main() {
    // --------------- get command line options -----------------
    let matches = get_matches();
    let fname = matches.value_of("input").unwrap();

    let mut file = File::open(fname).expect("Unable to open the file");
    let mut src = String::new();
    file.read_to_string(&mut src).expect("Unable to read the file");

    // --------------- parse into untyped ast   -----------------

    let mut code_map = CodeMap::new();
    let file_map = code_map.add_filemap("test".into(), src.clone());
    let cspan = span::CSpan::new(file_map.span());

    let parsed_terms = parser::parse_str(&src, &cspan);
    let program = match parsed_terms {
        Ok(program) => program,
        Err(e) => {
            e.print_err(&code_map);
            return;
        },
    };

    // ------------- annotate ast with type vars --------------

    let mut tenv = TypeEnv::new();
    let ast = annotate(&program, &mut tenv);
    // println!("{}", ast);
    // println!("initial tenv: {:#?}", tenv);

    // ------------ first unitfication pass ---------------
    let mut cs = Constraints::new(); cs.collect(&ast, &mut tenv);               // collect constraints
    // println!("{:#?}", cs);
    let mut unifier = Unifier::new();
    let mut last_sub = unifier.unify(cs.clone(), &mut tenv);                  // unify
    unifier.errs.print_errs(&code_map);
    // println!("{:#?}", last_sub);

    // ------------ resolve module constraints until it stabilizes ----------
    // this is an inefficient workaround because the type inference algorithm
    // is not a full HM but with constraint collection and unification as seperate
    // stages.
    let mut last_ast = subs(&ast, &mut last_sub);;
    let resolve_modules = move || loop {
        // collect constraints
        let mut new_cs = Constraints::new(); new_cs.collect(&last_ast, &mut tenv);
        // unify constraints
        let mut new_unifier = Unifier::new(); let mut new_sub = new_unifier.unify(new_cs, &mut tenv);
        let temp_ast = subs(&last_ast, &mut new_sub);
        if temp_ast != last_ast {
            last_ast = temp_ast;
            continue;
        }
        return last_ast;
    };
    let final_ast = resolve_modules();
    println!("{:#?}", final_ast);
    // println!("{:#?}", tenv);
    // println!("{:#?}", new_cs);
}
