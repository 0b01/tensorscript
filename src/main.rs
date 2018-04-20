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
/// 3. [ ] BUG: dimension mismatch
/// 4. [ ] BUG: non-determinism
/// 5. [ ] BUG: impl Hash for Type
/// 6. [ ] set up examples and tests
/// 7. [*] set up commandline
/// 8. [ ] more examples

extern crate pest;
#[macro_use]
extern crate pest_derive;
#[macro_use]
extern crate maplit;

extern crate codespan;
extern crate clap;
extern crate codespan_reporting;

#[macro_use]
mod typed_ast;
mod core;
mod parser;
mod span;
mod type_reconstruction;

use std::fs::File;
use std::io::Read;

use type_reconstruction::constraint::Constraints;
use type_reconstruction::unifier::Unifier;
use typed_ast::annotate::annotate;
use typed_ast::type_env::TypeEnv;

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
    let matches = get_matches();
    let fname = matches.value_of("input").unwrap();

    let mut file = File::open(fname).expect("Unable to open the file");
    let mut src = String::new();
    file.read_to_string(&mut src).expect("Unable to read the file");


    let mut code_map = CodeMap::new();
    let file_map = code_map.add_filemap("test".into(), src.clone());
    let cspan = span::CSpan::new(file_map.span());

    let program = parser::parse_str(&src, &cspan).unwrap();
    // println!("{:#?}", program);

    let mut tenv = TypeEnv::new();
    let ast = annotate(&program, &mut tenv);
    // println!("{}", ast);
    // println!("initial tenv: {:#?}", tenv);

    let mut cs = Constraints::new();
    cs.collect(&ast, &mut tenv);
    // println!("{:#?}", cs);

    let mut unifier = Unifier::new();
    let mut subs = unifier.unify(cs.clone(), &mut tenv);
    unifier.print_errs(&code_map);
    // println!("{:#?}", subs);
    // println!("{:#?}", subs.apply(&cs));
    let test = type_reconstruction::inferred_ast::subs(&ast, &mut subs);
    // println!("{:#?}", test);
    // println!("{:#?}", tenv);
}

// 1. initialize global scope
// 2. insert use::symbols, types into scope
// 3. insert node_decl symbols, types into scope
// 4. resolve weights and graph decls which should have the same type
// 5. create local scope
// 6. insert macros to scope macros
// 7. typecheck weights body against global_scope
// 8. for each functions in graph_decl, create scope and typecheck.

// a note on typechecking
// 1. for each generic type in the node type decl, create in scope
// 2. for each symbol on the RHS of fn_ty_sig, check macro, global types and LHS types
// 3. for user defined functions, check macro, global

// a note on generics
// 1. generics are constraints on tensor dims
// 2. f(x) does not typecheck given f::<?,10 -> ?,1>, x::<4,8>
// 3. f<?->n>(x<1>) does not typecheck
// algorithm
// 1. fix function type
// 2. for each arg, check rank against corresponding param rank
// 3. for each dimension, deref generic types, check referential equality
// 4. x::<1,1> + y::<1,1> is <1,1>
