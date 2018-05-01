#![feature(const_fn)]
#![feature(iterator_flatten)]
#![feature(transpose_result)]
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
/// 4. Hindley-Milner type inference for type reconstruction. This is consisted
/// of a few substeps.
///
///   a. Collect constraints. (handled in constraint.rs)
///       In this step, traverse typed ast and collect types of adjacent nodes that should
///       be equivalent. This generates a Constraint struct which is just a thin wrapper
///       around a btreeset of (Type, Type) tuple.
///
///   b. Unify constraints by generating substitutions.
///       This is a variant to Algorithm W in H-M type inference. Bascially, unify_one
///       function tries to replace 1 type var with a concrete type. The parent function, unify,
///       then uses that substitution on the rest of the constraints, thus eliminating the type
///       variable from the constraint set. The process is iterated until one of these conditions are met:
///       a) all type variable are exhausted. b) equivalence that can never happen. c) circular
///       type dependence (handled by occurs check).
///
///   c. Generate Substitutions
///       Now after the unification is complete, the function returns a list of substitutions that
///       should remove all type variables from the typed AST.
///
/// 5. code gen // ... todo
///
///
/// A note about `Span`s: Span contains the location in the source code for
/// error reporting. Think of it as a lightweight tag that can be associated with
/// data structures such as AST nodes, types, etc...
///

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
mod parsing;
mod span;
mod errors;
mod codegen;


use typing::constraint::Constraints;
use typing::unifier::Unifier;
use typing::annotate::Annotator;
use codegen::generator::Generator;
use typing::type_env::TypeEnv;
use typing::inferred_ast::subs;
use errors::{Emitter};
use parsing::ast_builder::ASTBuilder;
use span::CSpan;

use std::rc::Rc;
use std::cell::RefCell;
use std::fs::File;
use std::io::Read;
use std::process::exit;

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
            .help("Sets a custom input file")
            .takes_value(true)
            .required(true))
        .arg(Arg::with_name("print_ast")
            .long("print-ast")
            .help("Prints AST"))
        .get_matches()
}

fn main() {
    // --------------- get command line options -----------------
    let matches = get_matches();
    let fname = matches.value_of("input").unwrap();
    let mut file = File::open(fname).expect("Unable to open the file");
    let mut src = String::new();
    file.read_to_string(&mut src).expect("Unable to read the file");
    // -------------------- create emitter --------------------
    let mut code_map = CodeMap::new();
    let file_map = code_map.add_filemap(fname.to_owned().into(), src.clone());
    let emitter = Rc::new(RefCell::new(Emitter::new(code_map)));
    // --------------- parse into untyped ast   -----------------
    let cspan = CSpan::new(file_map.span());
    let builder = ASTBuilder::new(Rc::clone(&emitter), cspan);
    let parsed_terms = builder.parse_str(&src);
    let program = parsed_terms
        .unwrap_or_else(||{ emitter.borrow().print_errs(); exit(-1); });
    // ------------- annotate ast with type vars --------------
    let tenv = Rc::new(RefCell::new(TypeEnv::new()));
    let annotator = Annotator::new(Rc::clone(&emitter), Rc::clone(&tenv));
    let ast = annotator.annotate(&program);
    emitter.borrow().print_errs();
    // println!("{:#?}", ast);
    // println!("initial tenv: {:#?}", tenv);
    // ------------ first unitfication pass ---------------
    let mut cs = Constraints::new(Rc::clone(&emitter), Rc::clone(&tenv));
    cs.collect(&ast);
    let mut unifier = Unifier::new(Rc::clone(&emitter), Rc::clone(&tenv));
    let mut last_sub = unifier.unify(cs.clone());
    emitter.borrow().print_errs();
    // println!("{:#?}", last_sub);

    // ------------ resolve module constraints until it stabilizes ----------
    let mut last_ast = subs(&ast, &mut last_sub);;
    let em_clone = emitter.clone();
    let tenv_clone = tenv.clone();
    let resolve_ast = move || {
        loop {
            // collect constraints
            let mut new_cs = Constraints::new(Rc::clone(&em_clone), Rc::clone(&tenv_clone));
            new_cs.collect(&last_ast);
            em_clone.borrow().print_errs();
            // unify constraints
            let mut new_unifier = Unifier::new(Rc::clone(&em_clone), Rc::clone(&tenv_clone));
            let mut new_sub = new_unifier.unify(new_cs.clone());
            em_clone.borrow().print_errs();
            let temp_ast = subs(&last_ast, &mut new_sub);
            if temp_ast != last_ast {
                last_ast = temp_ast;
                continue;
            }
            return last_ast;
        }
    };
    let final_ast = resolve_ast();
    if matches.is_present("print_ast") {
        println!("{:#?}", final_ast);
        exit(0);
    }
    // ---------------------------- code gen -----------------------------------
    let mut generator = Generator::new(emitter.clone(), tenv.clone());
    generator.generate(&final_ast).unwrap();
    println!("{}", generator.buf);
}
