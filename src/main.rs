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
/// 1. implement module pattern matching
/// 2. type level computation (resolved tensor dimension)
///
extern crate pest;
#[macro_use]
extern crate pest_derive;
#[macro_use]
extern crate maplit;

extern crate codespan;
extern crate codespan_reporting;


#[macro_use]
mod typed_ast;
mod core;
mod parser;
mod type_reconstruction;
mod span;


use type_reconstruction::constraint::Constraints;
use type_reconstruction::subst::unify;
use typed_ast::annotate::annotate;
use typed_ast::type_env::TypeEnv;

const TEST_STR: &str = include_str!("../test.trs");

fn main() {
    let program = parser::parse_str(TEST_STR).unwrap();
    // println!("{:#?}", program);

    let mut tenv = TypeEnv::new();
    let ast = annotate(&program, &mut tenv);
    // println!("{}", ast);
    // println!("initial tenv: {:#?}", tenv);

    let mut cs = Constraints::new();
    cs.collect(&ast, &mut tenv);
    // println!("{:#?}", cs);

    let mut subs = unify(cs.clone(), &mut tenv);
    // println!("{:#?}", subs);
    // println!("{:#?}", subs.apply(&cs));
    let test = type_reconstruction::inferred_ast::subs(&ast, &mut subs);
    println!("{:#?}", test);
    println!("{:#?}", tenv);
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
