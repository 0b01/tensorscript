#[macro_use]
extern crate pest;
#[macro_use]
extern crate pest_derive;

mod parser;
mod typed_ast;
use typed_ast::type_env::TypeEnv;
use typed_ast::annotate::annotate;

const TEST_STR: &str = include_str!("../test.tss");

fn main() {
    let program = parser::parse_str(TEST_STR).unwrap();
    // println!("{:#?}", program);

    let mut tenv = TypeEnv::new();
    let ast = annotate(&program, &mut tenv);
    println!("{}", ast);
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
