extern crate pest;
#[macro_use]
extern crate pest_derive;

mod ast;
mod target;
mod grammar;
mod builder;
mod scope;

use scope::ScopeStack;

const TEST_STR: &str = include_str!("../test.tss");

fn main() {
    let program = builder::parse_str(TEST_STR).unwrap();
    println!("{:?}", program);
    let mut global_scope = ScopeStack::new();
    target::pytorch::gen(program);
}
