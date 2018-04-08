extern crate pest;
#[macro_use]
extern crate pest_derive;

mod ast;
mod target;
mod grammar;
mod builder;

const TEST_STR: &str = include_str!("../test.tss");

fn main() {
    let result = builder::parse_str(TEST_STR).unwrap();
    println!("{:?}", result);
    target::pytorch::gen(result);
}
