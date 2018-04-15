use std::collections::HashSet;

use typed_ast::typed_term::TyTerm;
use typed_ast::type_env::TypeEnv;
use typed_ast::Type;

#[derive(Debug, Hash, Eq, PartialEq)]
pub struct Equal(Type, Type);

#[derive(Debug)]
pub struct Constraints {
    constraints: HashSet<Equal>,
}

impl Constraints {
    fn new() -> Self {
        Constraints {
            constraints: HashSet::new(),
        }
    }
}

pub fn collect(typed_ast: &TyTerm, tenv: &TypeEnv) -> Constraints {
    Constraints::new()
}