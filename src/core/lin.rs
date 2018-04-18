#[macro_use]
use typed_ast::types;

use core::{MethodName, Op};
use typed_ast::{Type, TypeEnv};

// #[allow(non_camel_case_types)]
pub struct Linear;

impl Op for Linear {
    fn get_name(&self) -> String {
        "Linear".to_owned()
    }

    fn get_module_sig(&self, tenv: &mut TypeEnv) -> Vec<(MethodName, Type)> {
        use self::Type::*;
        vec![
            ("init_normal", fun!(args!(arg!("std", FLOAT)), Unit)),
            ("new", fun!(args!(arg!("in", INT)), module!("Linear"))),
            ("forward", UnresolvedModuleFun("lin", "Linear", "forward")),
        ]
    }

    /// output same shape as input
    fn resolve(&self, tenv: &mut TypeEnv, _module: Option<Type>, _fn_name: &str) -> Option<Type> {
        unimplemented!()
    }
}
