use core::{MethodName, Op};
use typed_ast::{Type, TypeEnv};

pub struct Conv2d;
pub struct Dropout2d;

impl Op for Conv2d {
    fn get_name() -> String {
        "Conv2d".to_owned()
    }

    fn get_module_sig(tenv: &mut TypeEnv) -> Vec<(MethodName, Type)> {
        vec![("forward", Type::UnresolvedModuleFun)]
    }
}

impl Op for Dropout2d {
    fn get_name() -> String {
        "Dropout2d".to_owned()
    }

    fn get_module_sig(tenv: &mut TypeEnv) -> Vec<(MethodName, Type)> {
        vec![("forward", Type::UnresolvedModuleFun)]
    }
}

#[allow(non_camel_case_types)]
pub struct maxpool2d;

impl Op for maxpool2d {
    fn get_name() -> String {
        "maxpool2d".to_owned()
    }

    fn get_module_sig(tenv: &mut TypeEnv) -> Vec<(MethodName, Type)> {
        vec![("forward", Type::UnresolvedModuleFun)]
    }
}
