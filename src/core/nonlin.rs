use core::{MethodName, Op};
use typed_ast::{Type, TypeEnv};

#[allow(non_camel_case_types)]
pub struct relu;
#[allow(non_camel_case_types)]
pub struct log_softmax;

impl Op for relu {
    fn get_name() -> String {
        "relu".to_owned()
    }

    fn get_module_sig(tenv: &mut TypeEnv) -> Vec<(MethodName, Type)> {
        vec![("forward", Type::UnresolvedModuleFun)]
    }
}

impl Op for log_softmax {
    fn get_name() -> String {
        "log_softmax".to_owned()
    }

    fn get_module_sig(tenv: &mut TypeEnv) -> Vec<(MethodName, Type)> {
        vec![("forward", Type::UnresolvedModuleFun)]
    }
}
