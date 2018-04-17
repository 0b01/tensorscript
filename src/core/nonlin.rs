use core::{MethodName, Op};
use typed_ast::Type;

#[allow(non_camel_case_types)]
pub struct relu;
#[allow(non_camel_case_types)]
pub struct log_softmax;

impl Op for relu {
    fn get_name() -> String {
        "relu".to_owned()
    }

    fn get_module_sig() -> Vec<(MethodName, Type)> {
        vec![
        ]
    }
}

impl Op for log_softmax {
    fn get_name() -> String {
        "log_softmax".to_owned()
    }

    fn get_module_sig() -> Vec<(MethodName, Type)> {
        vec![
        ]
    }
}