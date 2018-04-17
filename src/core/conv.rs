use core::{MethodName, Op};
use typed_ast::Type;

pub struct Conv2d;
pub struct Dropout2d;

impl Op for Conv2d {
    fn get_name() -> String {
        "Conv2d".to_owned()
    }

    fn get_module_sig() -> Vec<(MethodName, Type)> {
        vec![
        ]
    }
}

impl Op for Dropout2d {
    fn get_name() -> String {
        "Dropout2d".to_owned()
    }

    fn get_module_sig() -> Vec<(MethodName, Type)> {
        vec![
        ]
    }
}

#[allow(non_camel_case_types)]
pub struct maxpool2d;

impl Op for maxpool2d {
    fn get_name() -> String {
        "maxpool2d".to_owned()
    }

    fn get_module_sig() -> Vec<(MethodName, Type)> {
        vec![
        ]
    }
}
