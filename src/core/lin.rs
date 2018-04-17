use core::{MethodName, Op};
use typed_ast::Type;

// #[allow(non_camel_case_types)]
pub struct Linear;

impl Op for Linear {
    fn get_name() -> String {
        "Linear".to_owned()
    }

    fn get_module_sig() -> Vec<(MethodName, Type)> {
        use self::Type::*;
        vec![
            ("init_normal", FUN(box FN_ARGS(vec![
                FN_ARG(Some("std".to_owned()), box FLOAT)
            ]), box Unit)),
        ]
    }
}
