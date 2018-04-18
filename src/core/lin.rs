use core::{MethodName, Op};
use typed_ast::{Type, TypeEnv};

// #[allow(non_camel_case_types)]
pub struct Linear;

impl Op for Linear {
    fn get_name() -> String {
        "Linear".to_owned()
    }

    fn get_module_sig(tenv: &mut TypeEnv) -> Vec<(MethodName, Type)> {
        use self::Type::*;
        vec![
            (
                "init_normal",
                FUN(
                    box FnArgs(vec![FnArg(Some("std".to_owned()), box FLOAT)]),
                    box Unit,
                ),
            ),
            ("forward", UnresolvedModuleFun),
        ]
    }
}
