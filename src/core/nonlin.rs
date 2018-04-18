use core::{MethodName, Op};
use typed_ast::{Type, TypeEnv};

#[allow(non_camel_case_types)]
pub struct relu;
#[allow(non_camel_case_types)]
pub struct log_softmax;

impl Op for relu {
    fn get_name(&self,) -> String {
        "relu".to_owned()
    }

    fn get_module_sig(&self, tenv: &mut TypeEnv) -> Vec<(MethodName, Type)> {
        vec![("forward", Type::UnresolvedModuleFun("nonlin", "relu", "forward"))]
    }

    fn resolve(&self, tenv: &mut TypeEnv, _module: Option<Type>, fn_name: &str) -> Option<Type> {
        match fn_name {
            "self.forward" => relu::resolve_forward(tenv),
            _ => unimplemented!(),
        }
    }
}

impl relu {
    /// output same shape as input
    fn resolve_forward(tenv: &mut TypeEnv) -> Option<Type> {
        let ty = tenv.fresh_var();
        Some(fun!(ty.clone(), ty))
    }
}

impl Op for log_softmax {
    fn get_name(&self,) -> String {
        "log_softmax".to_owned()
    }

    fn get_module_sig(&self, tenv: &mut TypeEnv) -> Vec<(MethodName, Type)> {
        vec![("forward", Type::UnresolvedModuleFun("nonlin", "log_softmax", "forward"))]
    }

    fn resolve(&self, tenv: &mut TypeEnv, _module: Option<Type>, fn_name: &str) -> Option<Type> {
        match fn_name {
            "self.forward" => relu::resolve_forward(tenv),
            _ => unimplemented!(),
        }
    }
}

impl log_softmax {
    /// output same shape as input
    fn resolve_forward(tenv: &mut TypeEnv) -> Option<Type> {
        let ty = tenv.fresh_var();
        Some(fun!(ty.clone(), ty))
    }
}