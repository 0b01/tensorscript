use core::{MethodName, Op};
use typed_ast::{Type, TypeEnv};

pub struct Conv2d;
pub struct Dropout2d;

impl Op for Conv2d {
    fn get_name(&self) -> String {
        "Conv2d".to_owned()
    }

    fn get_module_sig(&self, tenv: &mut TypeEnv) -> Vec<(MethodName, Type)> {
        vec![("forward", Type::UnresolvedModuleFun("conv", "Conv2d", "forward"))]
    }
}

impl Op for Dropout2d {
    fn get_name(&self,) -> String {
        "Dropout2d".to_owned()
    }

    fn get_module_sig(&self,tenv: &mut TypeEnv) -> Vec<(MethodName, Type)> {
        vec![("forward", Type::UnresolvedModuleFun("conv", "Dropout2d", "forward"))]
    }
}

#[allow(non_camel_case_types)]
pub struct maxpool2d;

impl Op for maxpool2d {
    fn get_name(&self,) -> String {
        "maxpool2d".to_owned()
    }

    fn get_module_sig(&self,tenv: &mut TypeEnv) -> Vec<(MethodName, Type)> {
        vec![("forward", Type::UnresolvedModuleFun("conv", "maxpool2d", "forward"))]
    }
}
