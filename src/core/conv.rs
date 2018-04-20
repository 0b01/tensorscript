use core::{MethodName, Op};
use typed_ast::{Type, TypeEnv};
use typed_ast::typed_term::{TyFnAppArg, ArgsVecInto, Ty, TyTerm};
use span::CSpan;
use self::Type::*;

pub struct Conv2d;
pub struct Dropout2d;

impl Op for Conv2d {
    fn get_name(&self) -> &'static str {
        "Conv2d"
    }

    fn get_module_sig(&self, tenv: &mut TypeEnv) -> Vec<(MethodName, Type)> {
        vec![
            ("new", fun!(args!(
                arg!("in_ch", int!()),
                arg!("out_ch", int!()),
                arg!("kernel_size", int!())
                ), module!(self.get_name()))),
            ("forward", Type::UnresolvedModuleFun("conv", self.get_name(), "forward", CSpan::fresh_span()))
        ]
    }

    fn resolve(&self, tenv: &mut TypeEnv, module: Option<Type>, _fn_name: &str, inits: Option<Vec<TyFnAppArg>>) -> Option<Type> {
        println!("TODO!");
        None
    }
}

impl Op for Dropout2d {
    fn get_name(&self,) -> &'static str {
        "Dropout2d"
    }

    fn get_module_sig(&self, tenv: &mut TypeEnv) -> Vec<(MethodName, Type)> {
        let span = CSpan::fresh_span();
        let ty = tenv.fresh_var(&span);
        vec![
            ("new", fun!(args!(arg!("p", float!())), module!(self.get_name()))),
            ("forward", fun!(ty.clone(), ty))
        ]
    }
}

#[allow(non_camel_case_types)]
pub struct maxpool2d;

impl Op for maxpool2d {
    fn get_name(&self,) -> &'static str {
        "maxpool2d"
    }

    fn get_module_sig(&self, tenv: &mut TypeEnv) -> Vec<(MethodName, Type)> {
        vec![
            ("forward", Type::UnresolvedModuleFun("conv", self.get_name(), "forward", CSpan::fresh_span())),
        ]
    }

    fn resolve(&self, tenv: &mut TypeEnv, module: Option<Type>, _fn_name: &str, inits: Option<Vec<TyFnAppArg>>) -> Option<Type> {
        println!("TODO!");
        None
    }
}
