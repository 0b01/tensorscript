use self::Type::*;
use core::{MethodName, Op};
use span::CSpan;
use typed_ast::typed_term::TyFnAppArg;
use typed_ast::{Type, TypeEnv};

#[allow(non_camel_case_types)]
pub struct relu;
#[allow(non_camel_case_types)]
pub struct log_softmax;
#[allow(non_camel_case_types)]
pub struct sigmoid;

impl Op for sigmoid {
    fn get_name(&self) -> &'static str {
        "sigmoid"
    }

    fn get_module_sig(&self, tenv: &mut TypeEnv) -> Vec<(MethodName, Type)> {
        let ty = tenv.fresh_var(&CSpan::fresh_span());
        vec![
            (
                "forward",
                fun!("sigmoid", "forward", args!(arg!("x", ty.clone())), ty)
            )
        ]
    }

    fn resolve(
        &self,
        tenv: &mut TypeEnv,
        _module: Option<Type>,
        fn_name: &str,
        inits: Option<Vec<TyFnAppArg>>,
    ) -> Option<Type> {
        match fn_name {
            _ => unimplemented!(),
        }
    }
}
impl Op for relu {
    fn get_name(&self) -> &'static str {
        "relu"
    }

    fn get_module_sig(&self, tenv: &mut TypeEnv) -> Vec<(MethodName, Type)> {
        let ty = tenv.fresh_var(&CSpan::fresh_span());
        vec![("forward", fun!("relu", "forward", args!(arg!("x", ty.clone())), ty))]
    }

    fn resolve(
        &self,
        tenv: &mut TypeEnv,
        _module: Option<Type>,
        fn_name: &str,
        inits: Option<Vec<TyFnAppArg>>,
    ) -> Option<Type> {
        match fn_name {
            _ => unimplemented!(),
        }
    }
}

impl Op for log_softmax {
    fn get_name(&self) -> &'static str {
        "log_softmax"
    }

    fn get_module_sig(&self, tenv: &mut TypeEnv) -> Vec<(MethodName, Type)> {
        let ty = tenv.fresh_var(&CSpan::fresh_span());
        vec![(
            "forward",
            fun!("log_softmax", "forward", args!(arg!("x", ty.clone()), arg!("dim", int!())), ty),
        )]
    }

    fn resolve(
        &self,
        tenv: &mut TypeEnv,
        _module: Option<Type>,
        fn_name: &str,
        inits: Option<Vec<TyFnAppArg>>,
    ) -> Option<Type> {
        match fn_name {
            _ => unimplemented!(),
        }
    }
}
