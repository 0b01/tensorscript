use self::Type::*;
use core::{MethodName, Op};
use span::CSpan;
use typing::typed_term::TyFnAppArg;
use typing::{Type, TypeEnv};

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
        vec![
            (
                "forward",
                UnresolvedModuleFun("nonlin", self.get_name(), "forward", CSpan::fresh_span())
            )
        ]
    }

    fn resolve(
        &self,
        tenv: &mut TypeEnv,
        fn_name: &str,
        arg_ty: Type,
        ret_ty: Type,
        args: Vec<TyFnAppArg>,
        inits: Option<Vec<TyFnAppArg>>,
    ) -> Option<Type> {
        match fn_name {
            "forward" => {
                let ty = tenv.fresh_var(&CSpan::fresh_span());
                Some(fun!("sigmoid", "forward", args!(arg!("x", ty.clone())), ty))
            }
            _ => unimplemented!(),
        }
    }
}
impl Op for relu {
    fn get_name(&self) -> &'static str {
        "relu"
    }

    fn get_module_sig(&self, _tenv: &mut TypeEnv) -> Vec<(MethodName, Type)> {
        vec![
            (
                "forward",
                UnresolvedModuleFun("nonlin", self.get_name(), "forward", CSpan::fresh_span())
            )
        ]
    }

    fn resolve(&self,
        tenv: &mut TypeEnv,
        fn_name: &str,
        arg_ty: Type,
        ret_ty: Type,
        args: Vec<TyFnAppArg>,
        inits: Option<Vec<TyFnAppArg>>
    ) -> Option<Type> {
        match fn_name {
            "forward" => {
                let ty = tenv.fresh_var(&CSpan::fresh_span());
                Some(fun!("relu", "forward", args!(arg!("x", ty.clone())), ty))
            },
            _ => unimplemented!(),
        }
    }
}

impl Op for log_softmax {
    fn get_name(&self) -> &'static str {
        "log_softmax"
    }

    fn get_module_sig(&self, tenv: &mut TypeEnv) -> Vec<(MethodName, Type)> {
        vec![(
            "forward",
            UnresolvedModuleFun("nonlin", self.get_name(), "forward", CSpan::fresh_span())
        )]
    }

    fn resolve(
        &self,
        tenv: &mut TypeEnv,
        fn_name: &str,
        arg_ty: Type,
        ret_ty: Type,
        args: Vec<TyFnAppArg>,
        inits: Option<Vec<TyFnAppArg>>,
    ) -> Option<Type> {
        match fn_name {
            "forward" => {
                let ty = tenv.fresh_var(&CSpan::fresh_span());
                Some(fun!("log_softmax", "forward", args!(arg!("x", ty.clone()), arg!("dim", int!())), ty))
            }
            _ => unimplemented!(),
        }
    }
}
