use self::Type::*;
use core::{MethodName, Op};
use std::fmt::Write;
use span::CSpan;
use typing::typed_term::TyFnAppArg;
use typing::{Type, TypeEnv};
use errors::Diag;

#[allow(non_camel_case_types)]
#[derive(Debug, Clone)]
pub struct sigmoid;

impl Op for sigmoid {
    fn get_name(&self) -> &'static str {
        "sigmoid"
    }

    fn is_stateful(&self) -> bool { false }

    fn pytorch_name(&self) -> String {
        format!("F.{}", self.get_name())
    }

    fn ty_sigs(&self, _tenv: &mut TypeEnv) -> Vec<(MethodName, Type)> {
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
        _arg_ty: Type,
        _ret_ty: Type,
        _args: Vec<TyFnAppArg>,
        _inits: Option<Vec<TyFnAppArg>>,
    ) -> Option<Result<Type, Diag>> {
        match fn_name {
            "forward" => {
                let ty = tenv.fresh_var(CSpan::fresh_span());
                Some(Ok(fun!(self.get_name(), "forward", args!(arg!("x", ty.clone())), ty)))
            }
            _ => unimplemented!(),
        }
    }

    fn gen_fn_app(&self, name: &str, _args: &[TyFnAppArg]) -> Result<String, Diag> {
        let mut buf = String::new();
        match name {
            "forward" => {
                write!(buf, "").unwrap();
                Ok(buf)
            }
            _ => panic!("{} is not implemented", name),
        }
    }
}

#[allow(non_camel_case_types)]
#[derive(Debug, Clone)]
pub struct tanh;

impl Op for tanh {
    fn get_name(&self) -> &'static str {
         "tanh"
    }

    fn is_stateful(&self) -> bool { false }

    fn pytorch_name(&self) -> String {
        format!("F.{}", self.get_name())
    }

    fn ty_sigs(&self, _tenv: &mut TypeEnv) -> Vec<(MethodName, Type)> {
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
        _arg_ty: Type,
        _ret_ty: Type,
        _args: Vec<TyFnAppArg>,
        _inits: Option<Vec<TyFnAppArg>>,
    ) -> Option<Result<Type, Diag>> {
        match fn_name {
            "forward" => {
                let ty = tenv.fresh_var(CSpan::fresh_span());
                Some(Ok(fun!(self.get_name(), "forward", args!(arg!("x", ty.clone())), ty)))
            }
            _ => unimplemented!(),
        }
    }

    fn gen_fn_app(&self, name: &str, _args: &[TyFnAppArg]) -> Result<String, Diag> {
        let mut buf = String::new();
        match name {
            "forward" => {
                write!(buf, "").unwrap();
                Ok(buf)
            }
            _ => panic!("{} is not implemented", name),
        }
    }
}

#[allow(non_camel_case_types)]
#[derive(Debug, Clone)]
pub struct relu;

impl Op for relu {
    fn get_name(&self) -> &'static str {
        "relu"
    }

    fn is_stateful(&self) -> bool { false }

    fn pytorch_name(&self) -> String {
        format!("F.{}", self.get_name())
    }

    fn ty_sigs(&self, _tenv: &mut TypeEnv) -> Vec<(MethodName, Type)> {
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
        _arg_ty: Type,
        _ret_ty: Type,
        _args: Vec<TyFnAppArg>,
        _inits: Option<Vec<TyFnAppArg>>,
    ) -> Option<Result<Type, Diag>> {
        match fn_name {
            "forward" => {
                let ty = tenv.fresh_var(CSpan::fresh_span());
                Some(Ok(fun!(self.get_name(), "forward", args!(arg!("x", ty.clone())), ty)))
            }
            _ => unimplemented!(),
        }
    }

    fn gen_fn_app(&self, name: &str, _args: &[TyFnAppArg]) -> Result<String, Diag> {
        let mut buf = String::new();
        match name {
            "forward" => {
                write!(buf, "").unwrap();
                Ok(buf)
            }
            _ => panic!("{} is not implemented", name),
        }
    }
}

#[allow(non_camel_case_types)]
#[derive(Debug, Clone)]
pub struct leaky_relu;

impl Op for leaky_relu {
    fn get_name(&self) -> &'static str {
        "leaky_relu"
    }

    fn is_stateful(&self) -> bool { false }

    fn pytorch_name(&self) -> String {
        format!("F.{}", self.get_name())
    }

    fn ty_sigs(&self, _tenv: &mut TypeEnv) -> Vec<(MethodName, Type)> {
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
        _arg_ty: Type,
        _ret_ty: Type,
        _args: Vec<TyFnAppArg>,
        _inits: Option<Vec<TyFnAppArg>>,
    ) -> Option<Result<Type, Diag>> {
        match fn_name {
            "forward" => {
                let ty = tenv.fresh_var(CSpan::fresh_span());
                Some(Ok(fun!(self.get_name(), "forward", args!(arg!("x", ty.clone())), ty)))
            }
            _ => unimplemented!(),
        }
    }

    fn gen_fn_app(&self, name: &str, _args: &[TyFnAppArg]) -> Result<String, Diag> {
        let mut buf = String::new();
        match name {
            "forward" => {
                write!(buf, "").unwrap();
                Ok(buf)
            }
            _ => panic!("{} is not implemented", name),
        }
    }
}

#[allow(non_camel_case_types)]
#[derive(Debug, Clone)]
pub struct log_softmax;

impl Op for log_softmax {
    fn get_name(&self) -> &'static str {
        "log_softmax"
    }

    fn pytorch_name(&self) -> String {
        format!("F.{}", self.get_name())
    }

    fn is_stateful(&self) -> bool { false }

    fn ty_sigs(&self, _tenv: &mut TypeEnv) -> Vec<(MethodName, Type)> {
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
        _arg_ty: Type,
        _ret_ty: Type,
        _args: Vec<TyFnAppArg>,
        _inits: Option<Vec<TyFnAppArg>>,
    ) -> Option<Result<Type, Diag>> {
        match fn_name {
            "forward" => {
                let ty = tenv.fresh_var(CSpan::fresh_span());
                Some(Ok(fun!(self.get_name(), "forward", args!(arg!("x", ty.clone()), arg!("dim", int!())), ty)))
            }
            _ => unimplemented!(),
        }
    }

    fn gen_fn_app(&self, name: &str, args: &[TyFnAppArg]) -> Result<String, Diag> {
        let mut buf = String::new();
        match name {
            "forward" => {
                let args: Vec<_> = args.iter().map(|i| i.name.clone().unwrap()).collect();
                write!(buf, "{}", args.join(", ")).unwrap();
                Ok(buf)
            }
            _ => panic!("{} is not implemented", name),
        }
    }
}