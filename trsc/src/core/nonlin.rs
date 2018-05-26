use self::Type::*;
use core::{MethodName, Op, PyTorch, Resolve};
use std::fmt::Write;
use span::CSpan;
use typing::typed_term::TyFnAppArg;
use typing::{Type, TypeEnv};
use errors::Diag;

#[allow(non_camel_case_types)]
#[derive(Debug, Op)]
#[path = "nonlin"]
#[forward = "?()"]
pub struct sigmoid;

impl Resolve for sigmoid {
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
}

impl PyTorch for sigmoid {
    fn pytorch_name(&self) -> &'static str {
        "F.sigmoid"
    }
    fn gen_fn_app(&self, name: &str, _args: &[TyFnAppArg]) -> Result<String, Diag> {
        let buf = String::new();
        match name {
            "forward" => {
                // write!(buf, "").unwrap();
                Ok(buf)
            }
            _ => panic!("{} is not implemented", name),
        }
    }
}

#[allow(non_camel_case_types)]
#[derive(Debug, Op)]
#[path = "nonlin"]
#[forward = "?()"]
pub struct tanh;

impl Resolve for tanh {
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
}

impl PyTorch for tanh {
    fn pytorch_name(&self) -> &'static str {
        "F.tanh"
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
#[derive(Debug, Op)]
#[path = "nonlin"]
#[forward = "?()"]
pub struct relu;

impl Resolve for relu {
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
}

impl PyTorch for relu {
    fn pytorch_name(&self) -> &'static str {
        "F.relu"
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
#[derive(Debug, Op)]
#[path = "nonlin"]
#[forward = "?()"]
pub struct leaky_relu;

impl Resolve for leaky_relu {
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
}

impl PyTorch for leaky_relu {
    fn pytorch_name(&self) -> &'static str {
        "F.leaky_relu"
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
#[derive(Debug, Op)]
#[path = "nonlin"]
#[forward = "?()"]
pub struct log_softmax;

impl Resolve for log_softmax {
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
}

impl PyTorch for log_softmax {
    fn pytorch_name(&self) -> &'static str {
        "F.log_softmax"
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
