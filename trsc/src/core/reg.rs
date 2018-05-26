use core::{MethodName, Op, PyTorch, Resolve};
use errors::Diag;
use span::CSpan;
use typing::typed_term::TyFnAppArg;
use typing::{Type, TypeEnv};
use typing::typed_term::ArgsVecInto;
use std::fmt::Write;

#[derive(Debug, Op)]
#[path = "reg"]
#[new = "(p: float) -> self"]
#[forward = "?(x: tsr0) -> tsr0"]
pub struct Dropout2d;

impl Resolve for Dropout2d {
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

impl PyTorch for Dropout2d {

    fn pytorch_name(&self) -> &'static str {
        "nn.Dropout2d"
    }

    fn gen_fn_app(&self, name: &str, args: &[TyFnAppArg]) -> Result<String, Diag> {
        let mut buf = String::new();
        match name {
            "new" => {
                write!(buf, "{}(", self.get_name()).unwrap();
                let map = args.to_btreemap().unwrap();
                write!(buf, "p={})", map["p"].as_str().unwrap()).unwrap();
                Ok(buf)
            }
            "forward" => {
                let args: Vec<_> = args.iter().map(|i| i.name.clone().unwrap()).collect();
                write!(buf, "{}", args.join(", ")).unwrap();
                Ok(buf)
            }
            _ => panic!("{} is not implemented", name),
        }
    }
}

#[derive(Debug, Op)]
#[path = "reg"]
#[new = "(num_features: int) -> self"]
#[forward = "?(x: tsr0) -> tsr0"]
#[stateful]
pub struct BatchNorm1d;

impl Resolve for BatchNorm1d {
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

impl PyTorch for BatchNorm1d {
    fn pytorch_name(&self) -> &'static str {
        "nn.BatchNorm1d"
    }
    fn gen_fn_app(&self, name: &str, args: &[TyFnAppArg]) -> Result<String, Diag> {
        let mut buf = String::new();
        match name {
            "new" => {
                let map = args.to_btreemap().unwrap();
                write!(buf, "{}(", self.pytorch_name()).unwrap();
                write!(buf, "num_features={})",
                    map["num_features"].as_num().unwrap()).unwrap();
            }
            "forward" => {
                // let map = args.to_btreemap().unwrap();
                write!(buf, "x").unwrap();
            }
            _ => unimplemented!(),
        }

        Ok(buf)
    }
}
