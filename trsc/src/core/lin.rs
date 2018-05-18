use core::{MethodName, Op};
use errors::Diag;
use span::CSpan;
use typing::typed_term::{ArgsVecInto, TyFnAppArg, TyTerm};
use typing::{Type, TypeEnv};
use std::fmt::Write;

// #[allow(non_camel_case_types)]
#[derive(Debug, Clone)]
pub struct Linear;

impl Op for Linear {
    fn get_name(&self) -> &'static str {
        "Linear"
    }

    fn is_stateful(&self) -> bool { true }

    fn ty_sigs(&self, _tenv: &mut TypeEnv) -> Vec<(MethodName, Type)> {
        use self::Type::*;
        vec![
            (
                "init_normal",
                fun!("Linear", "init_normal", args!(arg!("std", float!())), unit!())
            ),
            (
                "new",
                fun!(
                    "Linear", "new",
                    args!(arg!("in", int!()), arg!("out", int!())),
                    module!(self.get_name())
                ),
            ),
            (
                "forward",
                UnresolvedModuleFun("lin", self.get_name(), "forward", CSpan::fresh_span()),
            ),
        ]
    }

    /// output same shape as input
    fn resolve(
        &self,
        _tenv: &mut TypeEnv,
        fn_name: &str,
        arg_ty: Type,
        ret_ty: Type,
        _args: Vec<TyFnAppArg>,
        inits: Option<Vec<TyFnAppArg>>, // ... refactor into span error
    ) -> Option<Result<Type, Diag>> {
        match fn_name {
            "forward" => {
                if inits.is_some() {
                    let hm = inits.unwrap().to_btreemap().unwrap();
                    if !hm.contains_key("in") {
                        panic!("Initatialize Linear with parameter in=");
                    } else if !hm.contains_key("out") {
                        panic!("Initatialize Linear with parameter out=");
                    }

                    let in_dim = hm.get("in").and_then(|t| unwrap_dim(t))?;
                    let out_dim = hm.get("out").and_then(|t| unwrap_dim(t))?;

                    let span = arg_ty.span();

                    let (a, b) = match (arg_ty.first_arg_ty()?.as_vec(), ret_ty.as_vec()) {
                        (None, None) => return None,
                        (Some(ref mut a), None) |
                        (None, Some(ref mut a)) => {
                            // modify the last dimension
                            let mut b = a.clone();
                            {
                                let mut last_arg_dim = a.last_mut().unwrap();
                                let mut last_ret_dim = b.last_mut().unwrap();
                                *last_arg_dim = Type::ResolvedDim(in_dim, CSpan::fresh_span());
                                *last_ret_dim = Type::ResolvedDim(out_dim, CSpan::fresh_span());
                            };

                            (a.clone(), b)
                        }
                        (Some(ref mut a), Some(ref mut b)) => {
                            if a.len() != b.len() {
                                // return dimension mismatch
                                return Some(Err(Diag::TypeError(arg_ty, ret_ty)));
                            }
                            // modify the last dimension
                            {
                                let mut last_arg_dim = a.last_mut().unwrap();
                                let mut last_ret_dim = b.last_mut().unwrap();
                                *last_arg_dim = Type::ResolvedDim(in_dim, CSpan::fresh_span());
                                *last_ret_dim = Type::ResolvedDim(out_dim, CSpan::fresh_span());
                            };

                            (a.clone(), b.clone())
                        }
                    };

                    Some(Ok(fun!(
                        self.get_name(),
                        "forward",
                        args!(arg!("x",Type::TSR(a, span))),
                        Type::TSR(b, span)
                    )))
                } else {
                    None
                }
            }
            _ => unimplemented!(),
        }
    }

    fn gen_fn_app(&self, name: &str, args: &[TyFnAppArg]) -> Result<String, Diag> {
        let mut buf = String::new();
        match name {
            "new" => {
                let map = args.to_btreemap().unwrap();
                write!(buf, "{}(", self.pytorch_name()).unwrap();
                write!(buf, "in_features={:?}, ", map["in"].as_num().unwrap()).unwrap();
                write!(buf, "out_features={:?})", map["out"].as_num().unwrap()).unwrap();
                Ok(buf)
            }
            "forward" => {
                let args: Vec<_> = args.iter().map(|i| i.name.clone().unwrap()).collect();
                write!(buf, "{}", args.join(", ")).unwrap();
                Ok(buf)
            }
            "init_normal" => {
                let map = args.to_btreemap().unwrap();
                write!(buf, "nn.init.normal_(").unwrap();
                write!(buf, "std={}", map["std"].as_float().unwrap()).unwrap();
                write!(buf, ")").unwrap();
                Ok(buf)
            }
            _ => panic!("{} is not implemented", name),
        }
    }
}

fn unwrap_dim(in_dim: &TyTerm) -> Option<i64> {
    match in_dim.ty() {
        Type::INT(_) => in_dim.as_num(),
        Type::ResolvedDim(num, _) => Some(num),
        _ => panic!("{:?} is not a numeric value!", in_dim),
    }
}
