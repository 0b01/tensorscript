use core::{MethodName, Op};
use errors::Diag;
use span::CSpan;
use typing::typed_term::{ArgsVecInto, Ty, TyFnAppArg, TyTerm};
use typing::{Type, TypeEnv};

// #[allow(non_camel_case_types)]
pub struct Linear;

impl Op for Linear {
    fn get_name(&self) -> &'static str {
        "Linear"
    }

    fn get_module_sig(&self, _tenv: &mut TypeEnv) -> Vec<(MethodName, Type)> {
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
                    // let arg_dim = arg_ty
                    //     .first_arg_ty()?
                    //     .last_dim()?
                    //     .as_num().unwrap();
                    // let ret_dim = ret_ty
                    //     .last_dim()?
                    //     .as_num().unwrap();

                    let in_dim = hm.get("in").and_then(|t| unwrap_dim(t))?;
                    let out_dim = hm.get("out").and_then(|t| unwrap_dim(t))?;

                    let span = arg_ty.span();

                    let mut arg_dim = arg_ty.first_arg_ty()?.as_vec()?;
                    let mut ret_dim = ret_ty.as_vec()?;
                    if arg_dim.len() != ret_dim.len() {
                        // return dimension mismatch
                        return Some(Err(Diag::TypeError(arg_ty, ret_ty)));
                    }
                    // modify the last dimension
                    {
                        let mut last_arg_dim = arg_dim.last_mut().unwrap();
                        let mut last_ret_dim = ret_dim.last_mut().unwrap();
                        *last_arg_dim = Type::ResolvedDim(in_dim, CSpan::fresh_span());
                        *last_ret_dim = Type::ResolvedDim(out_dim, CSpan::fresh_span());
                    };

                    Some(Ok(
                        fun!(
                            self.get_name(),
                            "forward",
                            args!(arg!("x",Type::TSR(arg_dim, span))),
                            Type::TSR(ret_dim, span)
                        )
                    ))
                } else {
                    None
                }
            }
            _ => unimplemented!(),
        }
    }
}

fn unwrap_dim(in_dim: &TyTerm) -> Option<i64> {
    match in_dim.ty() {
        Type::INT(_) => in_dim.int(),
        Type::ResolvedDim(num, _) => Some(num),
        _ => panic!("not a numeric value!"),
    }
}
