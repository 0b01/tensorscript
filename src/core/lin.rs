use core::{MethodName, Op};
use span::CSpan;
use typed_ast::typed_term::{ArgsVecInto, Ty, TyFnAppArg, TyTerm};
use typed_ast::{Type, TypeEnv};

// #[allow(non_camel_case_types)]
pub struct Linear;

impl Op for Linear {
    fn get_name(&self) -> &'static str {
        "Linear"
    }

    fn get_module_sig(&self, tenv: &mut TypeEnv) -> Vec<(MethodName, Type)> {
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
        tenv: &mut TypeEnv,
        fn_name: &str,
        arg_ty: Type,
        ret_ty: Type,
        inits: Option<Vec<TyFnAppArg>>, // ... refactor into span error
    ) -> Option<Type> {
        match fn_name {
            "forward" => {
                if inits.is_some() {
                    let hm = inits.unwrap().to_btreemap().unwrap();
                    if !hm.contains_key("in") {
                        panic!("Initatialize Linear with parameter in=");
                    } else if !hm.contains_key("out") {
                        panic!("Initatialize Linear with parameter out=");
                    }

                    let arg_dim = arg_ty
                        .first_arg_ty()?
                        .last_dim()?
                        .as_num();
                    let ret_dim = ret_ty
                        .last_dim()?
                        .as_num();

                    let in_dim = hm.get("in").and_then(unwrap_dim)?;
                    let out_dim = hm.get("out").and_then(unwrap_dim)?;

                    assert!((arg_dim == in_dim) && (ret_dim == out_dim));

                    None
                } else {
                    None
                }
            }
            _ => unimplemented!(),
        }
    }
}

fn unwrap_dim(in_dim: &Box<TyTerm>) -> Option<i64> {
    match in_dim.ty() {
        Type::INT(_) => in_dim.int(),
        Type::ResolvedDim(num, _) => Some(num),
        _ => panic!("not a numeric value!"),
    }
}
