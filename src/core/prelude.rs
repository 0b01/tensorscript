use core::{MethodName, Op};
use errors::Diag;
use span::CSpan;
use typing::typed_term::TyFnAppArg;
use typing::{Type, TypeEnv};

#[allow(non_camel_case_types)]
#[derive(Debug, Clone)]
pub struct view;

impl Op for view {
    fn get_name(&self) -> &'static str {
        "view"
    }

    fn gen_import(&self) -> String {
        unimplemented!()
    }

    fn is_stateful(&self) -> bool { false }

    fn get_module_sig(&self, _tenv: &mut TypeEnv) -> Vec<(MethodName, Type)> {
        use self::Type::*;
        vec![
            (
                "forward",
                UnresolvedModuleFun("prelude", self.get_name(), "forward", CSpan::fresh_span())
            )
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
        _inits: Option<Vec<TyFnAppArg>>, // ... refactor into span error
    ) -> Option<Result<Type, Diag>> {
        match fn_name {
            "forward" => {
                // println!("ret_ty: {:#?}\n, arg_ty: {:#?}", ret_ty, arg_ty);
                if !arg_ty.is_resolved() { return None; }
                let args_map = arg_ty.as_args_map()?;
                let arg_tsr = args_map.get("x")?.as_vec()?;
                let ret_tsr = ret_ty.as_vec()?;

                let resolved_arg_tsr: Vec<i64> = arg_tsr.iter().filter_map(|i| i.as_num()).collect();
                let resolved_ret_tsr: Vec<i64> = ret_tsr.iter().filter_map(|i| i.as_num()).collect();

                // suppose arg_ty = [!1, 10]
                //         ret_ty = ['100, 2, 5]
                // replace '100 with !1
                if ret_tsr.len() - resolved_ret_tsr.len() > 1 {
                    return Some(Err(
                        Diag::EllisionError("Cannot elide more than 1 tensor dimension in view function".to_owned(), ret_tsr[0].span())
                    ));
                }

                let ret_prod: i64 = resolved_ret_tsr.iter().product();
                let arg_prod: i64 = resolved_arg_tsr.iter().product();

                let is_only_one_arg_dim_unresolved = (arg_tsr.len() - resolved_arg_tsr.len()) == 1;
                if ret_prod == arg_prod && is_only_one_arg_dim_unresolved {
                    let unresolved_arg_dim = arg_tsr.iter().find(|i| i.as_num().is_none()).unwrap();
                    let unresolved_ret_dim = ret_tsr.iter().find(|i| i.as_num().is_none()).unwrap();
                    let modified_ret_ty = ret_tsr
                        .iter()
                        .map(|i|
                            if i == unresolved_ret_dim {
                                unresolved_arg_dim
                            } else {
                                i
                            } )
                        .cloned()
                        .collect();
                    Some(Ok(
                        fun!("view", "forward", arg_ty, tsr!(modified_ret_ty))
                    ))
                } else {
                    panic!("{} {}", ret_prod, arg_prod);// ...
                }
            }
            _ => unimplemented!(),
        }
    }
}
