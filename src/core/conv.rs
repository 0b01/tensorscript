use self::Type::*;
use core::{MethodName, Op};
use span::CSpan;
use typed_ast::typed_term::{ArgsVecInto, Ty, TyFnAppArg, TyTerm};
use typed_ast::{Type, TypeEnv};

pub struct Conv2d;
pub struct Dropout2d;

impl Op for Conv2d {
    fn get_name(&self) -> &'static str {
        "Conv2d"
    }

    fn get_module_sig(&self, tenv: &mut TypeEnv) -> Vec<(MethodName, Type)> {
        vec![
            (
                "new",
                Type::UnresolvedModuleFun("conv", self.get_name(), "new", CSpan::fresh_span()),
            ),
            (
                "forward",
                Type::UnresolvedModuleFun("conv", self.get_name(), "forward", CSpan::fresh_span()),
            ),
        ]
    }

    fn resolve( &self,
        tenv: &mut TypeEnv,
        fn_name: &str,
        arg_ty: Type,
        ret_ty: Type,
        inits: Option<Vec<TyFnAppArg>>
    ) -> Option<Type> {
        match fn_name {
            "forward" => {
                let forward_args = arg_ty.as_args_map()?;
                let x_ty = forward_args.get("x")?;
                if !x_ty.is_resolved() {
                    None
                } else {
                    // ...
                    println!(">>>>>>>>>>:\nTODO: Conv2d!");
                    println!("{:#?}, {:#?}", inits, ret_ty);
                    println!("x_ty: {:?}", x_ty);
                    println!("<<<<<<<<<<");
                    None
                }
            },
            "new" => {
                Some(fun!(
                    "Conv2d",
                    "new",
                    args!(
                        arg!("in_ch", int!()),
                        arg!("out_ch", int!()),
                        arg!("kernel_size", tenv.fresh_var(&CSpan::fresh_span()))
                    ),
                    module!(self.get_name())
                ))
            }
            _ => unimplemented!(),
        }
    }
}

impl Op for Dropout2d {
    fn get_name(&self) -> &'static str {
        "Dropout2d"
    }

    fn get_module_sig(&self, tenv: &mut TypeEnv) -> Vec<(MethodName, Type)> {
        let span = CSpan::fresh_span();
        vec![
            (
                "new",
                fun!("Dropout2d", "new", args!(arg!("p", float!())), module!(self.get_name())),
            ),
            (
                "forward",
                Type::UnresolvedModuleFun("conv", self.get_name(), "forward", CSpan::fresh_span()),
            ),
        ]
    }

    fn resolve(
        &self,
        tenv: &mut TypeEnv,
        fn_name: &str,
        arg_ty: Type,
        ret_ty: Type,
        inits: Option<Vec<TyFnAppArg>>,
    ) -> Option<Type> {
        match fn_name {
            "forward" => {
                let ty = tenv.fresh_var(&CSpan::fresh_span());
                Some(fun!("Dropout2d", "forward", ty.clone(), ty))
            }
            _ => unimplemented!(),
        }
    }
}

#[allow(non_camel_case_types)]
pub struct maxpool2d;

impl Op for maxpool2d {
    fn get_name(&self) -> &'static str {
        "maxpool2d"
    }

    fn get_module_sig(&self, tenv: &mut TypeEnv) -> Vec<(MethodName, Type)> {
        vec![
            (
                "forward",
                Type::UnresolvedModuleFun("conv", self.get_name(), "forward", CSpan::fresh_span()),
            )
        ]
    }

    fn resolve(
        &self,
        tenv: &mut TypeEnv,
        _fn_name: &str,
        arg_ty: Type,
        ret_ty: Type,
        inits: Option<Vec<TyFnAppArg>>,
    ) -> Option<Type> {
        // println!("TODO!");
        None
    }
}
