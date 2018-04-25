use self::Type::*;
use core::{MethodName, Op};
use span::CSpan;
use typed_ast::typed_term::{ArgsVecInto, Ty, TyFnAppArg, TyTerm};
use typed_ast::{Type, TypeEnv};

use self::TyTerm::*;
pub struct Conv2d;
pub struct Dropout2d;

macro_rules! read_2_tuple {
    ($var:expr) => {
        if let box TyExpr(box TyTuple(_,vs,_),_,_) = $var { // TyExpr
            (vs[0].clone().int()?, vs[1].clone().int()?)
        } else {
            panic!("{:#?}", $var);
        }
    };
}

impl Op for Conv2d {
    fn get_name(&self) -> &'static str {
        "Conv2d"
    }

    fn get_module_sig(&self, _tenv: &mut TypeEnv) -> Vec<(MethodName, Type)> {
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
                    let init_map = inits?.to_btreemap()?;
                    let kernel_size = init_map.get("kernel_size")?;
                    let kernel_ty = kernel_size.ty();
                    let (k0, k1) = if kernel_ty.is_tuple() {
                        let (k0,k1) = read_2_tuple!(kernel_size);
                        (k0, k1)
                    } else {
                        let k0 = kernel_size.int()?;
                        let k1 = k0;
                        (k0, k1)
                    };
                    println!("::::: {} {}", k0, k1);

                    // println!(">>>>>>>>>>:\nTODO: Conv2d!");
                    // // println!("{:#?}, {:#?}", inits, ret_ty);
                    // println!("x_ty: {:?}", x_ty);
                    // println!("kernel_ty: {:?}", kernel_ty);
                    // println!("<<<<<<<<<<");
                    // panic!();
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
