use core::{MethodName, Op};
use errors::Diag;
use span::CSpan;
use typing::typed_term::{ArgsVecInto, Ty, TyFnAppArg, TyTerm};
use typing::{Type, TypeEnv};

use self::TyTerm::*;
pub struct Conv2d;
#[allow(non_camel_case_types)]
pub struct maxpool2d;


macro_rules! read_2_tuple {
    ($var:expr) => {
        if let box TyExpr(box TyTuple(_,vs,_),_,_) = $var { // TyExpr
            (vs[0].clone().int()?, vs[1].clone().int()?)
        } else {
            panic!("{:#?}", $var);
        }
    };
}

macro_rules! read_from_init {
    ($var:expr, $default:expr) => {
        $var
            .map(|t| (t, t.ty()) )
            .and_then(|(t, ty)|
                if let Type::Tuple(..) = ty {
                    Some(read_2_tuple!(t))
                } else {
                    let p0 = t.int()?;
                    Some((p0, p0))
                }
            ).unwrap_or($default)
    };
}

impl Op for Conv2d {
    fn get_name(&self) -> &'static str {
        "Conv2d"
    }

    fn is_stateful(&self) -> bool { true }

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
        _ret_ty: Type,
        _args: Vec<TyFnAppArg>,
        inits: Option<Vec<TyFnAppArg>>
    ) -> Option<Result<Type, Diag>> {
        match fn_name {
            "forward" => {
                let forward_args = arg_ty.as_args_map()?;
                let x_ty = &forward_args["x"];
                if !x_ty.is_resolved() {
                    None
                } else {
                    let init_map = inits?.to_btreemap()?;
                    let (k0, k1) = read_from_init!(init_map.get("kernel_size"), (0, 0));
                    let (p0, p1) = read_from_init!(init_map.get("padding"), (0, 0));
                    let (d0, d1) = read_from_init!(init_map.get("dilation"), (1, 1));
                    let (s0, s1) = read_from_init!(init_map.get("stride"), (1, 1));


                    let in_ch = init_map.get("in_ch").map(|t|t.int().unwrap()).expect("does not have in_ch");
                    let out_ch = init_map.get("out_ch").map(|t|t.int().unwrap()).expect("does not have in_ch");

                    let dims = x_ty.as_vec()?;
                    let (n, c_in, h_in, w_in) = (
                        dims[0].to_owned(),
                        dims[1].to_owned().as_num().unwrap(),
                        dims[2].to_owned().as_num().unwrap(),
                        dims[3].to_owned().as_num().unwrap()
                    );

                    assert_eq!(c_in, in_ch);
                    // println!("BLAH: {:?}", x_ty);
                    let h_out = (h_in + 2 * p0 - d0 * (k0 -1) - 1) / s0 + 1;
                    let w_out = (w_in + 2 * p1 - d1 * (k1 -1) - 1) / s1 + 1;

                    let span = x_ty.span();

                    Some(Ok( // returns a function
                        fun!(
                            "Conv2d",
                            "forward",
                            arg_ty,
                            Type::TSR(vec![
                                n,
                                Type::ResolvedDim(out_ch, span),
                                Type::ResolvedDim(h_out, span),
                                Type::ResolvedDim(w_out, span),
                            ], span)
                        )
                    ))
                }
            },
            "new" => {
                Some(Ok(fun!(
                    "Conv2d",
                    "new",
                    args!(
                        arg!("in_ch", int!()),
                        arg!("out_ch", int!()),
                        arg!("kernel_size", tenv.fresh_var(&CSpan::fresh_span()))
                    ),
                    module!(self.get_name())
                )))
            }
            _ => unimplemented!(),
        }
    }
}

impl Op for maxpool2d {
    fn get_name(&self) -> &'static str {
        "maxpool2d"
    }

    fn is_stateful(&self) -> bool { false }

    fn get_module_sig(&self, _tenv: &mut TypeEnv) -> Vec<(MethodName, Type)> {
        vec![
            (
                "forward",
                Type::UnresolvedModuleFun("conv", self.get_name(), "forward", CSpan::fresh_span()),
            )
        ]
    }

    fn resolve(
        &self,
        _tenv: &mut TypeEnv,
        fn_name: &str,
        arg_ty: Type,
        _ret_ty: Type,
        args: Vec<TyFnAppArg>,
        _inits: Option<Vec<TyFnAppArg>>,
    ) -> Option<Result<Type, Diag>> {
        match fn_name {
            "forward" => {
                let args_ty_map = arg_ty.as_args_map()?;
                let x_ty = args_ty_map.get("x").expect("No x argument");
                let args_map = args.to_btreemap()?;

                if !x_ty.is_resolved() {
                    None
                } else {
                    let (k0, k1) = read_from_init!(args_map.get("kernel_size"), (0, 0));
                    let (p0, p1) = read_from_init!(args_map.get("padding"), (0, 0));
                    let (d0, d1) = read_from_init!(args_map.get("dilation"), (1, 1));
                    let (s0, s1) = read_from_init!(args_map.get("stride"), (k0, k1));

                    let dims = x_ty.as_vec()?;
                    let (n, c_in, h_in, w_in) = (
                        dims[0].to_owned(),
                        dims[1].to_owned(),
                        dims[2].to_owned().as_num().unwrap(),
                        dims[3].to_owned().as_num().unwrap()
                    );
                    // println!("BLAH: {:?}", x_ty);
                    let h_out = (h_in + 2 * p0 - d0 * (k0 -1) - 1) / s0 + 1;
                    let w_out = (w_in + 2 * p1 - d1 * (k1 -1) - 1) / s1 + 1;

                    let span = x_ty.span();

                    Some(Ok( // returns a function
                        fun!(
                            "maxpool2d",
                            "forward",
                            arg_ty,
                            Type::TSR(vec![
                                n,
                                c_in.clone(),
                                Type::ResolvedDim(h_out, span),
                                Type::ResolvedDim(w_out, span),
                            ], span)
                        )
                    ))
                }
            },
            _ => None,
        }
    }
}