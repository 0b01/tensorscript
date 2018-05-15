macro_rules! impl_same_shape_op {
    ($path:ident, $name:ident, $statefulness:expr) => {
        #[allow(non_camel_case_types)]
        #[derive(Debug, Clone)]
        pub struct $name;

        impl Op for $name {
            fn get_name(&self) -> &'static str {
                stringify!($name)
            }

            fn is_stateful(&self) -> bool { $statefulness }

            fn gen_import(&self) -> String {
                format!("F.{}", self.get_name())
            }

            fn get_module_sig(&self, _tenv: &mut TypeEnv) -> Vec<(MethodName, Type)> {
                vec![
                    (
                        "forward",
                        UnresolvedModuleFun(stringify!($path), self.get_name(), "forward", CSpan::fresh_span())
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
                        let ty = tenv.fresh_var(&CSpan::fresh_span());
                        Some(Ok(fun!(self.get_name(), "forward", args!(arg!("x", ty.clone())), ty)))
                    }
                    _ => unimplemented!(),
                }
            }

            fn gen_fn_app(&self, name: &str, args: &[TyFnAppArg]) -> Result<String, Diag> {
                let mut buf = String::new();
                match name {
                    "forward" => {
                        write!(buf, "");
                        Ok(buf)
                    }
                    _ => panic!("{} is not implemented", name),
                }
            }
        }
    };
}
