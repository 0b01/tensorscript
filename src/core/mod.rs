use typed_ast::typed_term::TyFnAppArg;
use typed_ast::{Type, TypeEnv};

mod prelude;
mod conv;
mod lin;
mod nonlin;

pub trait Op {
    fn get_name(&self) -> &'static str;
    fn get_module_sig(&self, tenv: &mut TypeEnv) -> Vec<(MethodName, Type)>;
    fn resolve(
        &self,
        tenv: &mut TypeEnv,
        fn_name: &str,
        arg_ty: Type,
        ret_ty: Type,
        args: Vec<TyFnAppArg>,
        inits: Option<Vec<TyFnAppArg>>,
    ) -> Option<Type> {
        unimplemented!();
    }
}

pub struct Core;

type MethodName = &'static str;

impl Core {
    pub fn import(path_name: &str, mod_name: &str, tenv: &mut TypeEnv) -> Vec<(MethodName, Type)> {
        Core::find(path_name, mod_name).get_module_sig(tenv)
    }

    pub fn find(path_name: &str, mod_name: &str) -> Box<Op> {
        match path_name {
            "conv" => match mod_name {
                "Conv2d" => box self::conv::Conv2d,
                "Dropout2d" => box self::conv::Dropout2d,
                "maxpool2d" => box self::conv::maxpool2d,
                _ => unimplemented!(),
            },
            "nonlin" => match mod_name {
                "relu" => box self::nonlin::relu,
                "log_softmax" => box self::nonlin::log_softmax,
                "sigmoid" => box self::nonlin::sigmoid,
                _ => unimplemented!(),
            },
            "lin" => match mod_name {
                "Linear" => box self::lin::Linear,
                _ => unimplemented!(),
            },
            "prelude" => match mod_name {
                "view" => box self::prelude::view,
                _ => {
                    panic!("{}", mod_name)
                }
            }
            _ => {
                println!("{}::{}", path_name, mod_name);
                unimplemented!()
            }
        }
    }
}
