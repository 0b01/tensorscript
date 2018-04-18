
use typed_ast::{Type, TypeEnv};
mod conv;
mod lin;
mod nonlin;

pub trait Op {
    fn get_name(&self) -> String;
    fn get_module_sig(&self, tenv: &mut TypeEnv) -> Vec<(MethodName, Type)>;
    fn resolve(&self, tenv: &mut TypeEnv, module: Option<Type>, fn_name: &str) -> Option<Type> {
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
                _ => unimplemented!(),
            },
            "lin" => match mod_name {
                "Linear" => box self::lin::Linear,
                _ => unimplemented!(),
            },
            _ => unimplemented!(),
        }
    }
}
