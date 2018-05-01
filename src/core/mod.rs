use typing::typed_term::TyFnAppArg;
use errors::Diag;
use typing::{Type, TypeEnv};

#[macro_use]
mod macros;

mod prelude;
mod conv;
mod lin;
mod reg;
mod nonlin;

pub trait Op {
    fn get_name(&self) -> &'static str;
    fn get_module_sig(&self, tenv: &mut TypeEnv) -> Vec<(MethodName, Type)>;
    fn gen_import(&self) -> String {
        format!("nn.{}", self.get_name())
    }
    fn is_stateful(&self) -> bool;
    fn resolve(
        &self,
        _tenv: &mut TypeEnv,
        _fn_name: &str,
        _arg_ty: Type,
        _ret_ty: Type,
        _args: Vec<TyFnAppArg>,
        _inits: Option<Vec<TyFnAppArg>>,
    ) -> Option<Result<Type, Diag>> {
        unimplemented!();
    }
}

pub struct Core;

type MethodName = &'static str;

impl Core {
    pub fn import(path_name: &str, mod_name: &str, tenv: &mut TypeEnv) -> Option<Vec<(MethodName, Type)>> {
        let op = Core::find(path_name, mod_name)?;
        Some(op.get_module_sig(tenv))
    }

    pub fn gen_import(path_name: &str, mod_name: &str) -> Option<String> {
        let op = Core::find(path_name, mod_name)?;
        Some(op.gen_import())
    }

    pub fn find(path_name: &str, mod_name: &str) -> Option<Box<Op>> {
        match path_name {
            "conv" => match mod_name {
                "Conv2d" => Some(box self::conv::Conv2d),
                "maxpool2d" => Some(box self::conv::maxpool2d),
                _ => None,
            },
            "nonlin" => match mod_name {
                "relu" => Some(box self::nonlin::relu),
                "tanh" => Some(box self::nonlin::tanh),
                "leaky_relu" => Some(box self::nonlin::leaky_relu),
                "log_softmax" => Some(box self::nonlin::log_softmax),
                "sigmoid" => Some(box self::nonlin::sigmoid),
                _ => None,
            },
            "lin" => match mod_name {
                "Linear" => Some(box self::lin::Linear),
                _ => None,
            },
            "prelude" => match mod_name {
                "view" => Some(box self::prelude::view),
                _ => None,
            }
            "reg" => match mod_name {
                "Dropout2d" => Some(box self::reg::Dropout2d),
                "BatchNorm1d" => Some(box self::reg::BatchNorm1d),
                _ => None,
            }
            _ => {
                println!("{}::{}", path_name, mod_name);
                None
            }
        }
    }
}
