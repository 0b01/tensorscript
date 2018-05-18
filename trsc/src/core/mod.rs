use typing::typed_term::TyFnAppArg;
use errors::Diag;
use typing::{Type, TypeEnv};
use std::collections::HashMap;
use std::fmt::Debug;

mod prelude;
mod conv;
mod lin;
mod reg;
mod nonlin;

pub trait Op: Debug {
    fn get_name(&self) -> &'static str;

    fn ty_sigs(&self, tenv: &mut TypeEnv) -> Vec<(MethodName, Type)>;

    fn pytorch_name(&self) -> String {
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

    fn gen_fn_app(&self, name: &str, _args: &[TyFnAppArg]) -> Result<String, Diag> {
        panic!("{:?}::{} function call is not yet implemented", self, name);
        // unimplemented!()
    }
}

#[derive(Debug)]
pub struct Core {
    maps: HashMap<&'static str, HashMap<&'static str, Box<Op>>>,
}

pub type MethodName = &'static str;

impl Core {
    pub fn new() -> Self {
        let maps = hashmap! {
            "conv" => hashmap! {
                "Conv2d" => box self::conv::Conv2d as Box<Op>,
                "maxpool2d" => box self::conv::maxpool2d as Box<Op>,
            },
            "nonlin" => hashmap! {
                "relu" => box self::nonlin::relu as Box<Op>,
                "tanh" => box self::nonlin::tanh as Box<Op>,
                "leaky_relu" => box self::nonlin::leaky_relu as Box<Op>,
                "log_softmax" => box self::nonlin::log_softmax as Box<Op>,
                "sigmoid" => box self::nonlin::sigmoid as Box<Op>,
            },
            "lin" => hashmap! {
                "Linear" => box self::lin::Linear as Box<Op>,
            },
            "prelude" => hashmap! {
                "view" => box self::prelude::view as Box<Op>,
            },
            "reg" => hashmap! {
                "Dropout2d" => box self::reg::Dropout2d as Box<Op>,
                "BatchNorm1d" => box self::reg::BatchNorm1d as Box<Op>,
            }
        };
        Self {
            maps,
        }
    }
    pub fn import(&self, path_name: &str, mod_name: &str, tenv: &mut TypeEnv) -> Option<Vec<(MethodName, Type)>> {
        let op = self.find(path_name, mod_name)?;
        Some(op.ty_sigs(tenv))
    }

    pub fn find(&self, path_name: &str, mod_name: &str) -> Option<&Box<Op>> {
        let ret = self.maps.get(path_name)?.get(mod_name)?;
        Some(ret)
    }

    pub fn find_mod(&self, mod_name:&str) -> Option<&Box<Op>> {
        self.maps.values()
            .map(|m| m.get(mod_name))
            .filter(|i|i.is_some())
            .collect::<Vec<Option<_>>>()
            .first()?
            .to_owned()
    }
}
