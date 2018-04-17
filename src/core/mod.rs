use typed_ast::Type;
mod conv;
mod nonlin;
mod lin;

trait Op {
    fn get_name() -> String;
    fn get_module_sig() -> Vec<(MethodName, Type)>;
    // fn get_type() -> ItemType;
}

pub struct Core;

type MethodName = &'static str;

impl Core {
    pub fn import(path_name: &str, mod_name: &str) -> Vec<(MethodName, Type)> {
        match path_name {
            "conv" => {
                match mod_name {
                    "Conv2d" => self::conv::Conv2d::get_module_sig(),
                    "Dropout2d" => self::conv::Dropout2d::get_module_sig(),
                    "maxpool2d" => self::conv::maxpool2d::get_module_sig(),
                    _ => unimplemented!(),
                }
            },
            "nonlin" => {
                match mod_name {
                    "relu" => self::nonlin::relu::get_module_sig(),
                    "log_softmax" => self::nonlin::log_softmax::get_module_sig(),
                    _ => unimplemented!(),
                }
            },
            "lin" => {
                match mod_name {
                    "Linear" => self::lin::Linear::get_module_sig(),
                    _ => unimplemented!(),
                }
            }
            _ => unimplemented!(),
        }
    }
}