use core::{MethodName, Op};
use typed_ast::{Type, TypeEnv};
use typed_ast::typed_term::{TyFnAppArg, ArgsVecInto, Ty, TyTerm};
use span::CSpan;

// #[allow(non_camel_case_types)]
pub struct Linear;

impl Op for Linear {
    fn get_name(&self) -> &'static str {
        "Linear"
    }

    fn get_module_sig(&self, tenv: &mut TypeEnv) -> Vec<(MethodName, Type)> {
        use self::Type::*;
        vec![
            ("init_normal", fun!(args!(arg!("std", float!())), unit!())),
            ("new", fun!(args!(arg!("in", int!()), arg!("out", int!())), module!(self.get_name()))),
            ("forward", UnresolvedModuleFun("lin", self.get_name(), "forward", CSpan::fresh_span())),
        ]
    }

    /// output same shape as input
    fn resolve(&self, tenv: &mut TypeEnv, module: Option<Type>, _fn_name: &str, inits: Option<Vec<TyFnAppArg>>) -> Option<Type> {
        if inits.is_some() {
            let hm = inits.unwrap().to_hashmap().unwrap();
            if !hm.contains_key("in") {
                panic!("Initatialize Linear with parameter in=");
            } else if !hm.contains_key("out") {
                panic!("Initatialize Linear with parameter out=");
            }

            let in_dim = hm.get("in").and_then(unwrap_dim).unwrap();
            let out_dim = hm.get("out").and_then(unwrap_dim).unwrap();

            // println!("({:?}, {:?})", in_dim, out_dim);
            // println!("{:#?}", module);
            // unimplemented!()
            None
        } else {
            None
        }
    }
}

fn unwrap_dim(in_dim: &Box<TyTerm>) -> Option<i64> {
    match in_dim.ty() {
        Type::INT(_) => in_dim.int(),
        Type::ResolvedDim(num, _) => Some(num),
        _ => panic!("not a numeric value!"),
    }
}