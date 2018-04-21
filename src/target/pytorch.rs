use ast::{Program, AST};
use std::collections::BTreeMap;

macro_rules! map(
    { $($key:expr => $value:expr),+ } => {
        {
            let mut m = ::std::collections::BTreeMap::new();
            $(
                m.insert($key, $value);
            )+
            m
        }
     };
);

// const IMPORT_MAP: BTreeMap<&str, BTreeMap<&str, (&str,&str)>> = map!{
//     "conv" => map! {
//         "Conv2d"=> ("nn", "Conv2d"),
//         "Dropout2d"=> ("nn", "Dropout2d"),
//         "maxpool2d"=> ("F", "max_pool2d")
//     },
//     "nonlin"=> map! {
//         "relu"=> ("F", "relu")
//     },
//     "lin"=> map! {
//         "Linear"=> ("nn", "Linear")
//     }
// }

pub fn gen(ast: Program) -> String {
    let mut buf = String::new();

    gen_imports(&mut buf, &ast);

    buf
}

pub fn gen_imports(buf: &mut String, ast: &Program) {
    // let out = vec![];
    // let uses: Vec<AST> = ast
    //     .to_list().unwrap()
    //     .into_iter()
    //     .filter(|node| node.is_UseStmt())
    //     .map(|node| {
    //         if let AST::UseStmt{mod_name, imported_names} = node {
    //             let dict = IMPORT_MAP.get(&mod_name);

    //         }
    //     })
    //     .collect();

    // println!("{:?}", uses);
}
