// use std::fs::File;
// use std::io::Read;


// #[test]
// fn mnist() {
//     use type_reconstruction::constraint::Constraints;
//     use type_reconstruction::unifier::Unifier;
//     use typed_ast::annotate::annotate;
//     use typed_ast::type_env::TypeEnv;
//     let fname = "examples/xor.trs";
//     let mut file = File::open(fname).expect("Unable to open the file");
//     let mut src = String::new();
//     file.read_to_string(&mut src).expect("Unable to read the file");


//     let mut code_map = CodeMap::new();
//     let file_map = code_map.add_filemap("test".into(), src.clone());
//     let cspan = span::CSpan::new(file_map.span());

//     let program = parser::parse_str(&src, &cspan).unwrap();
//     // println!("{:#?}", program);

//     let mut tenv = TypeEnv::new();
//     let ast = annotate(&program, &mut tenv);
//     // println!("{}", ast);
//     // println!("initial tenv: {:#?}", tenv);

//     let mut cs = Constraints::new();
//     cs.collect(&ast, &mut tenv);
//     // println!("{:#?}", cs);

//     let mut unifier = Unifier::new();
//     let mut subs = unifier.unify(cs.clone(), &mut tenv);
//     unifier.print_errs(&code_map);
//     // println!("{:#?}", subs);
//     // println!("{:#?}", subs.apply(&cs));
//     let test = type_reconstruction::inferred_ast::subs(&ast, &mut subs);
//     println!("{:#?}", test);
//     println!("{:#?}", tenv);

// }