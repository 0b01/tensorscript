#![recursion_limit="128"]
extern crate proc_macro;
extern crate syn;
#[macro_use]
extern crate quote;

mod attrs;
mod parser;

use attrs::*;
use std::collections::BTreeMap;
use proc_macro::TokenStream;
use parser::{parse_decl, FnDecl};

#[proc_macro_derive(Op, attributes(stateful, name, new, forward))]
pub fn derive(input: TokenStream) -> TokenStream {
    // Construct a string representation of the type definition
    let s = input.to_string();

    // Parse the string representation
    let ast = syn::parse_derive_input(&s).unwrap();

    // Build the impl
    let gen = impl_op(&ast);

    // Return the generated impl
    println!("{:#?}", gen);
    gen.parse().unwrap()
}

fn impl_op(ast: &syn::DeriveInput) -> quote::Tokens {
    if let syn::Body::Enum(_) = ast.body {
        panic!("Cannot derive Op for `enum`");
    }

    let name = &ast.ident;

    let stateful = get_is_stateful(&ast.attrs);
    let op_name = get_op_name(&ast.attrs);
    let fns = get_fns(&ast.attrs);
    let path = get_path(&ast.attrs).unwrap_or_else(|| panic!("no path supplied"));
    let fn_decls = get_fn_decls(path, &fns);
    let ty_sigs = gen_ty_sigs(&fn_decls);
    let resolution = gen_resolution(&fn_decls);

    quote! {
        impl Op for #name {

            fn get_name(&self) -> &'static str {
                #op_name
            }

            fn ty_sigs(&self, _tenv: &mut TypeEnv) -> Vec<(MethodName, Type)> {
                #ty_sigs
            }

            fn is_stateful(&self) -> bool {
                #stateful
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
                #resolution
            }

        }
    }
}


fn get_fn_decls(path: &str, ty_sigs: &BTreeMap<&str, String>) -> Vec<FnDecl> {
    let new_fn = parse_decl(path, "new", &ty_sigs["new"]);
    let forward_fn = parse_decl(path, "forward", &ty_sigs["forward"]);
    vec![new_fn, forward_fn]
}

fn gen_ty_sigs(decls: &[FnDecl]) -> quote::Tokens {
    let ty_sigs: Vec<quote::Tokens> = decls.iter().map(|i|gen_decl(i)).collect();
    quote! {
        vec![
            #(#ty_sigs),*
        ]
    }
}

fn gen_resolution(decls: &[FnDecl]) -> quote::Tokens {
    quote! {
        match fn_name {
            "forward" => {
                let ty = tenv.fresh_var(CSpan::fresh_span());
                Some(Ok(fun!(self.get_name(), "forward", args!(arg!("x", ty.clone())), ty)))
            }
            _ => unimplemented!(),
        }
    }
}

fn gen_decl(fn_decl: &FnDecl) -> quote::Tokens {
    let name = &fn_decl.name;
    let path = &fn_decl.path;

    if fn_decl.resolved {
        let params = &fn_decl.params;
        let tys = &fn_decl.tys;
        quote! {
            (
                #name,
                fun!(
                    self.get_name(),
                    #name,
                    args!(
                        #(
                            arg!(
                                #params,
                                #tys
                            )
                        )*
                    ),
                    module!(self.get_name())
                ),
            )
        }
    } else {
        quote! {
            (
                #name,
                Type::UnresolvedModuleFun(#path, self.get_name(), #name, CSpan::fresh_span()),
            )
        }
    }
}
