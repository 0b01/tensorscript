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

#[proc_macro_derive(Op, attributes(stateful, path, init_normal, new, forward))]
pub fn derive(input: TokenStream) -> TokenStream {
    // Construct a string representation of the type definition
    let s = input.to_string();

    // Parse the string representation
    let ast = syn::parse_derive_input(&s).unwrap();

    // Build the impl
    let gen = impl_op(&ast);

    // Return the generated impl
    gen.parse().unwrap()
}

fn impl_op(ast: &syn::DeriveInput) -> quote::Tokens {
    if let syn::Body::Enum(_) = ast.body {
        panic!("Cannot derive Op for `enum`");
    }

    let name = &ast.ident;

    let stateful = get_is_stateful(&ast.attrs);
    let op_name = name.to_string();
    let fns = get_fns(&ast.attrs);
    let path = get_path(&ast.attrs).unwrap_or_else(|| panic!("no path supplied"));
    let fn_decls = get_fn_decls(path, &fns);
    let ty_sigs = gen_ty_sigs(&fn_decls);

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
        }
    }
}


fn get_fn_decls(path: &str, ty_sigs: &BTreeMap<&str, String>) -> Vec<FnDecl> {
    ty_sigs
        .iter()
        .map(|(k,v)| {
            parse_decl(path, k, v)
        })
        .collect()
}

fn gen_ty_sigs(decls: &[FnDecl]) -> quote::Tokens {
    let ty_sigs: Vec<quote::Tokens> = decls.iter().map(|i|gen_decl(i)).collect();
    quote! {
        vec![
            #(#ty_sigs),*
        ]
    }
}

fn gen_decl(fn_decl: &FnDecl) -> quote::Tokens {
    let name = &fn_decl.name;
    let path = &fn_decl.path;

    if fn_decl.resolved {
        let params = &fn_decl.params;
        let tys = &fn_decl.tys;
        let ret = &fn_decl.ret;
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
                        ),*
                    ),
                    #ret
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
