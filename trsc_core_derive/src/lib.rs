extern crate proc_macro;
extern crate syn;
#[macro_use]
extern crate quote;

use proc_macro::TokenStream;

#[proc_macro_derive(Op, attributes(stateful))]
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
    let name = &ast.ident;
    let stateful = &ast.attrs[0].value;

    if let syn::Body::Enum(_) = ast.body {
        panic!("Not defined for enum");
    }

    println!("{:#?}", stateful);

    quote! {
        impl Op for #name {

            fn get_name(&self) -> &'static str {
                "test"
            }

            fn ty_sigs(&self, _tenv: &mut TypeEnv) -> Vec<(MethodName, Type)> {
                vec![]
            }

            fn is_stateful(&self) -> bool {
                false
            }

        }
    }
}
