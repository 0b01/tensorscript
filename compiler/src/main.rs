#[macro_use]
extern crate pest;
#[macro_use]
extern crate pest_derive;

use pest::Parser;

#[derive(Parser)]
#[grammar = "./tensorscript.pest"]
struct TensorScriptParser;

fn main() {
    let pairs = TensorScriptParser::parse(Rule::node_decl, "weights Mnist<?,c,h,w->?,OUT>{ conv1 = Conv2d::<?->?>::new() }");
    println!("{}", pairs.unwrap());
}

#[cfg(test)]
mod test {
    use super::*;
    #[test]
    fn true_lit() {
        parses_to! {
            parser: TensorScriptParser,
            input: "true",
            rule: Rule::keyword,
            tokens: [
                keyword(0, 4, [
                    true_lit(0, 4)
                ])
            ]
        };
    }

    #[test]
    fn false_lit() {
        parses_to! {
            parser: TensorScriptParser,
            input: "false",
            rule: Rule::keyword,
            tokens: [
                keyword(0, 5, [
                    false_lit(0, 5)
                ])
            ]
        };
    }

    #[test]
    fn zero_point() {
        parses_to! {
            parser: TensorScriptParser,
            input: "0.",
            rule: Rule::num_lit,
            tokens: [
                num_lit(0, 2, [
                    float_lit(0, 2)
                ])
            ]
        };
    }

    #[test]
    fn one_exp() {
        parses_to! {
            parser: TensorScriptParser,
            input: "1e10",
            rule: Rule::num_lit,
            tokens: [
                num_lit(0, 4, [
                    float_lit(0, 4, [
                    ])
                ])
            ]
        };
    }

    #[test]
    fn int_1() {
        parses_to! {
            parser: TensorScriptParser,
            input: "1",
            rule: Rule::num_lit,
            tokens: [
                num_lit(0, 1, [
                    int_lit(0, 1),
                ])
            ]
        };
    }

    #[test]
    fn parse_use() {
        parses_to! {
            parser: TensorScriptParser,
            input: "use conv::Conv2d",
            rule: Rule::use_stmt,
            tokens: [
                use_stmt(0, 16, [
                    use_lit(0, 3),
                    ident(4, 8),
                    ident(10, 16)
                ])
            ]
        };
    }

    #[test]
    fn parse_use_list() {
        parses_to! {
            parser: TensorScriptParser,
            input: "use conv::{Test, Two}",
            rule: Rule::use_stmt,
            tokens: [
                use_stmt(0, 21, [
                    use_lit(0, 3),
                    ident(4, 8),
                    ident_list(11, 20, [
                        ident(11, 15),
                        ident(17, 20),
                    ])
                ])
            ]
        };
    }

    #[test]
    fn parse_type_signature() {
        parses_to! {
            parser: TensorScriptParser,
            input: "<?,h,w -> 10, h>",
            rule: Rule::type_sig,
            tokens: [
                type_sig(0, 16, [
                    type_ident_list(1, 6, [
                        type_ident(1, 2),
                        type_ident(3, 4),
                        type_ident(5, 6)
                    ]),
                    type_ident_list(10, 15, [
                        type_ident(10, 12),
                        type_ident(14, 15)
                    ])
                ])
            ]
        };
    }

    #[test]
    fn parse_node_head() {
        parses_to! {
            parser: TensorScriptParser,
            input: "node Mnist<?,c,h,w->?,out>",
            rule: Rule::node_decl_head,
            tokens: [
                node_decl_head(0, 26, [
                    node_lit(0, 4),
                    cap_ident(5, 10),
                    type_sig(10, 26, [
                        type_ident_list(11, 18, [
                            type_ident(11, 12),
                            type_ident(13, 14),
                            type_ident(15, 16),
                            type_ident(17, 18)]),
                        type_ident_list(20, 25, [
                            type_ident(20, 21),
                            type_ident(22, 25)
                        ])
                    ])
                ])
            ]
        };
    }
    #[test]
    fn parse_node_decl_block() {
        parses_to! {
            parser: TensorScriptParser,
            input: "{ T = 3; B = 1; }",
            rule: Rule::node_decl_body,
            tokens: [
                node_decl_body(0, 17, [
                    macro_expr(2, 8, [
                        upper_ident(2, 3),
                        op_assign(4, 5),
                        int_lit(6, 7)
                    ]),
                    macro_expr(9, 15, [
                        upper_ident(9, 10),
                        op_assign(11, 12),
                        int_lit(13, 14)
                    ])
                ])
            ]
        };
    }

    #[test]
    fn parse_node() {
        parses_to! {
            parser: TensorScriptParser,
            input: "node Mnist<?,c,h,w->?,OUT>{ T = 3; B = 1; }",
            rule: Rule::node_decl,
            tokens: [
                node_decl(0, 43, [
                    node_decl_head(0, 26, [
                        node_lit(0, 4),
                        cap_ident(5, 10),
                        type_sig(10, 26, [
                            type_ident_list(11, 18, [
                                type_ident(11, 12),
                                type_ident(13, 14),
                                type_ident(15, 16),
                                type_ident(17, 18)]
                            ),
                            type_ident_list(20, 25, [
                                type_ident(20, 21),
                                type_ident(22, 25)]
                            )]
                        )]
                    ),
                    node_decl_body(26, 43, [
                        macro_expr(28, 34, [
                            upper_ident(28, 29),
                            op_assign(30, 31),
                            int_lit(32, 33)]
                        ),
                        macro_expr(35, 41, [
                            upper_ident(35, 36),
                            op_assign(37, 38),
                            int_lit(39, 40)]
                        )]
                    )]
                )]

        };
    }
}

