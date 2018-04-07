#[macro_use]
extern crate pest;
#[macro_use]
extern crate pest_derive;

use pest::Parser;

#[derive(Parser)]
#[grammar = "./tensorscript.pest"]
struct TensorScriptParser;

const test_str: &str = include_str!("../test.tss");

fn main() {
    let pairs = TensorScriptParser::parse(Rule::expr, "blah |> test() |> foo() ");
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
                    node_macro_assign(2, 8, [
                        upper_ident(2, 3),
                        op_assign(4, 5),
                        int_lit(6, 7)
                    ]),
                    node_macro_assign(9, 15, [
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
                        node_macro_assign(28, 34, [
                            upper_ident(28, 29),
                            op_assign(30, 31),
                            int_lit(32, 33)]
                        ),
                        node_macro_assign(35, 41, [
                            upper_ident(35, 36),
                            op_assign(37, 38),
                            int_lit(39, 40)]
                        )]
                    )]
                )]

        };
    }

    #[test]
    fn parse_fn_sig() {
        parses_to! {
            parser: TensorScriptParser,
            input: "(name=1, out=false)",
            rule: Rule::fn_sig,
            tokens: [
                fn_sig(0, 19, [
                    arg_list(1, 18, [
                        arg_pair(1, 7, [
                            ident(1, 5),
                            op_assign(5, 6),
                            int_lit(6, 7)]
                        ),
                        arg_pair(9, 18, [
                            ident(9, 12),
                            op_assign(12, 13),
                            bool_lit(13, 18, [
                                false_lit(13, 18)]
                            )]
                        )]
                    )]
                )
            ]
        }
    }

    #[test]
    fn parse_weights() {
        parses_to! {
            parser: TensorScriptParser,
            input: "weights Mnist<?,c,h,w->?,OUT>{ conv1 = Conv2d::<?->?>::new(); }",
            rule: Rule::weights_decl,
            tokens: [
                weights_decl(0, 63, [
                    weights_decl_head(0, 29, [
                        cap_ident(8, 13),
                        type_sig(13, 29, [
                            type_ident_list(14, 21, [
                                type_ident(14, 15),
                                type_ident(16, 17),
                                type_ident(18, 19),
                                type_ident(20, 21)]
                            ),
                            type_ident_list(23, 28, [
                                type_ident(23, 24),
                                type_ident(25, 28)]
                            )]
                        )]
                    ),
                    weights_decl_body(29, 63, [
                        weights_assign(31, 61, [
                            ident(31, 36),
                            op_assign(37, 38),
                            cap_ident(39, 45),
                            type_sig(47, 53, [
                                type_ident_list(48, 49, [
                                    type_ident(48, 49)]
                                ),
                                type_ident_list(51, 52, [
                                    type_ident(51, 52)]
                                )]
                            ),
                            ident(55, 58),
                            fn_sig(58, 60)]
                        )]
                    )]
                )
            ]
        }
    }

    #[test]
    fn parse_graph() {
        parses_to! {
            parser: TensorScriptParser,
            input: "graph Mnist<?,c,h,w->?,OUT>{ fn new() { } }",
            rule: Rule::graph_decl,
            tokens: [
                graph_decl(0, 43, [
                    graph_decl_head(0, 27, [
                        cap_ident(6, 11),
                        type_sig(11, 27, [
                            type_ident_list(12, 19, [
                                type_ident(12, 13),
                                type_ident(14, 15),
                                type_ident(16, 17),
                                type_ident(18, 19)]
                            ),
                            type_ident_list(21, 26, [
                                type_ident(21, 22),
                                type_ident(23, 26)]
                            )]
                        )]
                    ),
                    graph_decl_body(27, 43, [
                        fn_decl(29, 41, [
                            fn_head(29, 37, [
                                ident(32, 35),
                                fn_sig(35, 37)]
                            ),
                            fn_body(38, 41)]
                        )]
                    )]
                )
            ]
        }
    }

    #[test]
    fn parse_graph_2() {
        parses_to! {
            parser: TensorScriptParser,
            input: "graph Mnist<?,c,h,w->?,OUT>{}",
            rule: Rule::graph_decl,
            tokens: [
                graph_decl(0, 29, [
                    graph_decl_head(0, 27, [
                        cap_ident(6, 11),
                        type_sig(11, 27, [
                            type_ident_list(12, 19, [
                                type_ident(12, 13),
                                type_ident(14, 15),
                                type_ident(16, 17),
                                type_ident(18, 19)]
                            ),
                            type_ident_list(21, 26, [
                                type_ident(21, 22),
                                type_ident(23, 26)]
                            )]
                        )]
                    ),
                    graph_decl_body(27, 29, [])]
                )
            ]
        }
    }

    #[test]
    fn parse_block() {
        parses_to! {
            parser: TensorScriptParser,
            input: "{}",
            rule: Rule::block,
            tokens: []
        }
    }

    #[test]
    fn parse_stmt_assign() {
        parses_to! {
            parser: TensorScriptParser,
            input: "blah = 1;",
            rule: Rule::stmt,
            tokens: [
                stmt(0, 9, [
                    assignment(0, 9, [
                        ident(0, 4),
                        op_assign(5, 6),
                        expr_item(7, 8, [
                            ident(7, 8)]
                        )]
                    )]
                )
            ]
        }
    }

    #[test]
    fn parse_stmt_while_loop() {
        parses_to! {
            parser: TensorScriptParser,
            input: "while 1 { print(text=1); }",
            rule: Rule::stmt,
            tokens: [
                stmt(0, 26, [
                    while_loop(0, 26, [
                        while_lit(0, 5),
                        expr_item(6, 7, [
                            ident(6, 7)]
                        ),
                        stmt(10, 24, [
                            expr_item(10, 23, [
                                fn_call(10, 23, [
                                    ident(10, 15),
                                    fn_arg(16, 22, [
                                        ident(16, 20),
                                        op_assign(20, 21),
                                        expr_item(21, 22, [
                                            ident(21, 22)]
                                        )]
                                    )]
                                )]
                            )]
                        )]
                    )]
                )
            ]
        }
    }

    #[test]
    fn parse_expr_pipes() {
        parses_to! {
            parser: TensorScriptParser,
            input: "blah |> test() |> foo() ",
            rule: Rule::expr,
            tokens: [
                pipes(0, 24, [
                    expr_item(0, 4, [
                        ident(0, 4)]
                    ),
                    pipes(8, 24, [
                        expr_item(8, 14, [
                            fn_call(8, 14, [
                                ident(8, 12)]
                            )]
                        ),
                        expr_item(18, 23, [
                            fn_call(18, 23, [
                                ident(18, 21)]
                            )]
                        )]
                    )]
                )
            ]
        }
    }
}

