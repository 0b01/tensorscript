use pest::Parser;

#[derive(Parser)]
#[grammar = "tensorscript.pest"]
pub struct TensorScriptParser;