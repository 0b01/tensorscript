use std::str::FromStr;
use codespan::CodeMap;
use codespan_reporting::termcolor::StandardStream;
use codespan_reporting::{emit, ColorArg, Diagnostic, Severity };
use super::diagnostic::Diag;
use std::process::exit;

#[derive(Debug, Clone)]
pub struct Emitter {
    errs: Vec<Diag>,
    code_map: CodeMap,
    print_ast: bool,
}

impl Emitter {
    pub fn new(code_map: CodeMap, print_ast: bool) -> Self {
        Self {
            errs: vec![],
            code_map,
            print_ast,
        }
    }

    pub fn add(&mut self, e: Diag) {
        self.errs.push(e);
    }

    pub fn print_errs(&self) {
        let mut diagnostics: Vec<Diagnostic> = self.errs
            .iter()
            .map(|e|e.as_diagnostic(&self.code_map))
            .collect();
        let writer = StandardStream::stderr(ColorArg::from_str("auto").unwrap().into());
        let mut is_err = false;
        while let Some(diagnostic) = &diagnostics.pop() { // consumes so it only prints once
            if diagnostic.severity == Severity::Error { is_err = true }
            emit(&mut writer.lock(), &self.code_map, &diagnostic).unwrap();
        }
        if is_err && !self.print_ast { exit(-1) }
    }
}