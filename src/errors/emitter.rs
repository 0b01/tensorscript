use std::str::FromStr;
use codespan::CodeMap;
use codespan_reporting::termcolor::StandardStream;
use codespan_reporting::{emit, ColorArg, Diagnostic, Severity };
use super::diagnostic::TensorScriptDiagnostic;
use std::process::exit;

#[derive(Debug, Clone)]
pub struct Emitter {
    errs: Vec<TensorScriptDiagnostic>,
    code_map: CodeMap,
}

impl Emitter {
    pub fn new(code_map: CodeMap) -> Self {
        Self {
            errs: vec![],
            code_map,
        }
    }

    pub fn add(&mut self, e: TensorScriptDiagnostic) {
        self.errs.push(e);
    }

    pub fn print_errs(&self) {
        let diagnostics: Vec<Diagnostic> = self.errs
            .iter()
            .map(|e|e.as_diagnostic(&self.code_map))
            .collect();
        let writer = StandardStream::stderr(ColorArg::from_str("auto").unwrap().into());
        let mut is_err = false;
        for diagnostic in &diagnostics {
            if diagnostic.severity == Severity::Error {
                is_err = true;
            }
            emit(&mut writer.lock(), &self.code_map, &diagnostic).unwrap();
        }
        if is_err { exit(-1) }
    }
}