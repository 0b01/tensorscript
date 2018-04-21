use std::str::FromStr;
use codespan::CodeMap;
use codespan_reporting::termcolor::StandardStream;
use codespan_reporting::{emit, ColorArg, Diagnostic, Label, Severity};
use typed_ast::Type;
use codespan::{ByteSpan, LineIndex};

#[derive(Debug)]
pub enum TensorScriptDiagnostic {
    RankMismatch(Type, Type),
    DimensionMismatch(Type, Type),
    ParseError(String, ByteSpan),
}

impl TensorScriptDiagnostic {

    pub fn print_err(&self, code_map: &CodeMap) {
        let diagnostic = self.into_diagnostic(code_map);
        let writer = StandardStream::stderr(ColorArg::from_str("auto").unwrap().into());
        emit(&mut writer.lock(), &code_map, &diagnostic).unwrap();
    }

    pub fn into_diagnostic(&self, code_map: &CodeMap) -> Diagnostic {
        use self::TensorScriptDiagnostic::*;
        match self {
            DimensionMismatch(Type::ResolvedDim(v1, s1), Type::ResolvedDim(v2,s2)) => {
                Diagnostic::new(
                    Severity::Error,
                    format!("Dimension mismatch: {} != {}", v1, v2),
                )
                .with_label(Label::new_primary(s1.clone()))
                .with_label(Label::new_primary(s2.clone()))
            }

            RankMismatch(Type::TSR(dims1, s1), Type::TSR(dims2, s2)) => {
                Diagnostic::new(
                    Severity::Error,
                    format!("Tensor rank mismatch: rank({:?}) != rank({:?})", dims1, dims2),
                )
                .with_label(Label::new_primary(s1.clone()))
                .with_label(Label::new_primary(s2.clone()))
            },

            ParseError(msg, sp) => {
                // Since error points to the next line,
                // also print the line before
                let idx = sp.start();
                let file = code_map.find_file(idx).unwrap();
                let line = file.find_line(idx).unwrap();
                let prev_line = line.to_usize() - 1;
                let prev_line_span = file.line_span(LineIndex(prev_line as u32)).unwrap();
                Diagnostic::new(
                    Severity::Error,
                    format!("{} on line {}:", msg, prev_line + 1),
                )
                .with_label(Label::new_primary(prev_line_span))
                .with_label(Label::new_primary(sp.clone()))
            }

            _ => panic!("unimplemented errors")
        }
    }

}

pub struct Errors {
    pub errs: Vec<TensorScriptDiagnostic>,
}

impl Errors {
    pub fn new() -> Self {
        Errors { errs: vec![] }
    }

    pub fn add(&mut self, e: TensorScriptDiagnostic) {
        self.errs.push(e);
    }

    pub fn print_errs(&self, code_map: &CodeMap) {
        let diagnostics: Vec<Diagnostic> = self.errs.iter().map(|e|e.into_diagnostic(code_map)).collect();
        let writer = StandardStream::stderr(ColorArg::from_str("auto").unwrap().into());
        for diagnostic in &diagnostics {
            emit(&mut writer.lock(), &code_map, &diagnostic).unwrap();
        }
    }
}