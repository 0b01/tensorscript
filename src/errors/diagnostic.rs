use codespan_reporting::{Diagnostic, Label, Severity};
use typing::Type;
use std::str::FromStr;
use codespan::CodeMap;
use codespan_reporting::termcolor::StandardStream;
use codespan::{ByteSpan, LineIndex};

#[derive(Debug, Clone)]
pub enum TensorScriptDiagnostic {
    RankMismatch(Type, Type),
    DimensionMismatch(Type, Type),
    ParseError(String, ByteSpan),
    SymbolNotFound(String, ByteSpan),
    ImportError(String, ByteSpan),
    DuplicateVarInScope(String, Type, Type),
}

impl TensorScriptDiagnostic {
    pub fn as_diagnostic(&self, code_map: &CodeMap) -> Diagnostic {
        use self::TensorScriptDiagnostic::*;
        match self {
            DimensionMismatch(Type::ResolvedDim(v1, s1), Type::ResolvedDim(v2,s2)) => {
                Diagnostic::new(
                    Severity::Error,
                    format!("Dimension mismatch: {} != {}", v1, v2),
                )
                .with_label(Label::new_primary(*s1))
                .with_label(Label::new_primary(*s2))
            }

            RankMismatch(Type::TSR(dims1, s1), Type::TSR(dims2, s2)) => {
                Diagnostic::new(
                    Severity::Error,
                    format!("Tensor rank mismatch: rank({:?}) != rank({:?})", dims1, dims2),
                )
                .with_label(Label::new_primary(*s1))
                .with_label(Label::new_primary(*s2))
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
                .with_label(Label::new_primary(*sp))
            },

            SymbolNotFound(msg, sp) => {
                Diagnostic::new(
                    Severity::Error,
                    format!("Cannot find symbol `{}` in scope", msg),
                )
                .with_label(Label::new_primary(*sp))
            }

            ImportError(msg, sp) => {
                Diagnostic::new(
                    Severity::Error,
                    format!("Cannot import symbol `{}`", msg),
                )
                .with_label(Label::new_primary(*sp))
            }

            DuplicateVarInScope(name, ty1, ty2) => {
                Diagnostic::new(
                    Severity::Error,
                    format!("Duplicate symbol in scope: {}: {:?}, {:?}", name, ty1, ty2),
                )
                .with_label(Label::new_primary(ty1.span()))
                .with_label(Label::new_primary(ty2.span()))
            }

            _ => unimplemented!(),
        }
    }

}
