use codespan::CodeMap;

pub mod diagnostic;
pub mod emitter;

pub use self::emitter::Emitter;
pub use self::diagnostic::TensorScriptDiagnostic;

pub trait EmitErr {
    fn emit_err(&self);
}