use codespan::CodeMap;

pub mod diagnostic;
pub mod emitter;

pub use self::emitter::Emitter;
pub use self::diagnostic::Diag;