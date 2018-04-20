use codespan::{ByteIndex, Span};
use pest::Span as PestSpan;

pub struct CSpan;

impl CSpan {
    pub fn fresh_span() -> Span<ByteIndex> {
        // span can be any because it's not hashed
        Span::new(ByteIndex(0), ByteIndex(0))
    }
    pub fn from_pest(sp: PestSpan) -> Span<ByteIndex> {
        Span::new(ByteIndex(sp.start() as u32), ByteIndex(sp.end() as u32))
    }
}
