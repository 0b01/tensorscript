use codespan::{ByteIndex, Span, ByteOffset};
use pest::Span as PestSpan;
use codespan::ByteSpan;

/// a ByteSpan is an index into source code
pub struct CSpan {
    sp: ByteSpan,
}

impl CSpan {
    pub fn new(c: ByteSpan) -> CSpan {
        CSpan {
            sp: c,
        }
    }

    pub fn fresh_span() -> ByteSpan {
        // span can be any because it's taken into account for hashing
        Span::new(ByteIndex(0), ByteIndex(0))
    }

    pub fn from_pest(&self, sp: PestSpan) -> ByteSpan {
        // Span::new(ByteIndex(sp.start() as u32 + 1), ByteIndex(sp.end() as u32 + 1))
        self.sp.subspan(ByteOffset(sp.start() as i64), ByteOffset(sp.end() as i64))
    }
}
