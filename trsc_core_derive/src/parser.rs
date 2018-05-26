use quote::{ToTokens, Tokens};

#[derive(Debug, Clone)]
pub enum Type {
    Float,
    Int,
    Tsr,
    SelfTy,
    Unit,
    //...
}

impl Type {
    fn from_str(s: &str) -> Self {
        use self::Type::*;
        match s {
            "float" => Float,
            "int" => Int,
            "self" => SelfTy,
            "unit" => Unit,
            "tsr0" => Tsr,
            _ => panic!("Unknown type"),
        }
    }
}

impl ToTokens for Type {
    fn to_tokens(&self, tokens: &mut Tokens) {
        use self::Type::*;
        match self {
            Float => tokens.append(quote!{float!()}),
            Int => tokens.append(quote!{int!()}),
            Unit => tokens.append(quote!{unit!()}),
            SelfTy => tokens.append(quote!{module!(self.get_name())}),
            _ => unimplemented!(),
        }
    }
}

#[derive(Clone, Debug)]
pub struct FnDecl {
    pub resolved: bool,
    pub params: Vec<String>,
    pub tys: Vec<Type>,
    pub ret: Type,
    pub name: String,
    pub path: String,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum Token {
    LPAREN,
    RPAREN,
    WORD(String),
    SEMI,
    ARROW,
    QMARK,
    COMMA,
}

pub fn parse_decl(path: &str, name: &str, decl: &str) -> FnDecl {
    let tokens = lex(decl);
    parse(path, name, &tokens)
}

macro_rules! eat {
    ($it:ident, $ty:ident, $msg:expr) => {
        if let Some($ty) = $it.next() {} else {
            panic!($msg)
        }
    }
}

fn parse(path: &str, name: &str, toks: &[Token]) -> FnDecl {
    use self::Token::*;

    let mut it = toks.iter().peekable();
    let mut ret = FnDecl {
        name: name.to_owned(),
        path: path.to_owned(),
        resolved: true,
        params: vec![],
        tys: vec![],
        ret: self::Type::Float,
    };

    if let Some(QMARK) = it.peek() {
        ret.resolved = false;
        it.next();
    } else {
    }

    eat!(it, LPAREN, "lparen not found");

    while let Some(tok) = it.peek().cloned() {
        if *tok == RPAREN {
            it.next();
        } else if let WORD(ref name) = *tok {
            // param name
            ret.params.push(name.clone());
            it.next();
            eat!(it, SEMI, "semi");
            // param ty
            if let Some(WORD(ref tyword)) = it.next() {
                let ty: Type = Type::from_str(tyword.as_str());
                ret.tys.push(ty);
            } else {
                panic!("No param type specified");
            }
        } else if ARROW == *tok {
            // return type
            it.next();
            if let Some(WORD(ref tyword)) = it.next() {
                let ty: Type = Type::from_str(tyword.as_str());
                ret.ret = ty;
                return ret;
            } else {
                panic!("No ret ty specified");
            }
        } else {
            it.next();
        }
    }

    ret
}


fn lex(decl: &str) -> Vec<Token> {
    use self::Token::*;

    let mut it = decl.chars().peekable();
    let mut toks = vec![];
    while let Some(c) = it.peek().cloned() {
        match c {
            '(' => {
                toks.push(LPAREN);
                it.next();
            }
            ')' => {
                toks.push(RPAREN);
                it.next();
            }
            '?' => {
                toks.push(QMARK);
                it.next();
            }
            ' ' | '\n' => {
                it.next();
            }
            'A'...'z' => {
                let mut buf = String::new();
                while let Some(ch) = it.peek().cloned() {
                    if ch.is_alphanumeric() || ch == '_' {
                        buf.push(ch);
                        it.next();
                    } else {
                        break;
                    }
                }
                toks.push(WORD(buf));
            }
            ':' => {
                toks.push(SEMI);
                it.next();
            }
            ',' => {
                toks.push(COMMA);
                it.next();
            }
            '-' => {
                it.next();
                if let Some('>') = it.next() {
                    toks.push(ARROW);
                    it.next();
                } else {
                    panic!("malformed");
                }
            }
            _ => {
                panic!("{}", c);
            }
        }
    }
    toks
}