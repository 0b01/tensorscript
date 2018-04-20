macro_rules! err {
    ($msg:expr, $span:expr) => {
        TensorScriptDiagnostic::ParseError(
            $msg.to_owned(),
            $span.clone(),
        )
    };
}

macro_rules! eat {
    ($tokens:expr, $err:expr, $span:expr) => {
        {
            let t = $tokens.next();
            t.ok_or(err!($err, $span))
        }
    };


    ($tokens:expr, $rule:ident, $err:expr, $span:expr) => {
        {
            let t = $tokens.next();
            t
                .ok_or(err!($err, $span))
                .and_then(|val| {
                    if Rule::$rule != val.as_rule() {
                        Err(err!(
                            &format!("Type is not {:?}", $rule),
                            $span
                        ))
                    } else {
                        Ok(val)
                    }
                })
        }
    };

    ($tokens:expr, [$( $rule:ident ),+], $err:expr) => {
        $tokens.next()
            .ok_or(err!($err))
            .and_then(|val| {
                $(
                    if Rule::$rule == val.as_rule() {
                        return Ok(val);
                    }
                )*
                return Err(err!("Type is wrong"))
            })
    };
}

macro_rules! to_idents {
    ($ident_list:expr) => {
        $ident_list
            .into_inner()
            .map(|id| id.as_str())
            .map(String::from)
            .collect()
    };
}