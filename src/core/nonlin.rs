use self::Type::*;
use core::{MethodName, Op};
use std::fmt::Write;
use span::CSpan;
use typing::typed_term::TyFnAppArg;
use typing::{Type, TypeEnv};
use errors::Diag;

impl_same_shape_op!(nonlin, sigmoid, false, {
});

impl_same_shape_op!(nonlin, tanh, false, {
});

impl_same_shape_op!(nonlin, relu, false, {
});

impl_same_shape_op!(nonlin, log_softmax, false, {
}, (arg!("dim", int!())));

impl_same_shape_op!(nonlin, leaky_relu, false, {
    // todo... check supplied params
});
// }, (arg!("p", float!())));
