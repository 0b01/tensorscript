use self::Type::*;
use core::{MethodName, Op};
use span::CSpan;
use typing::typed_term::TyFnAppArg;
use typing::{Type, TypeEnv};
use errors::Diag;

impl_same_shape_op!(nonlin, sigmoid, {
});

impl_same_shape_op!(nonlin, tanh, {
});

impl_same_shape_op!(nonlin, relu, {
});

impl_same_shape_op!(nonlin, log_softmax, {
}, (arg!("dim", int!())));

impl_same_shape_op!(nonlin, leaky_relu, {

});
// }, (arg!("p", float!())));
