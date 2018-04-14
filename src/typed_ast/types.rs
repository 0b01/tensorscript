use typed_ast::type_env::TypeId;

#[derive(Debug, PartialEq, Clone)]
pub enum Type {
    Var(TypeId),
    Int,
    Float,
    Bool,
    Dim(TypeId),
    Fun { param_ty: Box<Type>, return_ty: Box<Type> },
    Tensor { rank: usize, dims: Vec<Type> },
}
