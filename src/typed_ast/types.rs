use typed_ast::type_env::TypeId;

#[derive(Debug, PartialEq, Clone)]
pub enum Type {
    Unit,
    Var(TypeId),
    Dim(TypeId),
    Fun { param_ty: Box<Type>, return_ty: Box<Type> },
    Tensor { rank: usize, dims: Vec<Type> },
}
