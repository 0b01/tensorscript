pub type TypeId = usize;

/// Possible types for function arguments
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum FuncArgType {
    /// A single value of the specified type
    Arg(TypeId),
    Args(Vec<TypeId>),
}

/// An item is anything that can be declared
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ItemType {
    Generic,

    /// A struct has a single definition with any number of
    /// fields and generics
    /// Structs can have impls which contain methods for that
    /// struct
    Struct {
        //TODO: fields, generics, etc.
    },

    /// Definition of a function's type
    Op {
        args: Vec<FuncArgType>,
        //TODO: Support array return types
        return_type: TypeId,
    },
}

impl ItemType {
    /// Returns true if this item type matches the given function signature (args, return type)
    /// Returns false if this item type is not a function
    /// Note: Variadic matching is only done one-way
    ///
    /// That means that this will return false:
    ///     self = Function {args: [Arg(1)], return_type: 0}
    ///     expected_args = [Variadic(1)]
    ///     return_type = 0
    ///
    /// However, this will return true (as expected):
    ///     self = Function {args: [Variadic(1)], return_type: 0}
    ///     expected_args = [Arg(1)]
    ///     return_type = 0
    pub fn matches_signature(
        &self,
        expected_args: &Vec<FuncArgType>,
        expected_return_type: TypeId,
    ) -> bool {
        let mut expected_args = expected_args.iter().peekable();
        match *self {
            ItemType::Op {
                ref args,
                return_type,
            } => {
                (return_type == expected_return_type &&
                // All the args must match an argument in expected_args
                args.iter().all(|arg| match *arg {
                    FuncArgType::Arg(type_id) => match expected_args.peek() {
                        Some(&&FuncArgType::Arg(arg_id)) if type_id == arg_id => {
                            expected_args.next();
                            true
                        },
                        _ => false,
                    },
                    _ => unimplemented!(),
                }))
            }
            _ => false,
        }
    }
}
