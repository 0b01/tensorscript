use parser::term::{NodeAssign, TensorTy, Term};
use std::collections::{HashMap, VecDeque};
use typed_ast::Type;
use core::Core;
use std::fmt::{Debug, Formatter, Error};

pub type TypeId = usize;

#[derive(Clone, Hash, Eq, PartialEq)]
pub enum ModName {
    Global,
    Named(String),
}

impl ModName {
    pub fn as_str(&self) -> &str {
        use self::ModName::*;
        match self {
            &Global => "",
            &Named(ref s) => s,
        }
    }
}

impl Debug for ModName {
    fn fmt(&self, f: &mut Formatter) -> Result<(), Error> {
        use self::ModName::*;
        match self {
            &Named(ref s) => write!(f, "MOD({})", s),
            &Global => write!(f, "MOD(Global)"),
        }
    }
}


/// Represents a single level of scope
#[derive(Debug)]
pub struct Scope {
    types: HashMap<String, Type>,
}

impl Scope {
    pub fn new() -> Scope {
        Scope {
            types: HashMap::new(),
        }
    }
}

#[derive(Debug)]
pub struct TypeEnv {
    counter: TypeId,
    current_mod: ModName,
    modules: HashMap<ModName, (VecDeque<Scope>, VecDeque<Scope>)>,
}

impl TypeEnv {
    pub fn new() -> Self {
        Self {
            counter: 0,
            current_mod: ModName::Global,
            modules: HashMap::new(),
        }
    }

    /// create new dimension type variable
    pub fn fresh_dim(&mut self) -> Type {
        self.counter += 1;
        Type::DIM(self.counter)
    }

    /// create new type variable
    pub fn fresh_var(&mut self) -> Type {
        self.counter += 1;
        // println!("new_var: {}", self.counter);
        Type::VAR(self.counter)
    }

    /// push scope onto stack during tree traversal
    pub fn push_scope(&mut self, mod_name: &ModName) {
        let stack = self.modules.get_mut(mod_name).unwrap();
        stack.0.push_back(Scope::new());
    }

    /// during constraint collection, push the popped scopes back
    pub fn push_scope_collection(&mut self, mod_name: &ModName) {
        let stack = self.modules.get_mut(mod_name).unwrap();
        let scp = stack.1.pop_front().unwrap();
        stack.0.push_back(scp);
    }

    /// exiting block during tree traversal
    pub fn pop_scope(&mut self, mod_name: &ModName) {
        let stack = self.modules.get_mut(mod_name).unwrap();
        let popped = stack.0.pop_back().unwrap();
        stack.1.push_back(popped);
    }

    /// resolve the type of an identifier
    pub fn resolve_type(&self, mod_name: &ModName, alias: &str) -> Option<Type> {
        self.resolve_type_inner(mod_name, alias).or(
            self.resolve_type_inner(&ModName::Global, alias)
        )
    }

    fn resolve_type_inner(&self, mod_name: &ModName, alias: &str) -> Option<Type> {
        let types = self.get_scoped_types(mod_name, alias);
        types.iter().last().cloned()
    }

    fn get_scoped_types(&self, mod_name: &ModName, alias: &str) -> Vec<Type> {
        let stack = self.modules.get(mod_name).unwrap();
        stack.0
            .iter()
            .rev()
            .map(|sc| sc.types.get(alias))
            .filter(|i| i.is_some())
            .map(|i| i.unwrap())
            .cloned()
            .collect::<Vec<Type>>()
    }

    pub fn add_type(&mut self, mod_name: &ModName, alias: &str, ty: Type) {
        let stack = self.modules.entry(mod_name.clone()).or_insert({
            let mut q = VecDeque::new();
            q.push_back(Scope::new());
            (q, VecDeque::new())
        });
        let top = stack.0.len() - 1;
        let scope = stack.0.get_mut(top).unwrap();
        if scope.types.contains_key(alias) {
            panic!("duplicate item");
        }
        let _ = scope.types.insert(alias.to_owned(), ty);
    }

    pub fn add_dim_alias(&mut self, mod_name: &ModName, alias: &str) {
        let tyvar = self.fresh_dim();
        self.add_type(mod_name, alias, tyvar);
    }

    pub fn add_resolved_dim_alias(&mut self, mod_name: &ModName, alias: &str, num: i64) {
        let tyvar = Type::ResolvedDim(num);
        self.add_type(mod_name, alias, tyvar);
    }

    pub fn add_tsr_alias(&mut self, mod_name: &ModName, alias: &str, tsr: &[String]) {
        // first insert all the dims
        tsr.iter()
            .map(|t| {
                if !self.exists(mod_name, t) {
                    self.add_dim_alias(mod_name, t);
                }
            })
            .collect::<Vec<()>>();

        // then insert the tensor itself
        let tsr = self.create_tensor(mod_name, tsr);
        self.add_type(mod_name, alias, tsr)
    }

    pub fn create_tensor(&mut self, mod_name: &ModName, dims: &[String]) -> Type {
        let dims_ty = dims.iter()
            .map(|t| self.resolve_type(mod_name, t).unwrap().clone())
            .collect();
        Type::TSR(dims_ty)
    }

    pub fn resolve_tensor(&mut self, mod_name: &ModName, t: &TensorTy) -> Type {
        match t {
            &TensorTy::Generic(ref dims) => self.create_tensor(mod_name, &dims),
            &TensorTy::TyAlias(ref alias) => self.resolve_type(mod_name, &alias).unwrap(),
        }
    }

    pub fn exists(&self, mod_name: &ModName, alias: &str) -> bool {
        let types = self.get_scoped_types(mod_name, alias);
        types.len() > 0
    }

    pub fn import_node_assign(&mut self, mod_name: &ModName, a: &NodeAssign) {
        match a {
            &NodeAssign::TyAlias {
                ident: ref id,
                rhs: TensorTy::Generic(ref tys),
            } => {
                self.add_tsr_alias(mod_name, id, tys);
            }
            &NodeAssign::ValueAlias {
                ident: ref id,
                rhs: Term::Integer(num),
            } => {
                self.add_resolved_dim_alias(mod_name, id, num);
            }
            _ => unimplemented!(),
        }
    }

    pub fn module(&self) -> &ModName {
        &self.current_mod
    }

    pub fn set_module(&mut self, scp: ModName) {
        self.current_mod = scp;
    }

    pub fn import_module(&mut self, path_name: &str, mod_name: &str) {
        let methods = Core::import(path_name, mod_name);
        for &(ref name, ref ty) in methods.iter() {
            self.add_type(&ModName::Named(mod_name.to_owned()), name, ty.clone());
        }
    }
}
