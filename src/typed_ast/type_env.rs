use parser::term::{NodeAssign, TensorTy, Term};
use std::collections::{HashMap, VecDeque};
use typed_ast::Type;

pub type TypeId = usize;

#[derive(Clone, Debug, Hash, Eq, PartialEq)]
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

/// Represents a single level of scope
#[derive(Debug)]
pub struct Scope {
    aliases: HashMap<String, Type>,
}

impl Scope {
    pub fn new() -> Scope {
        Scope {
            aliases: HashMap::new(),
        }
    }
}

#[derive(Debug)]
pub struct TypeEnv {
    counter: TypeId,
    current_mod: ModName,
    modules: HashMap<ModName, VecDeque<Scope>>,
}

impl TypeEnv {
    pub fn new() -> Self {
        Self {
            counter: 0,
            current_mod: ModName::Global,
            modules: HashMap::new(),
        }
    }

    pub fn fresh_dim(&mut self) -> Type {
        self.counter += 1;
        Type::DIM(self.counter)
    }

    pub fn fresh_var(&mut self) -> Type {
        self.counter += 1;
        // println!("new_var: {}", self.counter);
        Type::VAR(self.counter)
    }

    pub fn push_scope(&mut self, mod_name: &ModName) {
        let stack = self.modules.get_mut(mod_name).unwrap();
        stack.push_back(Scope::new());
    }

    pub fn pop_scope(&mut self, mod_name: &ModName) {
        let stack = self.modules.get_mut(mod_name).unwrap();
        stack.pop_back().unwrap();
    }

    pub fn resolve_alias(&self, mod_name: &ModName, alias: &str) -> Option<Type> {
        let aliases = self.get_scoped_aliases(mod_name, alias);
        aliases.iter().last().cloned()
    }

    fn get_scoped_aliases(&self, mod_name: &ModName, alias: &str) -> Vec<Type> {
        let stack = self.modules.get(mod_name).unwrap();
        stack
            .into_iter()
            .rev()
            .map(|sc| sc.aliases.get(alias))
            .filter(|i| i.is_some())
            .map(|i| i.unwrap())
            .cloned()
            .collect::<Vec<Type>>()
    }

    pub fn add_alias(&mut self, mod_name: &ModName, alias: &str, ty: Type) {
        let stack = self.modules.entry(mod_name.clone()).or_insert({
            let mut q = VecDeque::new();
            q.push_back(Scope::new());
            q
        });
        let top = stack.len() - 1;
        let scope = stack.get_mut(top).unwrap();
        if scope.aliases.contains_key(alias) {
            panic!("duplicate item");
        }
        let _ = scope.aliases.insert(alias.to_owned(), ty);
    }

    pub fn add_dim_alias(&mut self, mod_name: &ModName, alias: &str) {
        let tyvar = self.fresh_dim();
        self.add_alias(mod_name, alias, tyvar);
    }

    pub fn add_resolved_dim_alias(&mut self, mod_name: &ModName, alias: &str, num: i64) {
        let tyvar = Type::ResolvedDim(num);
        self.add_alias(mod_name, alias, tyvar);
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
        self.add_alias(mod_name, alias, tsr)
    }

    pub fn create_tensor(&mut self, mod_name: &ModName, dims: &[String]) -> Type {
        let dims_ty = dims.iter()
            .map(|t| self.resolve_alias(mod_name, t).unwrap().clone())
            .collect();
        Type::TSR(dims_ty)
    }

    pub fn resolve_tensor(&mut self, mod_name: &ModName, t: &TensorTy) -> Type {
        match t {
            &TensorTy::Generic(ref dims) => self.create_tensor(mod_name, &dims),
            &TensorTy::TyAlias(ref alias) => self.resolve_alias(mod_name, &alias).unwrap(),
        }
    }

    pub fn exists(&self, mod_name: &ModName, alias: &str) -> bool {
        let aliases = self.get_scoped_aliases(mod_name, alias);
        aliases.len() > 0
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
                rhs: Term::Integer(num), // ...
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
}
