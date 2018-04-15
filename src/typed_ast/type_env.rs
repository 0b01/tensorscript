use std::collections::{HashMap, VecDeque};
use typed_ast::Type;
use parser::term::{TensorTy, NodeAssign, Term};

pub type TypeId = usize;

#[derive(Clone, Debug, Hash, Eq, PartialEq)]
pub enum ModName {
    Global,
    Named(String),
}

/// Represents a single level of scope
#[derive(Debug)]
pub struct Scope {
    aliases: HashMap<String, Type>,
}

impl Scope {
    pub fn new() -> Scope {
        Scope { aliases: HashMap::new() }
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
            modules: {
                let mut hm = HashMap::new();
                hm.insert(ModName::Global, VecDeque::new());
                hm
            }
        }
    }

    pub fn fresh_dim(&mut self) -> Type {
        self.counter += 1;
        Type::Dim(self.counter)
    }

    pub fn fresh_var(&mut self) -> Type {
        self.counter += 1;
        Type::Var(self.counter)
    }

    pub fn push_scope(&mut self, node_name: &ModName) {
        let mut stack = self.modules.get_mut(node_name).unwrap();
        stack.push_back(Scope::new());
    }

    pub fn pop_scope(&mut self, node_name: &ModName) {
        let mut stack = self.modules.get_mut(node_name).unwrap();
        stack.pop_back().unwrap();
    }

    pub fn resolve_alias(&self, node_name: &ModName, alias: &str) -> Option<Type> {
        let aliases = self.get_scoped_aliases(node_name, alias);
        aliases.iter().last().cloned()
    }

    fn get_scoped_aliases(&self, node_name: &ModName, alias: &str) -> Vec<Type> {
        let stack = self.modules.get(node_name).unwrap();
        stack.into_iter()
            .rev()
            .map(|sc| sc.aliases.get(alias))
            .filter(|i| i.is_some())
            .map(|i| i.unwrap())
            .cloned()
            .collect::<Vec<Type>>()
    }

    pub fn add_alias(&mut self, node_name: &ModName, alias: &str, ty: Type) {
        let mut stack = self.modules.entry(node_name.clone()).or_insert({
            let mut q = VecDeque::new(); 
            q.push_back(Scope::new());
            q
        });
        let top = stack.len() - 1;
        let mut scope = stack.get_mut(top).unwrap();
        let _ = scope.aliases.insert(alias.to_owned(), ty);
    }

    pub fn add_dim_alias(&mut self, node_name: &ModName, alias: &str) {
        let tyvar = self.fresh_dim();
        self.add_alias(node_name, alias, tyvar);
    }

    pub fn add_tsr_alias(&mut self, node_name: &ModName, alias: &str, tsr: &[String]) {
        // first insert all the dims
        tsr.iter()
            .map(|t| {
                if !self.exists(node_name, t) {
                    self.add_dim_alias(node_name, t);
                }
            })
            .collect::<Vec<()>>();
        
        // then insert the tensor itself
        let tsr = self.make_tensor(node_name, tsr);
        self.add_alias(node_name, alias, tsr)
    }

    pub fn make_tensor(&mut self, node_name: &ModName, dims: &[String]) -> Type {
        let dims_ty = dims.iter()
            .map(|t| self.resolve_alias(node_name, t).unwrap().clone())
            .collect();
        Type::Tensor {
            rank: dims.len(),
            dims: dims_ty,
        }
    }

    pub fn exists(&self, node_name: &ModName, alias: &str) -> bool {
        let aliases = self.get_scoped_aliases(node_name, alias);
        aliases.len() > 0
    }

    pub fn import_node_assign(&mut self, node_name: &ModName, a: &NodeAssign) {
        match a {
            &NodeAssign::TyAlias {
                ident: ref id,
                rhs: TensorTy::Generic(ref tys)
            } => {
                self.add_tsr_alias(node_name, id, tys);
            },
            &NodeAssign::ValueAlias {
                ident: ref id,
                rhs: Term::Integer(_)
            } => {
                self.add_dim_alias(node_name, id);
            },
            _ => unimplemented!(),
        }
    }

    pub fn module(&self) -> ModName {
        self.current_mod.clone()
    }

    pub fn set_module(&mut self, scp: ModName) {
        self.current_mod = scp;
    }

}