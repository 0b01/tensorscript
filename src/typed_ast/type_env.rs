use std::collections::HashMap;
use typed_ast::Type;

pub type TypeId = usize;

#[derive(Debug)]
pub struct TypeEnv {
    counter: TypeId,
    aliases: HashMap<String, HashMap<String, Type>>,
}

impl TypeEnv {
    pub fn new() -> Self {
        Self { counter: 0, aliases: HashMap::new() }
    }

    pub fn fresh(&mut self) -> Type {
        self.counter += 1;
        Type::TyVar(self.counter)
    }

    pub fn resolve_alias(&self, node_name: &str, alias: &str) -> Option<Type> {
        let hm = self.aliases.get(node_name).unwrap();
        hm.get(alias).cloned()
    }

    fn add_alias(&mut self, node_name: &str, alias: &str, ty: Type) {
        let hm = self.aliases.entry(node_name.to_string()).or_insert(HashMap::new());
        let _ = hm.insert(alias.to_owned(), ty);
    }

    pub fn add_dim_alias(&mut self, node_name: &str, alias: &str) {
        let tyvar = self.fresh();
        self.add_alias(node_name, alias, tyvar);
    }

    pub fn add_tsr_alias(&mut self, node_name: &str, alias: &str, tsr: &[String]) {
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

    pub fn make_tensor(&mut self, node_name: &str, dims: &[String]) -> Type {
        let dims_ty = dims.iter()
            .map(|t| self.resolve_alias(node_name, t).unwrap())
            .collect();
        Type::Tensor {
            rank: dims.len(),
            dims: dims_ty,
        }
    }

    pub fn exists(&self, node_name: &str, alias: &str) -> bool {
        let hm = self.aliases.get(node_name).unwrap();
        hm.contains_key(alias)
    }
}