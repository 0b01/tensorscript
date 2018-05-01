use codespan::ByteSpan;
use core::Core;
use span::CSpan;
/// Type Environment holds the state during type reconstruction
/// which is really just a few tree traversals.
///
/// It handles, in broad strokes, 3 things:
/// 1. Type Aliasing during the first pass (annotate)
/// 2. pushing and popping scopes (during `annotate` and `collect`)
/// 3. module type and method type reconstruction
use parsing::term::{AliasAssign, TensorTy, Term};
use std::collections::{BTreeMap, VecDeque};
use std::fmt::{Debug, Error, Formatter};
use typing::typed_term::TyFnAppArg;
use typing::Type;
use errors::Diag;
use self::ModName::*;

pub type TypeId = usize;

#[derive(Clone, Hash, Eq, PartialEq, PartialOrd, Ord)]
pub enum ModName {
    Global,
    Named(String),
}

impl ModName {
    pub fn as_str(&self) -> &str {
        match self {
            Global => unimplemented!(),
            Named(ref s) => s,
        }
    }
}

impl Debug for ModName {
    fn fmt(&self, f: &mut Formatter) -> Result<(), Error> {
        match self {
            Named(ref s) => write!(f, "MOD({})", s),
            Global => write!(f, "MOD(Global)"),
        }
    }
}

/// Represents a single level of scope
#[derive(Debug)]
pub struct Scope {
    /// type information of aliases
    types: BTreeMap<Alias, Type>,
}

impl Scope {
    pub fn new() -> Scope {
        Scope {
            types: BTreeMap::new(),
        }
    }
}

type InitMap = BTreeMap<String, Vec<TyFnAppArg>>;

#[derive(Debug)]
pub struct TypeEnv {
    dim_counter: TypeId,
    var_counter: TypeId,
    current_mod: ModName,
    modules: BTreeMap<ModName, (VecDeque<Scope>, VecDeque<Scope>, InitMap)>,
}

#[derive(PartialEq, Eq, Hash, Clone, PartialOrd, Ord)]
pub enum Alias {
    Variable(String),
    Function(String),
}

impl Debug for Alias {
    fn fmt(&self, f: &mut Formatter) -> Result<(), Error> {
        match self {
            Alias::Function(a) => write!(f, "F({})", a),
            Alias::Variable(a) => write!(f, "V({})", a),
        }
    }
}

impl Alias {
    pub fn as_str(&self) -> &str {
        match self {
            Alias::Function(s) => s,
            Alias::Variable(s) => s,
        }
    }
}

impl TypeEnv {
    pub fn new() -> Self {
        let mut ret = Self {
            dim_counter: 0,
            var_counter: 0,
            current_mod: Global,
            modules: BTreeMap::new(),
        };

        // import basic functions such as view
        ret.import_prelude().unwrap();

        ret
    }

    /// create new dimension type variable
    pub fn fresh_dim(&mut self, span: &ByteSpan) -> Type {
        self.dim_counter += 1;
        Type::DIM(self.dim_counter, *span)
    }

    /// create new type variable
    pub fn fresh_var(&mut self, span: &ByteSpan) -> Type {
        self.var_counter += 1;
        // println!("new_var: {}", self.counter);
        Type::VAR(self.var_counter, *span)
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

    pub fn resolve_init(&self, mod_name: &ModName, alias: &str) -> Option<Vec<TyFnAppArg>> {
        let stack = &self.modules[mod_name];
        stack.2.get(alias).cloned()
    }

    /// resolve the type of an identifier
    /// first check current mod name, if it doesn not exist,
    /// then check in the global scope
    pub fn resolve_type(&self, mod_name: &ModName, alias: &Alias) -> Option<Type> {
        self.resolve_type_inner(mod_name, alias)
            .or_else(|| self.resolve_type_inner(&Global, alias))
    }

    /// inside the module or global scope, iterate over block scope and find
    /// the last defn of the alias which may be shadowed
    fn resolve_type_inner(&self, mod_name: &ModName, alias: &Alias) -> Option<Type> {
        let types = self.get_scoped_types(mod_name, alias);
        types.iter().last().cloned()
    }

    /// iterate over scopes and find the alias in each
    fn get_scoped_types(&self, mod_name: &ModName, alias: &Alias) -> Vec<Type> {
        let stack = self.modules.get(mod_name)
            .expect(&format!("BUG: Unable to find {:?} in {:?}.", alias, mod_name));
        stack
            .0
            .iter()
            .rev()
            .map(|sc| sc.types.get(alias))
            .filter(|i| i.is_some())
            .map(|i| i.unwrap())
            .cloned()
            .collect()
    }

    /// if current module does not exist, create and insert it, nop otherwise
    pub fn upsert_module(&mut self, mod_name: &ModName) {
        if !self.modules.contains_key(mod_name) {
            self.modules.insert(mod_name.clone(), {
                // if the module does not yet exist, add with an empty scope
                let mut q = VecDeque::new();
                q.push_back(Scope::new());
                (q, VecDeque::new(), BTreeMap::new())
            });
        }
    }

    /// add type alias in current scope
    pub fn add_type(&mut self, mod_name: &ModName, alias: &Alias, ty: Type) -> Result<(), Diag> {
        let stack = self.modules.entry(mod_name.clone()).or_insert({
            // if the module does not yet exist, add with an empty scope
            let mut q = VecDeque::new();
            q.push_back(Scope::new());
            (q, VecDeque::new(), BTreeMap::new())
        });

        let top = stack.0.len() - 1;
        let scope = &mut stack.0[top];
        if scope.types.contains_key(alias) {
            let orig_ty = &scope.types[alias];
            return Err(
                Diag::
                    DuplicateVarInScope(
                        alias.as_str().to_string(),
                        orig_ty.clone(),
                        ty,
                    )
                )
        }
        let _ = scope.types.insert(alias.clone(), ty);

        Ok(())
    }

    /// add type alias in current scope(allows replacement)
    pub unsafe fn add_type_allow_replace(&mut self, mod_name: &ModName, alias: &Alias, ty: Type) {
        let stack = self.modules.entry(mod_name.clone()).or_insert({
            // if the module does not yet exist, add with an empty scope
            let mut q = VecDeque::new();
            q.push_back(Scope::new());
            (q, VecDeque::new(), BTreeMap::new())
        });

        let top = stack.0.len() - 1;
        let scope = &mut stack.0[top];
        // if scope.types.contains_key(alias) {
        //     panic!("duplicate item");
        // }
        let _ = scope.types.insert(alias.clone(), ty);
    }

    /// replace an alias in scope
    pub unsafe fn replace_type(&mut self, mod_name: &ModName, alias: &Alias, ty: Type) {
        let stack = self.modules.entry(mod_name.clone()).or_insert({
            // if the module does not yet exist, add with an empty scope
            let mut q = VecDeque::new();
            q.push_back(Scope::new());
            (q, VecDeque::new(), BTreeMap::new())
        });

        for scope in &mut stack.0 {
            if scope.types.contains_key(alias) {
                let _ = scope.types.insert(alias.clone(), ty.clone());
            }
        }
    }

    /// add stateful initialization in current scope
    pub fn add_init(&mut self, mod_name: &ModName, alias: &str, ty: Vec<TyFnAppArg>) {
        let stack = self.modules.get_mut(&mod_name).unwrap();

        if stack.2.contains_key(alias) {
            panic!("duplicate item");
        }
        let _ = stack.2.insert(alias.to_owned(), ty);
    }

    /// tie an alias with a type variable dimension
    pub fn add_dim_alias(&mut self, mod_name: &ModName, alias: &Alias, span: &ByteSpan) -> Result<(), Diag> {
        let tyvar = self.fresh_dim(span);
        self.add_type(mod_name, alias, tyvar)
    }

    /// tie an alias with a resolved dimension
    pub fn add_resolved_dim_alias(
        &mut self,
        mod_name: &ModName,
        alias: &Alias,
        num: i64,
        span: &ByteSpan,
    ) -> Result<(), Diag> {
        let tyvar = Type::ResolvedDim(num, *span);
        self.add_type(mod_name, alias, tyvar)
    }

    /// tie an alias with a tensor
    pub fn add_tsr_alias(
        &mut self,
        mod_name: &ModName,
        alias: &Alias,
        tsr: &[String],
        span: &ByteSpan,
    ) -> Result<(), Diag> {
        // first insert all the dims
        for t in tsr.iter() {
            let alias = Alias::Variable(t.to_string());
            if !self.exists(mod_name, &alias) {
                self.add_dim_alias(mod_name, &alias, span)?;
            }
        }

        // then insert the tensor itself
        let tsr = self.create_tensor(mod_name, tsr, span);
        self.add_type(mod_name, alias, tsr)
    }

    // make a new tensor based on type signature
    pub fn create_tensor(
        &mut self,
        mod_name: &ModName,
        dims: &[String],
        span: &ByteSpan,
    ) -> Type {
        // each dimension alias in the tensor type signature must exist
        let dims_ty = dims.iter()
            .map(|t| {
                match t.parse::<i64>() {
                    Ok(i) => vec![Type::ResolvedDim(i, *span)],
                    Err(_e) => {
                        let alias = Alias::Variable(t.to_string());
                        let ty = self.resolve_type(mod_name, &alias)
                            .unwrap_or_else(|| self.fresh_dim(span))
                            .clone();
                        if let Type::TSR(vs, _) = ty {
                            vs
                        } else {
                            vec![ty]
                        }
                    }
                }
            })
            .flatten()
            .collect();
        // create the tensor type
        Type::TSR(dims_ty, *span)
    }

    /// generate a tensor from untyped ast tensor signature
    pub fn resolve_tensor(
        &mut self,
        mod_name: &ModName,
        t: &TensorTy,
        _span: &ByteSpan,
    ) -> Type {
        match t {
            TensorTy::Generic(ref dims, ref sp) => {
                self.create_tensor(mod_name, &dims, sp)
            }
            TensorTy::Tensor(ref alias, ref sp) => {
                self.resolve_type(mod_name, &Alias::Variable(alias.to_string()))
                    .unwrap().with_span(sp)
            }
        }
    }

    /// check if an alias exists
    pub fn exists(&self, mod_name: &ModName, alias: &Alias) -> bool {
        let types = self.get_scoped_types(mod_name, alias);
        !types.is_empty()
    }

    /// create aliases for an untyped AST node assign
    pub fn import_node_assign(&mut self, mod_name: &ModName, a: &AliasAssign) -> Result<(), Diag> {
        match a {
            AliasAssign::Tensor {
                ident: ref id,
                rhs: TensorTy::Generic(ref tys, ref sp),
                ..
            } => {
                self.add_tsr_alias(mod_name, &Alias::Variable(id.to_string()), tys, sp)
            }
            AliasAssign::Dimension {
                ident: ref id,
                rhs: Term::Integer(num, _),
                ref span,
            } => {
                self.add_resolved_dim_alias(mod_name, &Alias::Variable(id.to_string()), *num, span)
            }
            _ => unimplemented!(),
        }
    }

    pub fn import_top_level_ty_sig(&mut self, mod_name: &ModName, ty_sig: &TensorTy) -> Result<(), Diag> {
        if let TensorTy::Generic(dims, span) = ty_sig {
            // first insert all the dims
            for t in dims.iter().filter(|t| t.parse::<i64>().is_err()) {
                let alias =  Alias::Variable(t.to_string());
                if !self.exists(mod_name, &alias) {
                    self.add_dim_alias(mod_name, &alias, span)?;
                }
            }
        }

        Ok(())
    }

    /// get current module name
    pub fn module(&self) -> ModName {
        self.current_mod.clone()
    }

    /// set current module name
    pub fn set_module(&mut self, scp: ModName) {
        self.current_mod = scp;
    }

    /// import module type and associated methods into type environment
    pub fn import_module(&mut self, path_name: &str, mod_name: &str) -> Option<Result<(), Diag>> {
        let methods = Core::import(path_name, mod_name, self)?;

        Some(methods.iter().map(|(name, ty)| {
            self.add_type(
                &Named(mod_name.to_owned()),
                &Alias::Function(name.to_string()),
                ty.clone(),
            )
        })
        .collect())
    }

    pub fn import_prelude(&mut self) -> Result<(), Diag> {
        for fun in &vec!["view"] {
            self.add_type(&Global,
                &Alias::Variable(fun.to_string()),
                module!(fun.to_string())
            )?;
            self.import_module("prelude", fun);
        }
        Ok(())
    }

    pub fn resolve_unresolved(
        &mut self,
        ty: &Type,
        fn_name: &str,
        arg_ty: Type,
        ret_ty: Type,
        args: Vec<TyFnAppArg>,
        inits: Option<Vec<TyFnAppArg>>,
    ) -> Result<Option<(Type, bool)>, Diag> {
        // let (mod_name, mod_ty) = {
        //     if let Type::Module(name, opty, _) = module {
        //         (name, opty.clone().map(|i| *i))
        //     } else {
        //         panic!();
        //     }
        // };

        if let Type::UnresolvedModuleFun(ref p0, ref p1, ref p2, ref span) = ty {
            assert_eq!(fn_name.to_owned(), p2.to_owned());
            let find_result = Core::find(p0, p1);
            match find_result {
                Some(op) => {
                    let is_stateful = op.is_stateful();
                    op.resolve(self, fn_name, arg_ty, ret_ty, args, inits)
                        .transpose()
                        .map(|t|
                            t.map(|i| (i, is_stateful))
                        )
                }
                None =>
                    Err(Diag::SymbolNotFound(p1.to_string(), *span)),
            }
        } else {
            unimplemented!();
        }
    }
}
