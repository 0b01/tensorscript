#[allow(unused_imports)]
use codespan::ByteSpan;
#[allow(unused_imports)]
use parsing::term::{Decl, FieldAccess, FnApp, FnAppArg, FnDecl, FnDeclParam, FnTySig, TensorTy,
                   Term, ViewFn, WeightsAssign};
#[allow(unused_imports)]
use span::CSpan;
use typing::type_env::{Alias, ModName, TypeEnv};
#[allow(unused_imports)]
use typing::typed_term::{ArgsVecInto, Ty};
#[allow(unused_imports)]
use typing::typed_term::{TyDecl, TyFieldAccess, TyFnApp, TyFnAppArg, TyFnDecl, TyFnDeclParam,
                            TyGraphDecl, TyNodeDecl, TyTerm, TyUseStmt, TyWeightsAssign,
                            TyWeightsDecl, TyAliasAssign};
use typing::Type;
use std::rc::Rc;
use std::cell::RefCell;
use std::fmt::Write;
use std::collections::{ BTreeSet, BTreeMap };
use errors::{Diag, Emitter};
use core::Core;

pub struct Module {
    core: Rc<RefCell<Core>>,
    pub tenv: Rc<RefCell<TypeEnv>>,
    pub name: String,
    pub ty: Type,
    pub fns: Rc<RefCell<BTreeMap<String, TyFnDecl>>>,
    pub inits: Rc<RefCell<Vec<TyWeightsAssign>>>,
    pub buf: String,
    pub indent: usize,
}

impl Module {
    pub fn new(tenv: Rc<RefCell<TypeEnv>>, ty: Type, name: &str, core: Rc<RefCell<Core>>) -> Self {
        Self {
            core,
            tenv,
            ty,
            name: name.to_owned(),
            buf: String::new(),
            fns: Rc::new(RefCell::new(BTreeMap::new())),
            inits: Rc::new(RefCell::new(vec![])),
            indent: 0,
        }
    }

    pub fn set_fns(&mut self, fns: &Vec<TyFnDecl>) -> Result<(), Diag> {
        for f in fns.iter() {
            self.fns.borrow_mut().insert(
                f.name.as_str().to_owned(),
                f.clone()
            );
        }
        Ok(())
    }

    pub fn generate(&mut self) -> Result<(), Diag> {
        self.generate_class()?;
        let fns_clone = self.fns.clone();
        for (fn_name, f) in fns_clone.borrow().iter() {
            if fn_name == "new" {
                self.tab();
                self.generate_init_fn(f)?;
                self.shift_tab();
            }
            else {
                self.tab();
                self.generate_fn_decl(f)?;
                self.shift_tab();
            }
        }
        Ok(())
    }

    pub fn set_inits(&mut self, inits: &Vec<TyWeightsAssign>) -> Result<(), Diag> {
        self.inits = Rc::new(RefCell::new(inits.clone()));
        Ok(())
    }

    pub fn generate_class(&mut self) -> Result<(), Diag> {
        writeln!(self.buf, "class {}(nn.Module):", self.name)?;
        Ok(())
    }

    fn generate_fn_decl(&mut self, func: &TyFnDecl) -> Result<(), Diag> {
        self.generate_fn_decl_head(func.name.as_str(), func)?;
        Ok(())
    }

    fn generate_fn_decl_head(&mut self, name: &str, func: &TyFnDecl) -> Result<(), Diag> {
        let params = func.fn_params
            .iter()
            .map(|p| format!("{}", p.name))
            .collect::<Vec<_>>()
            .join(", ");
        self.indent()?;
        writeln!(self.buf, "def {}(self, {}):", name, params)?;
        Ok(())
    }

    fn generate_init_fn(&mut self, init_fn: &TyFnDecl) -> Result<(), Diag> {
        self.generate_fn_decl_head("__init__", init_fn)?;
        self.tab();
        let inits = self.inits.clone();
        for init in inits.borrow().iter() {
            self.indent()?;
            write!(self.buf, "self.{} = ", init.name)?;

            {
                let module_name = self.tenv.borrow()
                    .resolve_type(
                        &ModName::Global,
                        &Alias::Variable(init.mod_name.as_str().to_owned())
                    )
                    .unwrap()
                    .as_str().to_owned();
                let core = self.core.borrow();
                let module = core.find_mod(&module_name);
                println!("Op is {:?}", module.unwrap());
                // tood::
                // self.generate_fn_call();
            }
        }
        self.shift_tab();
        Ok(())
    }

    fn generate_fn_call(&mut self, mod_name: &str, fn_name: &str) -> Result<(), Diag> {
        Ok(())
    }

    #[inline(always)]
    fn indent(&mut self) -> Result<(), Diag> {
        write!(self.buf, "{}", " ".repeat(self.indent*4))?;
        Ok(())
    }

    #[inline(always)]
    fn tab(&mut self) {
        self.indent += 1;
    }

    #[inline(always)]
    fn shift_tab(&mut self) {
        self.indent -= 1;
    }
}

pub struct Generator {
    pub emitter: Rc<RefCell<Emitter>>,
    pub tenv: Rc<RefCell<TypeEnv>>,
    pub buf: String,
    pub imports: BTreeSet<(String, String)>,
    pub modules: BTreeMap<String, Module>,
    core: Rc<RefCell<Core>>,
}

impl Generator {
    pub fn new(emitter: Rc<RefCell<Emitter>>, tenv: Rc<RefCell<TypeEnv>>, core: Rc<RefCell<Core>>) -> Self {
        Self {
            emitter,
            tenv,
            buf: String::new(),
            imports: BTreeSet::new(),
            modules: BTreeMap::new(),
            core,
        }
    }

    pub fn generate(&mut self, term: &TyTerm) -> Result<(), Diag> {
        self.collect(term)?;
        self.generate_imports();
        self.generate_modules()?;
        Ok(())
    }

    fn generate_modules(&mut self) -> Result<(), Diag> {
        for (_name, module) in self.modules.iter_mut() {
            writeln!(self.buf, "")?;
            writeln!(self.buf, "# {:?}", module.ty)?;
            module.generate()?;
            writeln!(self.buf, "{}", module.buf)?;
        }
        Ok(())
    }

    fn generate_imports(&mut self) -> Result<(), Diag> {
        writeln!(self.buf, "import torch")?;
        writeln!(self.buf, "from torch.autograd import Variable")?;
        writeln!(self.buf, "import torch.nn as nn")?;
        writeln!(self.buf, "import torch.nn.functional as F")?;
        writeln!(self.buf, "import torch.optim as optim")?;
        writeln!(self.buf, "")?;
        writeln!(self.buf, "# import ops")?;

        for (path_name, mod_name) in self.imports.iter() {
            let import_stmt = self.core.borrow().gen_import(path_name, mod_name);
            writeln!(self.buf, "import {} as {}", import_stmt.unwrap(), mod_name)?;
        }

        Ok(())
    }

    fn collect(&mut self, term: &TyTerm) -> Result<(), Diag> {
        use self::TyTerm::*;
        match term {
            TyProgram(decls) => decls
                .iter()
                .map(|d| self.collect_decl(&d))
                .collect::<Result<_,_>>()?,
            _ => unimplemented!(),
        }
        Ok(())
    }

    fn collect_decl(&mut self, decl: &TyDecl) -> Result<(), Diag> {
        use self::TyDecl::*;
        match decl {
            TyUseStmt(stmt) => {
                for name in stmt.imported_names.iter() {
                    self.imports.insert(
                        (stmt.mod_name.to_owned(),
                         name.to_owned()
                        )
                    );
                }
            },
            TyNodeDecl(decl) => {
                let m = Module::new(self.tenv.clone(), decl.ty_sig.clone(), &decl.name, self.core.clone());
                self.modules.insert(decl.name.to_owned(), m);
            }
            TyWeightsDecl(decl) => {
                let mut m = self.modules.get_mut(&decl.name).unwrap();
                m.set_inits(&decl.inits)?;
            }
            TyGraphDecl(decl) => {
                let mut m = self.modules.get_mut(&decl.name).unwrap();
                m.set_fns(&decl.fns)?;
            }
            _ => panic!("{:#?}", decl)
        }
        Ok(())
    }
}

impl From<::std::fmt::Error> for Diag {
    fn from(error: ::std::fmt::Error) -> Diag {
        Diag::UnknownError
    }
}