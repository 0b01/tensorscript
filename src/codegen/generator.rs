use codespan::ByteSpan;
use parsing::term::{Decl, FieldAccess, FnApp, FnAppArg, FnDecl, FnDeclParam, FnTySig, TensorTy,
                   Term, ViewFn, WeightsAssign};
use span::CSpan;
use typing::type_env::{Alias, ModName, TypeEnv};
use typing::typed_term::{ArgsVecInto, Ty};
use typing::typed_term::{TyDecl, TyFieldAccess, TyFnApp, TyFnAppArg, TyFnDecl, TyFnDeclParam,
                            TyGraphDecl, TyNodeDecl, TyTerm, TyUseStmt, TyWeightsAssign,
                            TyWeightsDecl, TyAliasAssign};
use typing::Type;
use std::rc::Rc;
use std::cell::RefCell;
use std::fmt::Write;
use std::process::exit;
use std::collections::{ BTreeSet, BTreeMap };
use errors::{Diag, Emitter};
use core::Core;

pub struct Module {
    pub name: String,
    pub ty: Type,
    pub fns: Rc<RefCell<BTreeMap<String, TyFnDecl>>>,
    pub inits: Vec<TyWeightsAssign>,
    pub buf: String,
    pub indent: usize,
}

impl Module {
    pub fn new(ty: Type, name: &str) -> Self {
        Self {
            ty,
            name: name.to_owned(),
            buf: String::new(),
            fns: Rc::new(RefCell::new(BTreeMap::new())),
            inits: vec![],
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
        // self.generate_inits()?;
        let fns_clone = self.fns.clone();
        for (fn_name, f) in fns_clone.borrow().iter() {
            if fn_name == "new" {
                self.indent += 1;
                self.generate_inits(f)?;
                self.indent -= 1;
            }
            else {
                self.indent += 1;
                self.generate_fn(f)?;
                self.indent -= 1;
            }
        }
        Ok(())
    }

    pub fn set_inits(&mut self, inits: &Vec<TyWeightsAssign>) -> Result<(), Diag> {
        self.inits = inits.clone();
        Ok(())
    }

    pub fn generate_class(&mut self) -> Result<(), Diag> {
        writeln!(self.buf, "class {}(nn.Module):", self.name);
        Ok(())
    }

    pub fn generate_fn(&mut self, func: &TyFnDecl) -> Result<(), Diag> {
        let params = func.fn_params
            .iter()
            .map(|p| format!("{}", p.name))
            .collect::<Vec<_>>()
            .join(", ");
        write!(self.buf, "{}", " ".repeat(self.indent*4));
        writeln!(self.buf, "def {}(self, {}):", func.name.as_str(), params);
        Ok(())
    }

    pub fn generate_inits(&mut self, init_fn: &TyFnDecl) -> Result<(), Diag> {
        let params = init_fn.fn_params
            .iter()
            .map(|p| format!("{}", p.name))
            .collect::<Vec<_>>()
            .join(", ");
        write!(self.buf, "{}", " ".repeat(self.indent*4));
        writeln!(self.buf, "def __init__(self, {}):", params);
        // for init in self.inits.unwrap().iter() {
        //     ()
        // }
        Ok(())
    }
}

pub struct Generator {
    pub emitter: Rc<RefCell<Emitter>>,
    pub tenv: Rc<RefCell<TypeEnv>>,
    pub buf: String,
    pub imports: BTreeSet<(String, String)>,
    pub modules: BTreeMap<String, Module>,
}

impl Generator {
    pub fn new(emitter: Rc<RefCell<Emitter>>, tenv: Rc<RefCell<TypeEnv>>) -> Self {
        Self {
            emitter,
            tenv,
            buf: String::new(),
            imports: BTreeSet::new(),
            modules: BTreeMap::new(),
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
            writeln!(self.buf, "");
            writeln!(self.buf, "# {:?}", module.ty);
            module.generate()?;
            writeln!(self.buf, "{}", module.buf);
        }
        Ok(())
    }

    fn generate_imports(&mut self) {
        writeln!(self.buf, "import torch");
        writeln!(self.buf, "from torch.autograd import Variable");
        writeln!(self.buf, "import torch.nn as nn");
        writeln!(self.buf, "import torch.nn.functional as F");
        writeln!(self.buf, "import torch.optim as optim");
        writeln!(self.buf, "");
        writeln!(self.buf, "# import ops");

        for (path_name, mod_name) in self.imports.iter() {
            let import_stmt = Core::gen_import(path_name, mod_name);
            writeln!(self.buf, "import {} as {}", import_stmt.unwrap(), mod_name);
        }
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
                let m = Module::new(decl.ty_sig.clone(), &decl.name);
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