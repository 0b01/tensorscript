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
use std::collections::BTreeSet;
use errors::{Diag, Emitter};
use core::Core;

pub struct Generator {
    pub emitter: Rc<RefCell<Emitter>>,
    pub tenv: Rc<RefCell<TypeEnv>>,
    pub buf: String,
    pub imports: BTreeSet<(String, String)>,
}

impl Generator {
    pub fn new(emitter: Rc<RefCell<Emitter>>, tenv: Rc<RefCell<TypeEnv>>) -> Self {
        Self {
            emitter,
            tenv,
            buf: String::new(),
            imports: BTreeSet::new(),
        }
    }

    pub fn generate(&mut self, term: &TyTerm) -> Result<(), Diag> {
        use self::TyTerm::*;
        self.collect(term)?;
        self.generate_imports();
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
            _ => panic!("{:?}", decl),
        }
        Ok(())
    }
}