#[allow(unused_imports)]
use codespan::ByteSpan;
#[allow(unused_imports)]
use span::CSpan;
use typing::type_env::{Alias, ModName, TypeEnv};
#[allow(unused_imports)]
use typing::typed_term::{ArgsVecInto, Conversion};
#[allow(unused_imports)]
use typing::typed_term::{TyDecl, TyFieldAccess, TyFnApp, TyFnAppArg, TyFnDecl, TyFnDeclParam,
                            TyGraphDecl, TyNodeDecl, TyTerm, TyUseStmt, TyWeightsAssign,
                            TyWeightsDecl, TyAliasAssign};
use typing::Type;
use std::rc::Rc;
use std::cell::RefCell;
use std::fmt::Write;
use std::collections::{ BTreeSet, BTreeMap, VecDeque };
use errors::{Diag, Emitter};
use core::{Core, Op};

type VarName = String;
type FnName = String;

enum Item {
    FnApp(Option<VarName>, FnName, Vec<TyFnAppArg>),
    SelfFnApp(Option<VarName>, FnName, Vec<TyFnAppArg>),
    Ident(bool, String),
    ViewFn(Option<VarName>, Type),
}

pub struct Module {
    core: Rc<RefCell<Core>>,
    pub tenv: Rc<RefCell<TypeEnv>>,
    pub name: String,
    pub ty: Type,
    pub fns: Rc<RefCell<BTreeMap<String, TyFnDecl>>>,
    pub inits: Rc<RefCell<Vec<TyWeightsAssign>>>,
    pub buf: String,
    pub indent: usize,
    codegen_stack: VecDeque<Item>,
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
            codegen_stack: VecDeque::new(),
        }
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
        self.generate_class_head()?;
        let fns_clone = self.fns.clone();
        for (fn_name, f) in fns_clone.borrow().iter().rev() {
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

    fn collect_term(&mut self, term: &TyTerm, var: Option<String>) -> Result<(), Diag> {
        use self::TyTerm::*;
        match term {
            TyBlock{stmts, ret, ..} => {
                self.collect_term(&stmts, var.clone())?;
                self.collect_term(&ret, var)?;
            }
            TyList(terms) => terms
                .iter()
                .map(|t| self.collect_term(t, var.clone()))
                .collect::<Result<_,_>>()?,
            TyExpr(t,ty,_) => {
                self.collect_term(t, var)?;
            }
            TyFnApp(box fn_app) => {
                self.collect_fn_app(fn_app, var)?;
            },
            TyIdent(_t,i,..) => self.codegen_stack
                .push_back(Item::Ident(var.is_none(), i.as_str().to_owned())),
            TyInteger(..) => (),
            TyFloat(..) => (),
            _ => panic!("{:#?}", term),
        }
        Ok(())
    }

    fn generate_fn(&mut self) -> Result<(), Diag> {
        while let Some(item) = self.codegen_stack.pop_back() {
            match item {
                Item::FnApp(var_name, fn_name, args) => {
                    self.indent()?;
                    let mut is_global = false;
                    let module_name = self.tenv.borrow()
                        .resolve_type(
                            &ModName::Named(self.name.to_owned()),
                            &Alias::Variable(fn_name.to_owned()),
                        )
                        .unwrap_or_else(|| {
                            is_global = true;
                            self.tenv.borrow()
                                .resolve_type(
                                    &ModName::Global,
                                    &Alias::Variable(fn_name.to_owned()),
                                ).unwrap()
                        })
                        .as_string();

                    match var_name {
                        Some(v) => write!(self.buf, "{} = ", v)?,
                        None => write!(self.buf, "return ")?,
                    };

                    if is_global {
                        write!(self.buf, "{}(",  fn_name)?;
                    } else {
                        write!(self.buf, "self.{}(", fn_name)?;
                    }

                    let core_cloned = self.core.clone();
                    let core = core_cloned.borrow();
                    let op = core.find_mod(&module_name).unwrap();
                    let out = op.gen_fn_app("forward", args.as_slice())?;
                    write!(self.buf, "{}", out)?;
                    writeln!(self.buf, ")")?;
                }
                Item::SelfFnApp(var_name, fn_name, args) => {
                    self.indent()?;
                    match var_name {
                        Some(v) => write!(self.buf, "{} = ", v)?,
                        None => write!(self.buf, "return ")?,
                    };
                    write!(self.buf, "self.{}(", fn_name)?;
                    let s = args.to_btreemap().unwrap().keys().cloned().collect::<Vec<_>>().join(", ");
                    write!(self.buf, "{}", s)?;
                    writeln!(self.buf, ")")?;
                }
                Item::Ident(ret, name) => {
                    if ret {
                        self.indent()?;
                        writeln!(self.buf, "return {}", name)?;
                    }
                }
                Item::ViewFn(var_name, ty) => {
                    self.indent()?;
                    match var_name {
                        Some(name) => {
                            writeln!(self.buf, "{} = {}.view({})", name, name, ty.as_string())?;
                        }
                        None => {
                            writeln!(self.buf, "NONE")?;
                            // ...
                        }
                    }
                }
            }
        }
        Ok(())
    }

    fn collect_fn_app(&mut self, fn_app: &TyFnApp, var_name: Option<String>) -> Result<(), Diag> {
        if fn_app.mod_name == Some("view".to_owned()) {
            self.codegen_stack.push_back(Item::ViewFn(
                var_name,
                fn_app.ret_ty.clone(),
            ))
        } else {
            if fn_app.orig_name == Some("self".to_owned())  {
                self.codegen_stack.push_back(Item::SelfFnApp(
                    var_name,
                    fn_app.name.as_str().to_owned(),
                    fn_app.args.clone()
                ));
            } else {
                self.codegen_stack.push_back(Item::FnApp(
                    var_name,
                    fn_app.orig_name.clone().unwrap().to_owned(),
                    fn_app.args.clone()
                ));
            }
        }

        for arg in fn_app.args.iter() {
            self.collect_term(&arg.arg, arg.name.clone())?;
        }
        Ok(())
    }

    pub fn generate_class_head(&mut self) -> Result<(), Diag> {
        writeln!(self.buf, "class {}(nn.Module):", self.name)?;
        self.tab();
        self.indent()?;
        writeln!(self.buf, "'''{:?}'''", self.ty)?;
        self.shift_tab();
        Ok(())
    }

    fn generate_fn_decl(&mut self, func: &TyFnDecl) -> Result<(), Diag> {
        self.generate_fn_decl_head(func.name.as_str(), func)?;
        self.tab();
        // self.indent()?;
        // writeln!(self.buf, "'''{:?}'''", func.fn_ty)?;
        self.collect_term(&func.func_block, None)?;
        self.generate_fn()?;
        self.shift_tab();
        Ok(())
    }

    fn generate_fn_decl_head(&mut self, name: &str, func: &TyFnDecl) -> Result<(), Diag> {
        let params = func.fn_params
            .iter()
            .map(|p| format!("{}", p.name))
            .collect::<Vec<_>>();
        self.indent()?;
        if !params.is_empty() {
            writeln!(self.buf, "def {}(self, {}):", name, params.join(", "))?;
        } else {
            writeln!(self.buf, "def {}(self):", name)?;
        }
        Ok(())
    }

    fn generate_init_fn(&mut self, init_fn: &TyFnDecl) -> Result<(), Diag> {
        self.generate_fn_decl_head("__init__", init_fn)?;
        self.tab();
        let inits = self.inits.clone();
        for init in inits.borrow().iter() {
            self.indent()?;
            write!(self.buf, "self.{} = ", init.name)?;

            let module_name = self.tenv.borrow()
                .resolve_type(
                    &ModName::Global,
                    &Alias::Variable(init.mod_name.as_str().to_owned())
                )
                .unwrap()
                .as_string();
            let core_cloned = self.core.clone();
            let core = core_cloned.borrow();
            let op = core.find_mod(&module_name).unwrap();
            write!(self.buf, "{}", op.gen_fn_app(&init.fn_name, init.fn_args.as_slice())?)?;
            writeln!(self.buf, "")?;
        }
        self.indent()?;
        writeln!(self.buf, "# TODO: generate `fn new()` body")?;
        //...
        self.shift_tab();
        Ok(())
    }
}

pub struct Generator {
    pub emitter: Rc<RefCell<Emitter>>,
    pub tenv: Rc<RefCell<TypeEnv>>,
    pub buf: String,
    pub imports: BTreeSet<(String, String)>,
    pub modules: BTreeMap<String, Module>,
    core: Rc<RefCell<Core>>,
    pub indent: usize,
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
            indent: 0,
        }
    }

    pub fn generate(&mut self, term: &TyTerm) -> Result<(), Diag> {
        self.collect(term)?;
        self.generate_imports()?;
        self.generate_modules()?;
        Ok(())
    }

    fn generate_modules(&mut self) -> Result<(), Diag> {
        for (_name, module) in self.modules.iter_mut() {
            writeln!(self.buf, "")?;
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
            // _ => unimplemented!(),
            _ => (),
        }
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

impl From<::std::fmt::Error> for Diag {
    fn from(error: ::std::fmt::Error) -> Diag {
        Diag::UnknownError
    }
}