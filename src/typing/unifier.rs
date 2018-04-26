use typing::{Type, TypeEnv};
use typing::type_env::TypeId;
use span::CSpan;
use errors::{Emitter, TensorScriptDiagnostic, EmitErr};
use std::rc::Rc;
use std::cell::RefCell;

use typing::constraint::{Constraints, Equals};
use typing::subst::Substitution;

pub struct Unifier {
    pub emitter: Rc<RefCell<Emitter>>,
}

impl EmitErr for Unifier {
    fn emit_err(&self) {
        self.emitter.borrow().print_errs();
    }
}

impl Unifier {

    pub fn new(emitter: Rc<RefCell<Emitter>>) -> Unifier {
        Unifier {
            emitter,
        }
    }

    pub fn unify(&mut self, cs: Constraints, tenv: &mut TypeEnv) -> Substitution {
        if cs.is_empty() {
            Substitution::empty()
        } else {
            let emitter = cs.emitter.clone();
            let mut it = cs.set.into_iter();
            let mut subst = self.unify_one(it.next().unwrap(), tenv, emitter.clone());
            let subst_tail = subst.apply(&Constraints {set: it.collect(), emitter});
            let subst_tail: Substitution = self.unify(subst_tail, tenv);
            subst.compose(subst_tail)
        }
    }

    fn unify_one(&mut self, eq: Equals, tenv: &mut TypeEnv, emitter: Rc<RefCell<Emitter>>) -> Substitution {
        use self::Type::*;
        // println!("{:?}", eq);
        match eq {
            Equals(Unit(_), Unit(_)) => Substitution::empty(),
            Equals(INT(_), INT(_)) => Substitution::empty(),
            Equals(FLOAT(_), FLOAT(_)) => Substitution::empty(),
            Equals(BOOL(_), BOOL(_)) => Substitution::empty(),

            Equals(INT(_), ResolvedDim(_, _)) => Substitution::empty(),
            Equals(ResolvedDim(_, _), INT(_)) => Substitution::empty(),

            Equals(a @ ResolvedDim(_, _), b @ ResolvedDim(_, _)) => {
                if a.as_num() == b.as_num() {
                    Substitution::empty()
                } else {
                    self.add_err(TensorScriptDiagnostic::DimensionMismatch(a.clone(), b.clone()));
                    Substitution::empty()
                }
            }

            Equals(VAR(tvar, _), ty) => self.unify_var(tvar, ty),
            Equals(ty, VAR(tvar, _)) => self.unify_var(tvar, ty),

            Equals(DIM(tvar, _), ty) => self.unify_var(tvar, ty),
            Equals(ty, DIM(tvar, _)) => self.unify_var(tvar, ty),

            Equals(FnArgs(v1, _), FnArgs(v2, _)) => self.unify(
                Constraints {
                    set: v1.into_iter().zip(v2).map(|(i, j)| Equals(i, j)).collect(),
                    emitter,
                },
                tenv,
            ),

            Equals(FnArg(Some(a), ty1, _), FnArg(Some(b), ty2, _)) => {
                if a == b {
                    self.unify(
                        Constraints {
                            set: btreeset!{ Equals(*ty1, *ty2)},
                            emitter,
                        },
                        tenv)
                } else {
                    panic!("supplied parameter is incorrect! {} != {}", a, b);
                }
            }

            Equals(FUN(m1,n1,p1, r1, _), FUN(m2,n2,p2, r2, _)) => {
                if n1 == n2 {
                    self.unify(
                        Constraints{
                            set: btreeset!{
                                Equals(*p1, *p2),
                                Equals(*r1, *r2),
                            },
                            emitter,
                        },
                        tenv,
                    )
                } else {
                    println!("{} {} {} {}", m1, m2, n1, n2);
                    panic!()
                }
            },

            Equals(Tuple(vs1, _), Tuple(vs2, _)) => self.unify(
                Constraints {
                    set: vs1.into_iter().zip(vs2).map(|(i,j)| Equals(i,j)).collect(),
                    emitter,
                },
                tenv
            ),

            Equals(ts1 @ TSR(_, _), ts2 @ TSR(_, _)) => {
                if ts1.as_rank() == ts2.as_rank() {
                    match (ts1, ts2) {
                        (TSR(dims1, s1), TSR(dims2, s2)) => self.unify(
                            Constraints {
                                set: dims1
                                    .into_iter()
                                    .zip(dims2)
                                    .map(|(i, j)| Equals(i.with_span(&s1), j.with_span(&s2)))
                                    .collect(),
                                emitter
                            },
                            tenv,
                        ),
                        _ => unimplemented!(),
                    }
                } else {
                    self.add_err(TensorScriptDiagnostic::RankMismatch(ts1, ts2));
                    Substitution::empty()
                }
            }

            Equals(Module(n1, Some(box ty1), _), Module(n2, Some(box ty2), _)) => self.unify(
                Constraints {
                    set: btreeset!{
                        if n1 == n2 {
                            Equals(ty1, ty2)
                        } else {
                            panic!();
                        }
                    },
                    emitter,
                },
                tenv,
            ),

            Equals(u @ UnresolvedModuleFun(_, _, _, _), ty) => {
                Substitution(btreemap!(
                    u => ty,
                ))
            }

            _ => {
                panic!("{:#?}", eq);
            }
        }
    }

    fn unify_var(&mut self, tvar: TypeId, ty: Type) -> Substitution {
        use self::Type::*;

        let span = CSpan::fresh_span();
        match ty.clone() {
            VAR(tvar2, _) => {
                if tvar == tvar2 {
                    Substitution::empty()
                } else {
                    Substitution(btreemap!{ VAR(tvar, span) => ty })
                }
            }
            DIM(tvar2, _) => {
                if tvar == tvar2 {
                    Substitution::empty()
                } else {
                    Substitution(btreemap!{ VAR(tvar, span) => ty })
                }
            }
            _ => if occurs(tvar, &ty) {
                panic!("circular type")
            } else {
                Substitution(btreemap!{ VAR(tvar, span) => ty })
            },
        }
    }


    pub fn add_err(&mut self, err: TensorScriptDiagnostic) {
        self.emitter.borrow_mut().add(err);
    }
}

fn occurs(tvar: TypeId, ty: &Type) -> bool {
    use self::Type::*;
    match ty {
        FUN(_,_, ref p, ref r, _) => occurs(tvar, &p) | occurs(tvar, &r),
        VAR(ref tvar2, _) => tvar == *tvar2,
        _ => false,
    }
}
