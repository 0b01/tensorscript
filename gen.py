import astunparse, ast, astpretty
from ast import *

fname = "./raw_fizzbuzz.py"
with open(fname) as f:
    txt = f.read()

class RewriteName(NodeTransformer):
    def visit_BoolOp(self, node):
        # print astpretty.pprint(node)
        if isinstance(node.op, And):
            funcname = "tf.logical_and"
            return copy_location(Call(
                func=Name(id=funcname, ctx=Load()),
                args=(
                    self.visit(node.values[0]),
                    self.visit(node.values[1])
                    ),
                keywords=(),
                starargs=(),
                kwargs=(),
            ), node)
        else:
            return node

    def visit_BinOp(self, node):
        # print astpretty.pprint(node)
        if isinstance(node.op, Mod):
            return copy_location(Call(
                func=Name(id="tf.mod", ctx=Load()),
                args=(node.left, node.right),
                keywords=(),
                starargs=(),
                kwargs=(),
            ), node)
        return node

    def visit_Compare(self, node):
        # print astpretty.pprint(node)
        if len(node.ops) == 1 and isinstance(node.ops[0], Eq):
            return copy_location(Call(
                func=Name(id="tf.equal", ctx=Load()),
                args=(self.visit(node.left), map(lambda i: self.visit(i), node.comparators)),
                keywords=(),
                starargs=(),
                kwargs=(),
            ), node)
        elif len(node.ops) == 1 and isinstance(node.ops[0], Lt):
            return copy_location(Call(
                func=Name(id="tf.less", ctx=Load()),
                args=(self.visit(node.left), map(lambda i: self.visit(i), node.comparators)),
                keywords=(),
                starargs=(),
                kwargs=(),
            ), node)
        else:
            return node

    def visit_If(self, node):
        # print astpretty.pprint(node)
        return copy_location(Call(
            func=Name(id="tf.cond", ctx=Load()),
            args=(
                self.visit(node.test), 
                Lambda(
                    args=arguments(args=[],defaults=[],vararg=[],kwarg=[]),
                    body=self.visit(node.body[0])
                    ),
                Lambda(
                    args=arguments(args=[],defaults=[],vararg=[],kwarg=[]),
                    body=self.visit(node.orelse[0])
                    ),
                ),
            keywords=(),
            starargs=(),
            kwargs=(),
        ), node)
    
    def visit_Assign(self, node):
        # print astpretty.pprint(node)
        return copy_location(Call(
            func=Name(id="tf.assign", ctx=Load()),
            args=(
                self.visit(node.targets[0]), 
                self.visit(node.value),
            ),
            keywords=(),
            starargs=(),
            kwargs=(),
        ), node)
    
    def visit_While(self, node):
        # print astpretty.pprint(node)
        return copy_location(Call(
            func=Name(id="tf.while_loop", ctx=Load()),
            args=(
                Lambda(
                    args=arguments(args=[],defaults=[],vararg=[],kwarg=[]),
                    body=self.visit(node.test), 
                    ),
                map(self.visit, node.body),
            ),
            keywords=(),
            starargs=(),
            kwargs=(),
        ), node)


myast = ast.parse(txt)
myast = RewriteName().visit(myast)
# print astpretty.pprint(myast)
print(astunparse.unparse(myast))
