---
layout: post
title:  "Tensorflow FizzBuzz Revisited"
date:   2018-02-16 00:00:00 -0400
categories: jekyll update
---

**interviewer**: Good morning. You file indicates an application from two years ago but didn't pass the coding interview.

**me**: Yes. I had some trouble with FizzBuzz.

**interviewer**: FizzBuzz? That seems too easy for your credentials.

**me**: Unfortunately, the whiteboard didn't have a GPU.

**interviewer**: Well. Don't stress about it. Many of our top performers interviewed multiple times.

**me**: I am confident, now that I have two more years of experience.

**interviewer**: I like your spirit. Let's get started. Since you mentioned it, do a FizzBuzz in the language of your choice.

**me**: Can you repeat the requirement just so we are on the same page?

**interviewer**: Sure. Print from 1 to 100, except when n is divisible by 3 print "fizz", by 5 print "buzz", and if it's divisible by 15 print "fizzbuzz".

**me**: Very well.

*ruminates and writes down*

```python
length = 100
arr = [''] * (length-1)
i = 1
while i < length:
    if n % 3 == 0 and n % 5 == 0:
        arr[n-1] = 'FizzBuzz'
    elif n % 3 == 0:
        arr[n-1] = 'Fizz'
    elif n % 5 == 0:
        arr[n-1] = 'Buzz'
    else:
        arr
```

**interviewer**: Your code is almost correct. Do you see what's wrong?

**me**: I know what you are thinking. This is only one piece of the puzzle.

**interviewer**: *\*confused\** Could you explain your strategy?

**me**: I am going to run it in TensorFlow. Now I just need to transform the abstract syntax tree.

**interviewer**: You got it but you are overthinking it. Well, we can move on to systems design questions.

**me**: It doesn't very long. Let me explain:

Tensorflow specializes in Machine Learning but its internal graph data structure is suitable for general dataflow programming. Now my subgoal is: write some AST transforms to transpile regular python into TensorFlow function calls. For example, python loops into `tf.while_loop`.

```python
import astunparse, ast, astpretty
from ast import *

fname = "./raw_fizzbuzz.py"
with open(fname) as f:
    txt = f.read()
myast = ast.parse(txt)

# transform

print(astunparse.unparse(myast))
```

Python has a built-in package called [ast](https://docs.python.org/2/library/ast.html). The `NodeTransformer` provides handy tree modifications. Allow me to demonstrate.

```python
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
        # omitted ...
        else:
            return node

myast = RewriteName().visit(myast)
```

This function visits all `BoolOp` nodes and replaces `==` operator with corresponding `tf.logical_and` function call.

I won't include all the transforms. If anyone at your company has to write a lot of tensorflow-ism(which is rare), I have posted it [here](https://gist.github.com/rickyhan/eea717ee5de492e52b84d3bea357e40e).

The result(after some manual cleanup):

```python
import tensorflow as tf
length = 100
arr = tf.Variable([str(i) for i in range(1, length+1)])
graph = tf.while_loop(
    (lambda i, _: tf.less(i, length+1)), 
    (lambda i, _: (tf.add(i,1), tf.cond(
        tf.logical_and(tf.equal(tf.mod(i, 3), 0), tf.equal(tf.mod(i, 5), 0)),
        (lambda : tf.assign(arr[(i - 1)], 'FizzBuzz')),
        (lambda : tf.cond(tf.equal(tf.mod(i, 3), 0),
            (lambda : tf.assign(arr[(i - 1)], 'Fizz')),
            (lambda : tf.cond(tf.equal(tf.mod(i, 5), 0),
                (lambda : tf.assign(arr[(i - 1)], 'Buzz')),
                (lambda : arr)))))))),
    [1, arr])
with tf.Session() as sess:
    tf.global_variables_initializer().run()
    idx, array = sess.run(graph)
    print array
```

Since TensorFlow has `tf.cond` and `tf.while_loop`, it is Turing complete like a lot of programming languages. FizzBuzz, quicksort, Dijkstra's algorithm all can be implemented in Tensorflow.

And here is the result:

```
g@g:~/Desktop/py2tf⟫ python test.py 
2018-02-17 00:31:12.564103: W tensorflow/core/platform/cpu_feature_guard.cc:45] The TensorFlow library wasn't compiled to use SSE4.1 instructions, but these are available on your machine and could speed up CPU computations.
2018-02-17 00:31:12.564124: W tensorflow/core/platform/cpu_feature_guard.cc:45] The TensorFlow library wasn't compiled to use SSE4.2 instructions, but these are available on your machine and could speed up CPU computations.
2018-02-17 00:31:12.564128: W tensorflow/core/platform/cpu_feature_guard.cc:45] The TensorFlow library wasn't compiled to use AVX instructions, but these are available on your machine and could speed up CPU computations.
2018-02-17 00:31:12.564132: W tensorflow/core/platform/cpu_feature_guard.cc:45] The TensorFlow library wasn't compiled to use AVX2 instructions, but these are available on your machine and could speed up CPU computations.
2018-02-17 00:31:12.564135: W tensorflow/core/platform/cpu_feature_guard.cc:45] The TensorFlow library wasn't compiled to use FMA instructions, but these are available on your machine and could speed up CPU computations.
['1' '2' 'Fizz' '4' 'Buzz' 'Fizz' '7' '8' 'Fizz' 'Buzz' '11' 'Fizz' '13'
 '14' 'FizzBuzz' '16' '17' 'Fizz' '19' 'Buzz' 'Fizz' '22' '23' 'Fizz'
 'Buzz' '26' 'Fizz' '28' '29' 'FizzBuzz' '31' '32' 'Fizz' '34' 'Buzz'
 'Fizz' '37' '38' 'Fizz' 'Buzz' '41' 'Fizz' '43' '44' 'FizzBuzz' '46' '47'
 'Fizz' '49' 'Buzz' 'Fizz' '52' '53' 'Fizz' 'Buzz' '56' 'Fizz' '58' '59'
 'FizzBuzz' '61' '62' 'Fizz' '64' 'Buzz' 'Fizz' '67' '68' 'Fizz' 'Buzz'
 '71' 'Fizz' '73' '74' 'FizzBuzz' '76' '77' 'Fizz' '79' 'Buzz' 'Fizz' '82'
 '83' 'Fizz' 'Buzz' '86' 'Fizz' '88' '89' 'FizzBuzz' '91' '92' 'Fizz' '94'
 'Buzz' 'Fizz' '97' '98' 'Fizz' 'Buzz']
```

**interviewer**: Don't apply again.

# Conclusion

On a more serious note, I have been thinking about a framework-agnostic DSL that transpiles to Tensorflow and PyTorch with strong typing, dimension and ownership checking and lots of syntax sugars. This is borne out of frustration: I am so tired of deciphering undocumented code. I get it - people are lazy and unsafe unless the compiler forces them to annotate. At most 4 dimensions! However, this will be an undertaking if I decide to do it.

```rust
use conv::{Conv2d, Dropout2d, maxpool2d};
use linear::Linear;
use loss::log_softmax;
use relu;

// here are weights that need to be allocated(on CPU or GPU)

declare Mnist<?,c,h,w -> ?,10>;

weights Mnist {
    conv1: Conv2d<?,c,hi,wi -> ?,c,ho,wo>::new(in_ch=1, out_ch=10, kernel_size=5),
    conv2: Conv2d<?,c,hi,wi -> ?,c,ho,wo>::new(in_ch=10, out_ch=20, kernel_size=5),
    dropout: Dropout2d<?,c,h,w -> ?,c,h,w>::new(p=0.5),
    fc1: Linear<?,320 -> ?,50>::new(),
    fc2: Linear<?,50 -> ?,10>::new(),
}

ops Mnist {

    // runs automatically
    op new() {
        Self::weights()
        conv1.init_normal();
        conv2.init_normal();
    }

    op forward(x) {
        x
        |> conv1            |> maxpool2d(kernel_size=2)
        |> conv2 |> dropout |> maxpool2d(kernel_size=2)
        |> view(?, 320)
        |> fc1 |> relu
        |> self.fc2
        |> log_softmax(dim=1)
    }

    // impure function, needs annotation
    op fc2(self, x: <?,50>) -> <?,10>{
        x |> fc2 |> relu
    }
}

```