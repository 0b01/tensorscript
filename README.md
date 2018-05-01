# TensorScript

Dependently-typed tensor computation.

## Features

* Static type inference
* Compile time type checking
* Dependently typed tensors
* Multiple targets(Tensorflow, PyTorch, more to come!)
* Pipes operator

### Pipes operator

Pipes operator is a syntax sugar for chained function calls inspired by F#, Elixir and R.
For example,

```rust
x |> lin1 |> leaky_relu(p=0.2) |> sigmoid
```

compiles to

```python
x = lin1(x)
x = leaky_relu(x, p=0.2)
x = sigmoid(x)
```

## Development

[![Build Status](https://travis-ci.org/rickyhan/tensorscript-compiler.svg?branch=master)](https://travis-ci.org/rickyhan/tensorscript-compiler)

The language is currently being actively worked on, and not useable in production or development.

### Todo

1. [*] implement module pattern matching
2. [*] type level computation (resolved tensor dimension)
3. [*] BUG: dimension mismatch for mnist example
            need to create fresh type variables for different static forward functions
4. [*] BUG: non-determinism
5. [*] BUG: impl Hash, Eq for Type
6. [*] set up examples and tests
7. [*] set up commandline
8. [*] more examples
9. [*] better errors in parser
10. [ ] code gen: PyTorch
11. [ ] add more examples
12. [*] lift dim and tsr to top level
13. [ ] add dim level computation dim1 * dim1
14. [ ] use Linear as L; aliasing
15. [ ] add binary ops (+, -, *, /, %)
16. [ ] add if else expression
17. [ ] add let binding
18. [ ] add more tests
