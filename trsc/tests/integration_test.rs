extern crate assert_cli;

#[test]
fn test_no_input() {
    assert_cli::Assert::main_binary()
        .fails()
        .unwrap();
}

#[test]
fn test_xor() {
    assert_cli::Assert::main_binary()
        .with_args(&["--in", "tests/input/xor.trs"])
        .succeeds()
        .and()
        .stdout().is(include_str!("output/xor.py"))
        .unwrap();
}

#[test]
fn test_mnist() {
    assert_cli::Assert::main_binary()
        .with_args(&["--in", "tests/input/mnist.trs"])
        .succeeds()
        .and()
        .stdout().is(include_str!("output/mnist.py"))
        .unwrap();
}

#[test]
fn test_gan() {
    assert_cli::Assert::main_binary()
        .with_args(&["--in", "tests/input/gan.trs"])
        .succeeds()
        .and()
        .stdout().is(include_str!("output/gan.py"))
        .unwrap();
}