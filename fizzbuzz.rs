use nn::conv::{Conv2d, Dropout2d, maxpool2d};
use nn::linear::Linear;

// here are weights that need to be allocated(on CPU or GPU)
weights Mnist {
    conv1: Conv2d::new::<?,c,hi,wi -> ?,c,ho,wo>(1, 10, kernel_size=5),
    conv2: Conv2d::new::<?,c,hi,wi -> ?,c,ho,wo>(10, 20, kernel_size=5),
    dropout: Dropout2d::new::<?,c,h,w -> ?,c,h,w>(p=0.5),
    fc1: Linear::new::<?,320 -> ?,50>(),
    fc2: Linear::new::<?,50 -> ?,10>(),
}

ops Mnist::<?,c,h,w -> ?,10> {
    op new() {
        Mnist::weights()
    }

    op forward(x) {
        x
        |> conv1            |> maxpool2d(kernel_size=2)
        |> conv2 |> dropout |> maxpool2d(kernel_size=2)
        |> view(?, 320)
        |> fc1 |> relu
        |> fc2 |> relu
        |> log_softmax(dim=1)
    }
}