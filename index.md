---
layout: post
title: TensorScript
---
TensorScript is a high-level language for specifying finite-dimensioned tensor computation.

# Example: XOR network

```rust
use lin::Linear;
use nonlin::{sigmoid, relu};

node Xor<[?,2] -> [?,1]> { }
weights Xor<[?,2] -> [?,1]> {
    fc1 = Linear::new(in=2, out=3);
    fc2 = Linear::<[?,3]->[?,1]>::new(in=3, out=1);
}
graph Xor<[?,2] -> [?,1]> {
    def forward {
        x |> fc1 |> sigmoid
        |> fc2
    }
}
```

# Example: GAN

```rust
use lin::Linear;
use reg::{BatchNorm1d};
use nonlin::{leaky_relu, tanh, sigmoid};

dim noise_dim = 100;
dim image_dim = 28;
dim flattened_image_dim = 784;
tsr noise = [?, noise_dim];
tsr flattened_image = [?, flattened_image_dim];
tsr image = [?, 1, image_dim, image_dim];

node Generator<noise -> image> {}
weights Generator<noise -> image> {
    lin1 = Linear::new(in=noise_dim, out=128);
    lin2 = Linear::new(in=128, out=256);
    bn1 = BatchNorm1d::new(num_features=256);
    lin3 = Linear::new(in=256, out=512);
    bn2 = BatchNorm1d::new(num_features=512);
    lin4 = Linear::new(in=512, out=1024);
    bn3 = BatchNorm1d::new(num_features=1024);
    lin5 = Linear::new(in=1024, out=flattened_image_dim);
}
graph Generator<noise -> image> {
    def new() -> Self {
        self
    }
    def forward {
        x
        |> lin1 |> leaky_relu(p=0.2)
        |> lin2 |> bn1 |> leaky_relu(p=0.2)
        |> lin3 |> bn2 |> leaky_relu(p=0.2)
        |> lin4 |> bn3 |> leaky_relu(p=0.2)
        |> lin5 |> tanh
        |> view(_, 1, image_dim, image_dim)
    }
}

node Discriminator<image -> [?, 1]> {}
weights Discriminator<image -> [?,1]> {
    lin1 = Linear::new(in=flattened_image_dim, out=512);
    lin2 = Linear::new(in=512, out=256);
    lin3 = Linear::new(in=256, out=1);
}
graph Discriminator<image -> [?,1]> {
    def new() -> Self {
        self
    }
    def forward {
        x |> view(?, flattened_image_dim)
        |> lin1 |> leaky_relu(p=0.2)
        |> lin2 |> leaky_relu(p=0.2)
        |> lin3 |> sigmoid
    }
}
```

# Example: MNIST

```rust
use conv::{Conv2d, maxpool2d};
use reg::Dropout2d;
use nonlin::{relu, log_softmax};
use lin::Linear;

node Mnist<[?, IMAGE] -> LABELS> {
    // this is where you declare type level constants
    dim FC1 = 320;
    dim FC2 = 50;

    // Prediction
    dim OUT = 10;
    // Channel
    dim C = 1;

    dim W = 28;                 // Image Width
    dim H = 28;                 // Image Height
    tsr IMAGE = [C,H,W];        // Tensor alias

    tsr LABELS = [?,OUT];
}

weights Mnist<[?, IMAGE] -> LABELS> {
    conv1 = Conv2d::new(in_ch=1, out_ch=10, kernel_size=(5,5));
    conv2 = Conv2d::new(in_ch=10, out_ch=20, kernel_size=5);
    dropout = Dropout2d::new(p=0.5);
    fc1 = Linear::<[?,FC1] -> [?,FC2]>::new(in=FC1, out=FC2);
    fc2 = Linear::<[?,FC2] -> [?,OUT]>::new(in=FC2, out=OUT);
}

graph Mnist<[?, IMAGE] -> LABELS> {

    def new() -> Self {
        fc1.init_normal(std=1.);
        fc2.init_normal(std=1.);
        self
    }

    def forward {
        x
        |> conv1            |> maxpool2d(kernel_size=2) |> relu
        |> conv2 |> dropout |> maxpool2d(kernel_size=2) |> relu
        |> view(_, FC1)
        |> fc1 |> relu
        |> self.example()
        |> log_softmax(dim=1)
    }

    def example(x: [?,FC2]) -> LABELS {
        x |> fc2 |> relu
    }

}
```
