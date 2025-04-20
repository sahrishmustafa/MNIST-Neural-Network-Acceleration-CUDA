# MNIST-Neural-Network-Acceleration-CUDA
 
This repository contains multiple implementations of a simple Neural Network (NN) trained on the [MNIST](http://yann.lecun.com/exdb/mnist/) handwritten digits dataset. The purpose is to explore different parallel computing models and optimizations, from serial CPU execution to advanced GPU and OpenACC acceleration.

---

## Repository Structure

```bash
.
├── data/               # Contains MNIST dataset files
│   ├── t10k-images.idx3-ubyte
│   ├── t10k-labels.idx1-ubyte
│   ├── train-images.idx3-ubyte
│   └── train-labels.idx1-ubyte
│
├── src/
│   ├── V1/             # Serial CPU implementation
│   │   ├── nn.c
│   │   └── Makefile
│   ├── V2/             # Naive GPU (CUDA) implementation
│   │   ├── nn.cu
│   │   └── Makefile
│   ├── V3/             # Optimized GPU (CUDA) implementation
│   │   ├── nn.cu
│   │   └── Makefile
│   ├── V4/             # Tensor Core + Batch Processing (CUDA)
│   │   ├── nn.cu
│   │   └── Makefile
│   └── V5/             # OpenACC implementation
│       ├── nn.cpp
│       └── Makefile
│
└── README.md
