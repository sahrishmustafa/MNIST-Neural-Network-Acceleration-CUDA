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

```
## Project Objective

To evaluate and benchmark different parallel programming models for neural network training on the MNIST dataset using:

- **Serial execution (CPU)**
- **CUDA-based GPU acceleration (naive and optimized)**
- **Tensor Core acceleration with batching**
- **OpenACC-based GPU parallelism**

---

## Prerequisites

- **Linux/Unix-based OS**
- **CUDA Toolkit** (for V2, V3, V4)
- **NVIDIA GPU** (Tensor Cores recommended for V4)
- **OpenACC compiler** (e.g., PGI/NVIDIA HPC compiler) for V5
- GCC for serial version
- Make utility

---

## How to Build and Run

### Step 1: Prepare the Dataset

Place the following MNIST files inside the `/data` directory:

- `train-images.idx3-ubyte`
- `train-labels.idx1-ubyte`
- `t10k-images.idx3-ubyte`
- `t10k-labels.idx1-ubyte`
---

### Step 2: Navigate to Any Version

For example, to run the **serial implementation**:

```bash
cd src/V1
make
./run
