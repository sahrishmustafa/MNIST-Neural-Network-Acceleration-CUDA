#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cuda_fp16.h>
#include <cublas_v2.h>
#include <cublasLt.h>

#define INPUT_SIZE    784
#define HIDDEN_SIZE   128
#define OUTPUT_SIZE   10
#define LEARNING_RATE 0.01
#define EPOCHS        3
#define BATCH_SIZE    64
#define NUM_CLASSES   10  // Digits 0-9
#define NUM_TRAIN     60000
#define NUM_TEST      10000

// Timer function
double get_time(clock_t start) {
    return (double)(clock() - start) / CLOCKS_PER_SEC;
}

// Allocate memory for a 2D matrix (host side)
double** allocateMatrix(int rows, int cols) {
    double** mat = (double**)malloc(rows * sizeof(double*));
    for (int i = 0; i < rows; i++) {
        mat[i] = (double*)malloc(cols * sizeof(double));
    }
    return mat;
}

// Free allocated matrix memory (host side)
void freeMatrix(double** mat, int rows) {
    for (int i = 0; i < rows; i++) {
        free(mat[i]);
    }
    free(mat);
}

// Activation functions on host
void relu(double* x, int size) {
    for (int i = 0; i < size; i++) {
        x[i] = (x[i] > 0) ? x[i] : 0;
    }
}

void softmax(float* x, int size) {
    float max_val = x[0];
    for(int i=1; i<size; i++) if(x[i]>max_val) max_val=x[i];
    
    float sum = 0.0f;
    for(int i=0; i<size; i++) {
        x[i] = expf(x[i]-max_val);
        sum += x[i];
    }
    for(int i=0; i<size; i++) x[i] /= sum;
}

bool is_half_nan(__half val) {
    uint16_t x = *((uint16_t*)&val);
    return ((x & 0x7FFF) > 0x7C00);
}

bool is_half_inf(__half val) {
    uint16_t x = *((uint16_t*)&val);
    return ((x & 0x7FFF) == 0x7C00);
}

bool is_half_valid(__half val) {
    uint16_t x = *((uint16_t*)&val);
    return ((x & 0x7C00) != 0x7C00);  // Not NaN or Inf
}
// -----------------------
// CUDA Kernel Definitions
// -----------------------

// Each thread computes one neuron in the hidden layer.
// Suggested kernel launch: <<<2, 64>>>
__global__ void forwardHidden(double* d_W1, double* d_b1, double* d_input, double* d_hidden) 
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    
    __shared__ double s_input[INPUT_SIZE];
    
    // More efficient loading with 64 threads per block
    // Each thread loads ~12-13 elements instead of ~98
    for (int j = threadIdx.x; j < INPUT_SIZE; j += blockDim.x) {
        s_input[j] = d_input[j];
    }
    
    __syncthreads();
    
    if(i < HIDDEN_SIZE) 
    {
        double sum = d_b1[i];
        for (int j = 0; j < INPUT_SIZE; j++)
        {
            sum += d_W1[i * INPUT_SIZE + j] * s_input[j];
        }
        d_hidden[i] = (sum > 0) ? sum : 0;  // ReLU
    }
}

// Each thread computes one neuron in the output layer.
__global__ void forwardOutput(double* d_W2, double* d_b2, double* d_hidden, double* d_output) 
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Shared memory for hidden layer activations
    __shared__ double s_hidden[HIDDEN_SIZE];
    
    // Collaborative loading of hidden layer values into shared memory
    // With OUTPUT_SIZE threads (typically 10 for MNIST), each thread loads multiple elements
    for (int j = threadIdx.x; j < HIDDEN_SIZE; j += blockDim.x) {
        s_hidden[j] = d_hidden[j];
    }
    
    // Wait for all threads to finish loading
    __syncthreads();
    
    if(i < OUTPUT_SIZE) {
        double sum = d_b2[i];
        for (int j = 0; j < HIDDEN_SIZE; j++){
            sum += d_W2[i * HIDDEN_SIZE + j] * s_hidden[j];
        }
        d_output[i] = sum;  // Softmax will be applied on host later
    }
}
// Compute hidden gradient: d_dHidden = (W2^T * d_dOutput) * (hidden > 0)
// (This kernel remains unchanged.)
__global__ void computeHiddenGradients(double* d_W2, double* d_dOutput, double* d_hidden, double* d_dHidden) 
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Create shared memory for d_dOutput (accessed by all threads)
    __shared__ double s_dOutput[OUTPUT_SIZE];
    
    // Collaboratively load d_dOutput into shared memory
    for (int j = threadIdx.x; j < OUTPUT_SIZE; j += blockDim.x) {
        s_dOutput[j] = d_dOutput[j];
    }
    
    // Ensure all threads have loaded the data
    __syncthreads();
    
    if(i < HIDDEN_SIZE) 
    {
        double sum = 0.0;
        for (int j = 0; j < OUTPUT_SIZE; j++)
        {
            // Use shared memory instead of global memory
            sum += d_W2[j * HIDDEN_SIZE + i] * s_dOutput[j];
        }
        d_dHidden[i] = (d_hidden[i] > 0) ? sum : 0.0;
    }
}

// Update output layer weights and biases
// Launch with OUTPUT_SIZE blocks and HIDDEN_SIZE threads per block.
__global__ void updateOutputLayer(double* d_W2, double* d_b2, double* d_dOutput, double* d_hidden) 
{
    int linear_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int i = linear_idx / HIDDEN_SIZE; // output neuron
    int j = linear_idx % HIDDEN_SIZE; // hidden neuron

    if(i < OUTPUT_SIZE && j < HIDDEN_SIZE) {
    d_W2[i * HIDDEN_SIZE + j] -= LEARNING_RATE * d_dOutput[i] * d_hidden[j];
    }

    // Bias updates (single thread per output neuron)
    if(linear_idx < OUTPUT_SIZE) {
    d_b2[linear_idx] -= LEARNING_RATE * d_dOutput[linear_idx];
    }
}

// Update hidden layer weights and biases
// Launch with HIDDEN_SIZE blocks and INPUT_SIZE threads per block.
__global__ void updateHiddenLayer(double* d_W1, double* d_b1, double* d_dHidden, double* d_input) 
{
    int i = blockIdx.x;  // hidden neuron index
    int j = threadIdx.x; // input neuron index
    if(i < HIDDEN_SIZE && j < INPUT_SIZE) {
        d_W1[i * INPUT_SIZE + j] -= LEARNING_RATE * d_dHidden[i] * d_input[j];
    }
    if(j == 0 && i < HIDDEN_SIZE) {
        d_b1[i] -= LEARNING_RATE * d_dHidden[i];
    }
}

// -----------------------
// Neural Network Structure (Parameters reside on device)
// -----------------------
typedef struct 
{
    // Device pointers for parameters (flattened arrays)
    __half* d_W1; // [HIDDEN_SIZE x INPUT_SIZE]
    __half* d_W2; // [OUTPUT_SIZE x HIDDEN_SIZE]
    __half* d_b1; // [HIDDEN_SIZE]
    __half* d_b2; // [OUTPUT_SIZE]
    cublasHandle_t cublas_handle;
} NeuralNetwork;

// -----------------------
// Create and Initialize Network (Parameters on Device)
// -----------------------
NeuralNetwork* createNetwork() 
{
    printf("Creating neural network...\n");
    NeuralNetwork* net = (NeuralNetwork*)malloc(sizeof(NeuralNetwork));
    cublasStatus_t status = cublasCreate(&net->cublas_handle);
    if (status != CUBLAS_STATUS_SUCCESS) {
        printf("CUBLAS initialization failed\n");
        exit(1);
    }
    
    // Allocate device memory in FP16
    printf("Allocating device memory...\n");
    cudaError_t err;
    err = cudaMalloc((void**)&net->d_W1, HIDDEN_SIZE * INPUT_SIZE * sizeof(__half));
    if (err != cudaSuccess) printf("cudaMalloc W1 failed: %s\n", cudaGetErrorString(err));
    err = cudaMalloc((void**)&net->d_W2, OUTPUT_SIZE * HIDDEN_SIZE * sizeof(__half));
    if (err != cudaSuccess) printf("cudaMalloc W2 failed: %s\n", cudaGetErrorString(err));
    err = cudaMalloc((void**)&net->d_b1, HIDDEN_SIZE * sizeof(__half));
    if (err != cudaSuccess) printf("cudaMalloc b1 failed: %s\n", cudaGetErrorString(err));
    err = cudaMalloc((void**)&net->d_b2, OUTPUT_SIZE * sizeof(__half));
    if (err != cudaSuccess) printf("cudaMalloc b2 failed: %s\n", cudaGetErrorString(err));

    // Temporary host memory in float to fill and convert
    printf("Initializing weights...\n");
    float* h_W1_f = (float*)malloc(HIDDEN_SIZE * INPUT_SIZE * sizeof(float));
    float* h_W2_f = (float*)malloc(OUTPUT_SIZE * HIDDEN_SIZE * sizeof(float));
    float* h_b1_f = (float*)calloc(HIDDEN_SIZE, sizeof(float));
    float* h_b2_f = (float*)calloc(OUTPUT_SIZE, sizeof(float));

    // Fill float arrays
    srand(time(NULL));
    for (int i = 0; i < HIDDEN_SIZE * INPUT_SIZE; i++)
        h_W1_f[i] = ((float)rand() / RAND_MAX) * 0.01f;
    for (int i = 0; i < OUTPUT_SIZE * HIDDEN_SIZE; i++)
        h_W2_f[i] = ((float)rand() / RAND_MAX) * 0.01f;

    // Check initial weights
    printf("First few weights:\n");
    for(int i=0; i<5; i++) printf("W1[%d]=%.6f ", i, h_W1_f[i]);
    printf("\n");
    for(int i=0; i<5; i++) printf("W2[%d]=%.6f ", i, h_W2_f[i]);
    printf("\n");

    // Allocate temporary __half arrays
    __half *h_W1_h = (__half*)malloc(HIDDEN_SIZE * INPUT_SIZE * sizeof(__half));
    __half *h_W2_h = (__half*)malloc(OUTPUT_SIZE * HIDDEN_SIZE * sizeof(__half));
    __half *h_b1_h = (__half*)malloc(HIDDEN_SIZE * sizeof(__half));
    __half *h_b2_h = (__half*)malloc(OUTPUT_SIZE * sizeof(__half));

    // Convert float -> half (on host)
    // In createNetwork(), modify weight initialization:
    for (int i = 0; i < HIDDEN_SIZE * INPUT_SIZE; i++)
        h_W1_f[i] = ((float)rand() / RAND_MAX - 0.5f) * sqrtf(2.0f/INPUT_SIZE);
    for (int i = 0; i < OUTPUT_SIZE * HIDDEN_SIZE; i++)
        h_W2_f[i] = ((float)rand() / RAND_MAX - 0.5f) * sqrtf(2.0f/HIDDEN_SIZE);
    for (int i = 0; i < HIDDEN_SIZE; i++)
        h_b1_h[i] = __float2half(h_b1_f[i]);
    for (int i = 0; i < OUTPUT_SIZE; i++)
        h_b2_h[i] = __float2half(h_b2_f[i]);

    // Check for NaN/Inf in converted weights
    int nan_count = 0;
    for(int i=0; i<HIDDEN_SIZE*INPUT_SIZE; i++) {
        if(is_half_nan(h_W1_h[i]) || is_half_inf(h_W1_h[i])) nan_count++;
    }
    printf("W1 NaN/Inf count: %d\n", nan_count);
    
    nan_count = 0;
    for(int i=0; i<OUTPUT_SIZE*HIDDEN_SIZE; i++) {
        if(is_half_nan(h_W2_h[i]) || is_half_inf(h_W2_h[i])) nan_count++;
    }
    printf("W2 NaN/Inf count: %d\n", nan_count);

    // Upload to device
    printf("Copying weights to device...\n");
    err = cudaMemcpy(net->d_W1, h_W1_h, HIDDEN_SIZE * INPUT_SIZE * sizeof(__half), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) printf("cudaMemcpy W1 failed: %s\n", cudaGetErrorString(err));
    err = cudaMemcpy(net->d_W2, h_W2_h, OUTPUT_SIZE * HIDDEN_SIZE * sizeof(__half), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) printf("cudaMemcpy W2 failed: %s\n", cudaGetErrorString(err));
    err = cudaMemcpy(net->d_b1, h_b1_h, HIDDEN_SIZE * sizeof(__half), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) printf("cudaMemcpy b1 failed: %s\n", cudaGetErrorString(err));
    err = cudaMemcpy(net->d_b2, h_b2_h, OUTPUT_SIZE * sizeof(__half), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) printf("cudaMemcpy b2 failed: %s\n", cudaGetErrorString(err));

    // Free host temp memory
    free(h_W1_f); free(h_W2_f); free(h_b1_f); free(h_b2_f);
    free(h_W1_h); free(h_W2_h); free(h_b1_h); free(h_b2_h);

    printf("Network created successfully\n");
    return net;
}

/*Supporting Kernels to cublas forward pass*/
////////////////////////////////////////////
__global__ void biasReLU(__half* data, __half* bias, int size) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size) {
        float val = __half2float(data[i]) + __half2float(bias[i]);
        data[i] = __float2half(fmaxf(0.0f, val));
    }
}

__global__ void biasOnly(__half* data, __half* bias, int size) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size) {
        float val = __half2float(data[i]) + __half2float(bias[i]);
        data[i] = __float2half(val);
    }
}

__global__ void __halfToFloatArray(__half* input, float* output, int size) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size)
        output[i] = __half2float(input[i]);
}
////////////////////////////////////////////

// -----------------------
// Modified Forward Pass on GPU
// -----------------------
// The input (d_input) is a pointer to one image (of size INPUT_SIZE) already on the device.
void forwardGPU(NeuralNetwork* net, __half* d_input, float* hidden, float* output, __half* d_hidden_temp, __half* d_output_temp) 
{
    printf("Starting forward pass...\n");
    size_t size_hidden = HIDDEN_SIZE * sizeof(__half);
    size_t size_output = OUTPUT_SIZE * sizeof(__half);

    cublasHandle_t handle = net->cublas_handle;
    // Set math mode to allow Tensor Core operations
    cublasStatus_t status = cublasSetMathMode(handle, CUBLAS_TENSOR_OP_MATH);
    if (status != CUBLAS_STATUS_SUCCESS) {
        printf("Failed to set math mode\n");
    }

    float alpha = 1.0f;
    float beta = 0.0f;

    // In forwardGPU, before GEMM operations:
    printf("Input range check:\n");
    __half h_input[INPUT_SIZE];
    cudaMemcpy(h_input, d_input, INPUT_SIZE*sizeof(__half), cudaMemcpyDeviceToHost);
    for(int i=0; i<min(5,INPUT_SIZE); i++) 
        printf("%.4f ", __half2float(h_input[i]));
    printf("\n");
    // ---------------------------------------
    // Hidden Layer: d_hidden_temp = W1 * d_input
    // ---------------------------------------
    printf("Computing hidden layer...\n");
    status = cublasGemmEx(handle,
        CUBLAS_OP_N, CUBLAS_OP_N,
        HIDDEN_SIZE, 1, INPUT_SIZE,
        &alpha,
        net->d_W1, CUDA_R_16F, HIDDEN_SIZE,
        d_input, CUDA_R_16F, INPUT_SIZE,
        &beta,
        d_hidden_temp, CUDA_R_16F, HIDDEN_SIZE,
        CUBLAS_COMPUTE_32F_FAST_16F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);

    if (status != CUBLAS_STATUS_SUCCESS) {
        printf("Hidden layer GEMM failed: %d\n", status);
    }

    // Add bias and apply ReLU
    printf("Applying ReLU...\n");
    biasReLU<<<(HIDDEN_SIZE + 255) / 256, 256>>>(d_hidden_temp, net->d_b1, HIDDEN_SIZE);
    cudaDeviceSynchronize();

    // Check for CUDA errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("BiasReLU kernel failed: %s\n", cudaGetErrorString(err));
    }

    // ---------------------------------------
    // Output Layer: d_output_temp = W2 * d_hidden_temp
    // ---------------------------------------
    printf("Computing output layer...\n");
    status = cublasGemmEx(handle,
        CUBLAS_OP_N, CUBLAS_OP_N,
        OUTPUT_SIZE, 1, HIDDEN_SIZE,
        &alpha,
        net->d_W2, CUDA_R_16F, OUTPUT_SIZE,
        d_hidden_temp, CUDA_R_16F, HIDDEN_SIZE,
        &beta,
        d_output_temp, CUDA_R_16F, OUTPUT_SIZE,
        CUBLAS_COMPUTE_32F_FAST_16F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);

    if (status != CUBLAS_STATUS_SUCCESS) {
        printf("Output layer GEMM failed: %d\n", status);
    }

    // Add bias (no ReLU)
    printf("Adding output bias...\n");
    biasOnly<<<(OUTPUT_SIZE + 255) / 256, 256>>>(d_output_temp, net->d_b2, OUTPUT_SIZE);
    cudaDeviceSynchronize();

    err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("BiasOnly kernel failed: %s\n", cudaGetErrorString(err));
    }

    // Copy back to host in float for softmax
    printf("Copying output to host...\n");
    __halfToFloatArray<<<(OUTPUT_SIZE + 255) / 256, 256>>>(d_output_temp, output, OUTPUT_SIZE);
    cudaDeviceSynchronize();

    err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("__halfToFloatArray kernel failed: %s\n", cudaGetErrorString(err));
    }

    // Debug print before softmax
    printf("Pre-softmax output: ");
    for(int i=0; i<min(5,OUTPUT_SIZE); i++) printf("%.4f ", output[i]);
    printf("\n");

    softmax(output, OUTPUT_SIZE);

    // Debug print after softmax
    printf("Post-softmax output: ");
    for(int i=0; i<min(5,OUTPUT_SIZE); i++) printf("%.4f ", output[i]);
    printf("\n");

    // Check weights and activations for NaN/Inf
    __half w_check[10];
    cudaMemcpy(w_check, net->d_W1, 10*sizeof(__half), cudaMemcpyDeviceToHost);
    printf("First 10 W1 weights: ");
    for(int i=0; i<min(10,HIDDEN_SIZE*INPUT_SIZE); i++) printf("%.4f ", __half2float(w_check[i]));
    printf("\n");

    __half act_check[HIDDEN_SIZE];
    cudaMemcpy(act_check, d_hidden_temp, HIDDEN_SIZE*sizeof(__half), cudaMemcpyDeviceToHost);

    int nan_count = 0, inf_count = 0;
    for(int i=0; i<HIDDEN_SIZE; i++) {
        if(is_half_nan(act_check[i])) nan_count++;
        if(is_half_inf(act_check[i])) inf_count++;
    }
    printf("Hidden layer - NaNs: %d, Infs: %d\n", nan_count, inf_count);

    cublasDestroy(handle);
    printf("Forward pass completed\n");
}

/*Supporting Kernels to cublas backward pass*/
////////////////////////////////////////////
// ReLU derivative kernel for FP16
__global__ void reluDerivativeKernel(__half* hidden, __half* dHidden, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        dHidden[idx] = __hmul(dHidden[idx], 
                             __hgt(hidden[idx], __float2half(0.0f)) ? 
                             __float2half(1.0f) : __float2half(0.0f));
    }
}

////////////////////////////////////////////

// -----------------------
// cuBLAS-powered Backward Pass (FP16)
// -----------------------
void backwardGPU(NeuralNetwork* net, __half* d_input, __half* d_hidden, 
        __half* d_output, double* h_target) {

    printf("Starting backward pass...\n");
    cublasHandle_t handle = net->cublas_handle;
    cublasStatus_t status = cublasSetMathMode(handle, CUBLAS_TENSOR_OP_MATH);
    if (status != CUBLAS_STATUS_SUCCESS) {
        printf("Failed to set math mode\n");
    }

    // Temporary buffers
    __half *d_dOutput, *d_dHidden;
    cudaError_t err = cudaMalloc(&d_dOutput, OUTPUT_SIZE * sizeof(__half));
    if (err != cudaSuccess) printf("cudaMalloc d_dOutput failed: %s\n", cudaGetErrorString(err));
    err = cudaMalloc(&d_dHidden, HIDDEN_SIZE * sizeof(__half));
    if (err != cudaSuccess) printf("cudaMalloc d_dHidden failed: %s\n", cudaGetErrorString(err));

    // Compute output gradient (convert to FP16)
    float h_output[OUTPUT_SIZE];
    __half h_dOutput_half[OUTPUT_SIZE];
    __halfToFloatArray<<<(OUTPUT_SIZE+255)/256, 256>>>(d_output, h_output, OUTPUT_SIZE);
    cudaDeviceSynchronize();

    printf("Output and target values:\n");
    for(int i=0; i<min(5,OUTPUT_SIZE); i++) {
        printf("[%d] out=%.4f target=%.1f ", i, h_output[i], h_target[i]);
    }
    printf("\n");

    for(int i=0; i<OUTPUT_SIZE; i++) {
        h_dOutput_half[i] = __float2half(h_output[i] - h_target[i]);
        if(is_half_nan(h_dOutput_half[i]) || is_half_inf(h_dOutput_half[i])) {
            printf("ERROR: Invalid gradient at %d: output=%.4f target=%.1f grad=%.4f\n",
                i, h_output[i], h_target[i], __half2float(h_dOutput_half[i]));
        }
    }
    err = cudaMemcpy(d_dOutput, h_dOutput_half, OUTPUT_SIZE*sizeof(__half), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) printf("cudaMemcpy d_dOutput failed: %s\n", cudaGetErrorString(err));

    // Constants for cuBLAS
    __half alpha = __float2half(1.0f);
    __half neg_lr = __float2half(-LEARNING_RATE);

    // In backwardGPU before weight updates:
    // Check gradient norm
    float grad_norm;
    status = cublasNrm2Ex(handle, OUTPUT_SIZE, d_dOutput, CUDA_R_16F, 1, &grad_norm, CUDA_R_32F, CUDA_R_32F);
    if (status != CUBLAS_STATUS_SUCCESS) {
        printf("Gradient norm computation failed\n");
    }
    printf("Output grad norm: %.4f\n", grad_norm);

    // 1. Compute hidden gradient: d_dHidden = W2^T * d_dOutput
    cublasGemmEx(handle,
        CUBLAS_OP_T, CUBLAS_OP_N,
        HIDDEN_SIZE, 1, OUTPUT_SIZE,
        &alpha,
        net->d_W2, CUDA_R_16F, OUTPUT_SIZE,
        d_dOutput, CUDA_R_16F, OUTPUT_SIZE,
        &alpha,
        d_dHidden, CUDA_R_16F, HIDDEN_SIZE,
        CUBLAS_COMPUTE_32F_FAST_16F,
        CUBLAS_GEMM_DEFAULT_TENSOR_OP);

    // Apply ReLU derivative
    reluDerivativeKernel<<<(HIDDEN_SIZE+255)/256, 256>>>(d_hidden, d_dHidden, HIDDEN_SIZE);

    // 2. Update output layer weights: W2 -= lr * d_dOutput * hidden^T
    cublasGemmEx(handle,
        CUBLAS_OP_N, CUBLAS_OP_T,
        OUTPUT_SIZE, HIDDEN_SIZE, 1,
        &neg_lr,
        d_dOutput, CUDA_R_16F, OUTPUT_SIZE,
        d_hidden, CUDA_R_16F, HIDDEN_SIZE,
        &alpha,
        net->d_W2, CUDA_R_16F, OUTPUT_SIZE,
        CUBLAS_COMPUTE_32F_FAST_16F,
        CUBLAS_GEMM_DEFAULT_TENSOR_OP);

    // 3. Update hidden layer weights: W1 -= lr * d_dHidden * input^T
    cublasGemmEx(handle,
        CUBLAS_OP_N, CUBLAS_OP_T,
        HIDDEN_SIZE, INPUT_SIZE, 1,
        &neg_lr,
        d_dHidden, CUDA_R_16F, HIDDEN_SIZE,
        d_input, CUDA_R_16F, INPUT_SIZE,
        &alpha,
        net->d_W1, CUDA_R_16F, HIDDEN_SIZE,
        CUBLAS_COMPUTE_32F_FAST_16F,
        CUBLAS_GEMM_DEFAULT_TENSOR_OP);

    // 4. Update biases
    cublasAxpyEx(handle, OUTPUT_SIZE,
        &neg_lr, CUDA_R_16F,
        d_dOutput, CUDA_R_16F, 1,
        net->d_b2, CUDA_R_16F, 1,
        CUDA_R_16F);

    cublasAxpyEx(handle, HIDDEN_SIZE,
        &neg_lr, CUDA_R_16F,
        d_dHidden, CUDA_R_16F, 1,
        net->d_b1, CUDA_R_16F, 1,
        CUDA_R_16F);

    
    __half h_output_check[OUTPUT_SIZE];
    cudaMemcpy(h_output_check, d_dOutput, OUTPUT_SIZE*sizeof(__half), cudaMemcpyDeviceToHost);
    for(int i=0; i<OUTPUT_SIZE; i++) {
        if(is_half_nan(h_output_check[i]) || is_half_inf(h_output_check[i])) {
            printf("NaN/Inf detected in d_dOutput at %d: %f (hex: %04x)\n", 
                i, __half2float(h_output_check[i]), 
                *((uint16_t*)&h_output_check[i]));
            break;
        }
    }

    // In backwardGPU, after computing gradients:
    float max_grad = 1.0f;  // Clip gradients to [-1, 1]
    for(int i=0; i<OUTPUT_SIZE; i++) {
        float grad = h_output[i] - h_target[i];
        if(grad > max_grad) grad = max_grad;
        if(grad < -max_grad) grad = -max_grad;
        h_dOutput_half[i] = __float2half(grad);
    }

    cudaFree(d_dOutput);
    cudaFree(d_dHidden);
}

// -----------------------
// Utility: Flatten 2D host matrix into contiguous 1D array
// -----------------------
double* flatten2D(double** mat, int rows, int cols) 
{
    double* flat = (double*)malloc(rows * cols * sizeof(double));
    for (int i = 0; i < rows; i++) 
    {
        for (int j = 0; j < cols; j++) 
        {
            flat[i * cols + j] = mat[i][j];
        }
    }
    return flat;
}


void train(NeuralNetwork* net, __half* d_train_images, double* h_train_labels, int numImages) {
    printf("Starting training...\n");
    // Temporary device buffers for one sample
    __half *d_input_temp, *d_hidden_temp, *d_output_temp;
    size_t size_input = INPUT_SIZE * sizeof(__half);
    size_t size_hidden = HIDDEN_SIZE * sizeof(__half);
    size_t size_output = OUTPUT_SIZE * sizeof(__half);

    cudaError_t err = cudaMalloc((void**)&d_input_temp, size_input);
    if (err != cudaSuccess) printf("cudaMalloc d_input_temp failed: %s\n", cudaGetErrorString(err));
    err = cudaMalloc((void**)&d_hidden_temp, size_hidden);
    if (err != cudaSuccess) printf("cudaMalloc d_hidden_temp failed: %s\n", cudaGetErrorString(err));
    err = cudaMalloc((void**)&d_output_temp, size_output);
    if (err != cudaSuccess) printf("cudaMalloc d_output_temp failed: %s\n", cudaGetErrorString(err));

    clock_t total_start = clock();

    for (int epoch = 0; epoch < EPOCHS; epoch++) {
        printf("\nStarting epoch %d...\n", epoch+1);
        clock_t epoch_start = clock();
        double loss = 0.0;
        int correct = 0;

        float hidden[HIDDEN_SIZE], output[OUTPUT_SIZE];

        for (int i = 0; i < numImages; i++) {
            if(i % 1000 == 0) printf("Processing sample %d...\n", i);
            
            // Get pointer to i-th input image
            __half* d_image_i = d_train_images + i * INPUT_SIZE;

            // Get label (still on host)
            double* h_label_i = h_train_labels + i * OUTPUT_SIZE;

            // Forward pass
            forwardGPU(net, d_image_i, hidden, output, d_hidden_temp, d_output_temp);

            // Backward pass
            backwardGPU(net, d_image_i, d_hidden_temp, d_output_temp, h_label_i);

            // Compute loss and accuracy on host
            double sample_loss = 0.0;
            for (int k = 0; k < OUTPUT_SIZE; k++) {
                if(output[k] <= 0.0f) {
                    printf("WARNING: Invalid output[%d]=%.6f at sample %d\n", k, output[k], i);
                    output[k] = 1e-8f;  // prevent log(0)
                }
                sample_loss -= h_label_i[k] * log(output[k]);
            }
            
            if(isnan(sample_loss) || isinf(sample_loss)) {
                printf("ERROR: Invalid loss at sample %d: %.4f\n", i, sample_loss);
                printf("Outputs: ");
                for(int k=0; k<OUTPUT_SIZE; k++) printf("%.4f ", output[k]);
                printf("\nLabels: ");
                for(int k=0; k<OUTPUT_SIZE; k++) printf("%.1f ", h_label_i[k]);
                printf("\n");
                exit(1);
            }
            
            loss += sample_loss;

            int pred = 0, actual = 0;
            for (int j = 0; j < OUTPUT_SIZE; j++) {
                if (output[j] > output[pred]) pred = j;
                if (h_label_i[j] > h_label_i[actual]) actual = j;
            }
            if (pred == actual) correct++;
        }

        printf("Epoch %d - Loss: %.4f - Train Accuracy: %.2f%% - Time: %.3fs\n",
               epoch + 1, loss / numImages, (correct / (double)numImages) * 100, get_time(epoch_start));
    }

    printf("Total training time: %.3fs\n", get_time(total_start));

    cudaFree(d_input_temp);
    cudaFree(d_hidden_temp);
    cudaFree(d_output_temp);
    printf("Training completed\n");
}


// For evaluation, we similarly use the labels on the host.
void evaluate(NeuralNetwork* net, __half* d_test_images, double* h_test_labels, int numImages) {
    __half *d_hidden_temp, *d_output_temp;
    cudaMalloc(&d_hidden_temp, HIDDEN_SIZE * sizeof(__half));
    cudaMalloc(&d_output_temp, OUTPUT_SIZE * sizeof(__half));

    float output[OUTPUT_SIZE];
    int correct = 0;

    for (int i = 0; i < numImages; i++) {
        __half* d_image_i = d_test_images + i * INPUT_SIZE;
        
        // Forward pass
        forwardGPU(net, d_image_i, NULL, output, d_hidden_temp, d_output_temp);

        // Find predicted class
        int pred = 0;
        for (int j = 1; j < OUTPUT_SIZE; j++) {
            if (output[j] > output[pred]) pred = j;
        }

        // Find actual class
        int actual = 0;
        double* h_label_i = h_test_labels + i * OUTPUT_SIZE;
        for (int j = 1; j < OUTPUT_SIZE; j++) {
            if (h_label_i[j] > h_label_i[actual]) actual = j;
        }

        if (pred == actual) correct++;
    }

    printf("Test Accuracy: %.2f%%\n", (correct / (double)numImages) * 100);
    cudaFree(d_hidden_temp);
    cudaFree(d_output_temp);
}

// -----------------------
// Data Loading Functions (Unchanged)
// -----------------------
double** loadMNISTImages(const char* filename, int numImages) {
    FILE* file = fopen(filename, "rb");
    if (!file) {
        printf("Error opening %s\n", filename);
        exit(1);
    }
    fseek(file, 16, SEEK_SET);
    double** images = allocateMatrix(numImages, INPUT_SIZE);
    for (int i = 0; i < numImages; i++) {
        for (int j = 0; j < INPUT_SIZE; j++) {
            unsigned char pixel;
            if (fread(&pixel, sizeof(unsigned char), 1, file) != 1) {
                fprintf(stderr, "Error: Failed to read pixel\n");
                fclose(file);
                exit(EXIT_FAILURE);
            }
            images[i][j] = pixel / 255.0;
        }
    }
    fclose(file);
    return images;
}

double** loadMNISTLabels(const char* filename, int numLabels) {
    FILE* file = fopen(filename, "rb");
    if (!file) {
        printf("Error opening %s\n", filename);
        exit(1);
    }
    fseek(file, 8, SEEK_SET);
    double** labels = allocateMatrix(numLabels, OUTPUT_SIZE);
    for (int i = 0; i < numLabels; i++) {
        unsigned char label;
        if (fread(&label, sizeof(unsigned char), 1, file) != 1) {
            fprintf(stderr, "Error: Failed to read label\n");
            fclose(file);
            exit(EXIT_FAILURE);
        }
        for (int j = 0; j < OUTPUT_SIZE; j++) {
            labels[i][j] = (j == label) ? 1.0 : 0.0;
        }
    }
    fclose(file);
    return labels;
}

// -----------------------
// Free Network Memory (Host and Device Part)
// -----------------------
void freeNetwork(NeuralNetwork* net) {
    cudaFree(net->d_W1);
    cudaFree(net->d_W2);
    cudaFree(net->d_b1);
    cudaFree(net->d_b2);
    free(net);
}

// -----------------------
// Main function
// -----------------------
int main() 
{
    printf("MNIST Neural Network - Optimized GPU Version (V3)\n(Update: Launch Configurations)\n(Update: Shared Memory Usage)\n\n");

    // Load the entire dataset on the host (2D arrays)
    double** train_images = loadMNISTImages("../../data/train-images.idx3-ubyte", NUM_TRAIN);
    double** train_labels = loadMNISTLabels("../../data/train-labels.idx1-ubyte", NUM_TRAIN);
    double** test_images  = loadMNISTImages("../../data/t10k-images.idx3-ubyte", NUM_TEST);
    double** test_labels  = loadMNISTLabels("../../data/t10k-labels.idx1-ubyte", NUM_TEST);

    // Flatten the host arrays into contiguous 1D arrays
    double* h_train_images_flat = flatten2D(train_images, NUM_TRAIN, INPUT_SIZE);
    double* h_train_labels_flat = flatten2D(train_labels,  NUM_TRAIN, OUTPUT_SIZE);
    double* h_test_images_flat  = flatten2D(test_images,  NUM_TEST, INPUT_SIZE);
    double* h_test_labels_flat  = flatten2D(test_labels,  NUM_TEST, OUTPUT_SIZE);

    // Allocate device memory for input data in half
    __half *d_train_images, *d_test_images;
    cudaMalloc((void**)&d_train_images, NUM_TRAIN * INPUT_SIZE * sizeof(__half));
    cudaMalloc((void**)&d_test_images, NUM_TEST * INPUT_SIZE * sizeof(__half));

    // Convert and copy to half (on host)
    __half* h_train_half = (__half*)malloc(NUM_TRAIN * INPUT_SIZE * sizeof(__half));
    __half* h_test_half  = (__half*)malloc(NUM_TEST * INPUT_SIZE * sizeof(__half));

    for (int i = 0; i < NUM_TRAIN * INPUT_SIZE; i++)
    h_train_half[i] = __float2half((float)h_train_images_flat[i]);

    for (int i = 0; i < NUM_TEST * INPUT_SIZE; i++)
    h_test_half[i] = __float2half((float)h_test_images_flat[i]);

    cudaMemcpy(d_train_images, h_train_half, NUM_TRAIN * INPUT_SIZE * sizeof(__half), cudaMemcpyHostToDevice);
    cudaMemcpy(d_test_images,  h_test_half,  NUM_TEST  * INPUT_SIZE * sizeof(__half), cudaMemcpyHostToDevice);

    // Clean up
    free(h_train_half);
    free(h_test_half);

    // Free the flattened image arrays and original 2D arrays if no longer needed.
    free(h_train_images_flat);
    free(h_test_images_flat);
    freeMatrix(train_images, NUM_TRAIN);
    freeMatrix(test_images, NUM_TEST);
    freeMatrix(train_labels, NUM_TRAIN);
    freeMatrix(test_labels, NUM_TEST);
    
    NeuralNetwork* net = createNetwork();
    
    // Train using the dataset already resident on the GPU and labels on host.
    train(net, d_train_images, h_train_labels_flat, NUM_TRAIN);
    
    // Evaluate using the dataset already resident on the GPU and labels on host.
    evaluate(net, d_test_images, h_test_labels_flat, NUM_TEST);
    
    freeNetwork(net);
    
    // Free device image dataset memory
    cudaFree(d_train_images);
    cudaFree(d_test_images);
    
    // Free the host label arrays, since we are keeping them.
    free(h_train_labels_flat);
    free(h_test_labels_flat);
    
    return 0;
}

