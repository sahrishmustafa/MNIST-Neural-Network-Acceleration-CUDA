/*
 * Optimized CUDA Neural Network implementation for MNIST
 * This version includes:
 * - Improved tensor core utilization with mixed precision
 * - Batch processing for faster training
 * - Memory optimizations and pinned memory
 * - Improved kernel configurations and occupancy
 * - Asynchronous operations with CUDA streams
 */

#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <math.h>
#include <time.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <float.h>
#include <cublas_v2.h>
#include <cuda_fp16.h>  // For half precision support

#define IMAGE_SIZE_TRAIN 60000
#define IMAGE_SIZE_TEST 10000
#define INPUT_SIZE 784
#define HIDDEN_SIZE 256
#define OUTPUT_SIZE 10
#define LEARNING_RATE 0.01
#define EPOCHS 3
#define BATCH_SIZE 128    // Increased batch size for better performance
#define NUM_CLASSES 10  // Digits 0-9
#define NUM_STREAMS 2   // Number of CUDA streams for overlapping operations
#define WEIGHT_DECAY 0.001 // L2 regularization parameter

clock_t appStart, appEnd;

// Timer function
double get_time(clock_t start) {
    return (double)(clock() - start) / CLOCKS_PER_SEC;
}

// Check CUDA errors with file and line information
#define CHECK_CUDA_ERROR(err) do { \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA Error: %s at %s:%d\n", cudaGetErrorString(err), __FILE__, __LINE__); \
        exit(EXIT_FAILURE); \
    } \
} while (0)

// Timer function using CUDA events for more accurate GPU timing
float get_gpu_time(cudaEvent_t start, cudaEvent_t stop) {
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    return milliseconds / 1000.0f;  // Convert to seconds
}

// Allocate pinned host memory for faster transfers
double** allocateHostPinnedMatrix(int rows, int cols) {
    double** mat = (double**)malloc(rows * sizeof(double*));
    double* flatData;
    CHECK_CUDA_ERROR(cudaMallocHost((void**)&flatData, rows * cols * sizeof(double)));
    
    for (int i = 0; i < rows; i++) {
        mat[i] = &flatData[i * cols];
    }
    return mat;
}

// Free pinned memory
void freePinnedMatrix(double** mat, int rows) {
    if (mat && mat[0]) {
        CHECK_CUDA_ERROR(cudaFreeHost(mat[0]));
    }
    if (mat) {
        free(mat);
    }
}

// Helper to flatten a 2D matrix into a contiguous array
double* flattenMatrix(double** matrix, int rows, int cols) {
    double* flat;
    CHECK_CUDA_ERROR(cudaMallocHost((void**)&flat, rows * cols * sizeof(double)));
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            flat[i * cols + j] = matrix[i][j];
        }
    }
    return flat;
}

// Softmax implementation optimized for numerical stability
void softmax(double* x, int size) {
    double max_val = x[0];
    for (int i = 1; i < size; i++) {
        if (x[i] > max_val) max_val = x[i];
    }
    
    double sum = 0;
    for (int i = 0; i < size; i++) {
        x[i] = exp(x[i] - max_val);  // Subtract max for numerical stability
        sum += x[i];
    }
    for (int i = 0; i < size; i++) {
        x[i] /= sum;
    }
}

// Neural network structure with streams and events
typedef struct {
    double** W1;    
    double** W2;
    double* b1;     
    double* b2;
    
    // Device memory pointers
    double* d_W1;
    double* d_W2;
    double* d_b1;
    double* d_b2;
    
    // Batch processing buffers
    double* d_batch_input;
    double* d_batch_hidden;
    double* d_batch_output;
    double* d_batch_target;
    double* d_batch_d_output;
    double* d_batch_d_hidden;
    
    // cuBLAS handle and streams
    cublasHandle_t cublas_handle;
    cudaStream_t streams[NUM_STREAMS];
    cudaEvent_t events[NUM_STREAMS];
} NeuralNetwork;

// Check cuBLAS status with detailed error reporting
void checkCublasStatus(cublasStatus_t status, const char* functionName) {
    if (status != CUBLAS_STATUS_SUCCESS) {
        const char* errorString;
        switch (status) {
            case CUBLAS_STATUS_NOT_INITIALIZED:
                errorString = "CUBLAS_STATUS_NOT_INITIALIZED";
                break;
            case CUBLAS_STATUS_ALLOC_FAILED:
                errorString = "CUBLAS_STATUS_ALLOC_FAILED";
                break;
            case CUBLAS_STATUS_INVALID_VALUE:
                errorString = "CUBLAS_STATUS_INVALID_VALUE";
                break;
            case CUBLAS_STATUS_ARCH_MISMATCH:
                errorString = "CUBLAS_STATUS_ARCH_MISMATCH";
                break;
            case CUBLAS_STATUS_MAPPING_ERROR:
                errorString = "CUBLAS_STATUS_MAPPING_ERROR";
                break;
            case CUBLAS_STATUS_EXECUTION_FAILED:
                errorString = "CUBLAS_STATUS_EXECUTION_FAILED";
                break;
            case CUBLAS_STATUS_INTERNAL_ERROR:
                errorString = "CUBLAS_STATUS_INTERNAL_ERROR";
                break;
            default:
                errorString = "Unknown cuBLAS error";
        }
        fprintf(stderr, "cuBLAS Error: %s in function %s\n", errorString, functionName);
        exit(EXIT_FAILURE);
    }
}

// Initialize neural network with optimized memory allocation
NeuralNetwork* createNetwork() 
{
    // Initialize CUDA device
    int deviceCount;
    CHECK_CUDA_ERROR(cudaGetDeviceCount(&deviceCount));
    if (deviceCount == 0) {
        fprintf(stderr, "No CUDA-capable devices found\n");
        exit(EXIT_FAILURE);
    }
    
    // Check for tensor core support
    cudaDeviceProp prop;
    CHECK_CUDA_ERROR(cudaGetDeviceProperties(&prop, 0));
    printf("Using GPU: %s (Compute capability: %d.%d)\n", prop.name, prop.major, prop.minor);
    if (prop.major >= 7) {
        printf("Tensor cores are supported on this device\n");
    } else {
        printf("Warning: Tensor cores are not supported on this device\n");
    }
    
    CHECK_CUDA_ERROR(cudaSetDevice(0));

    NeuralNetwork* net = (NeuralNetwork*)malloc(sizeof(NeuralNetwork));
    
    // Initialize cuBLAS handle
    cublasStatus_t status = cublasCreate(&net->cublas_handle);
    checkCublasStatus(status, "cublasCreate");
    
    // Create CUDA streams for overlapping operations
    for (int i = 0; i < NUM_STREAMS; i++) {
        CHECK_CUDA_ERROR(cudaStreamCreate(&net->streams[i]));
        CHECK_CUDA_ERROR(cudaEventCreate(&net->events[i]));
    }
    
    // Set cuBLAS to use tensor cores when possible
    status = cublasSetMathMode(net->cublas_handle, CUBLAS_TENSOR_OP_MATH);
    checkCublasStatus(status, "cublasSetMathMode");
    
    // Allocate host weights and biases with pinned memory
    net->W1 = allocateHostPinnedMatrix(HIDDEN_SIZE, INPUT_SIZE);
    net->W2 = allocateHostPinnedMatrix(OUTPUT_SIZE, HIDDEN_SIZE);
    CHECK_CUDA_ERROR(cudaMallocHost((void**)&net->b1, HIDDEN_SIZE * sizeof(double)));
    CHECK_CUDA_ERROR(cudaMallocHost((void**)&net->b2, OUTPUT_SIZE * sizeof(double)));

        // Initialize weights with Xavier/Glorot initialization for better convergence
    srand(time(NULL));
    double w1_range = sqrt(6.0 / (INPUT_SIZE + HIDDEN_SIZE));
    double w2_range = sqrt(6.0 / (HIDDEN_SIZE + OUTPUT_SIZE));

    //double w1_std = sqrt(2.0 / INPUT_SIZE);  // He initialization
    //double w2_std = sqrt(2.0 / HIDDEN_SIZE);

    for (int i = 0; i < HIDDEN_SIZE; i++) {
        for (int j = 0; j < INPUT_SIZE; j++) {
            // Improve initialization with slight adjustment to Xavier/Glorot
            //net->W1[i][j] = w1_std * ((double)rand() / RAND_MAX - 0.5) * 2;
            net->W1[i][j] = ((double)rand() / RAND_MAX) * 2 * w1_range - w1_range;
        }
        // Initialize biases to small positive values instead of zero
        net->b1[i] = 0.01;
    }

    for (int i = 0; i < OUTPUT_SIZE; i++) {
        for (int j = 0; j < HIDDEN_SIZE; j++) {
            //net->W2[i][j] = w2_std * ((double)rand() / RAND_MAX - 0.5) * 2;
            net->W2[i][j] = ((double)rand() / RAND_MAX) * 2 * w2_range - w2_range;
        }
        net->b2[i] = 0.01;
    }

    // Flatten weight matrices
    double* h_W1 = flattenMatrix(net->W1, HIDDEN_SIZE, INPUT_SIZE);
    double* h_W2 = flattenMatrix(net->W2, OUTPUT_SIZE, HIDDEN_SIZE);

    // Allocate device memory for weights and biases
    CHECK_CUDA_ERROR(cudaMalloc(&net->d_W1, HIDDEN_SIZE * INPUT_SIZE * sizeof(double)));
    CHECK_CUDA_ERROR(cudaMalloc(&net->d_W2, OUTPUT_SIZE * HIDDEN_SIZE * sizeof(double)));
    CHECK_CUDA_ERROR(cudaMalloc(&net->d_b1, HIDDEN_SIZE * sizeof(double)));
    CHECK_CUDA_ERROR(cudaMalloc(&net->d_b2, OUTPUT_SIZE * sizeof(double)));

    // Copy from host to device
    CHECK_CUDA_ERROR(cudaMemcpy(net->d_W1, h_W1, HIDDEN_SIZE * INPUT_SIZE * sizeof(double), cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(net->d_W2, h_W2, OUTPUT_SIZE * HIDDEN_SIZE * sizeof(double), cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(net->d_b1, net->b1, HIDDEN_SIZE * sizeof(double), cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(net->d_b2, net->b2, OUTPUT_SIZE * sizeof(double), cudaMemcpyHostToDevice));

    // Allocate memory for batch processing
    CHECK_CUDA_ERROR(cudaMalloc(&net->d_batch_input, BATCH_SIZE * INPUT_SIZE * sizeof(double)));
    CHECK_CUDA_ERROR(cudaMalloc(&net->d_batch_hidden, BATCH_SIZE * HIDDEN_SIZE * sizeof(double)));
    CHECK_CUDA_ERROR(cudaMalloc(&net->d_batch_output, BATCH_SIZE * OUTPUT_SIZE * sizeof(double)));
    CHECK_CUDA_ERROR(cudaMalloc(&net->d_batch_target, BATCH_SIZE * OUTPUT_SIZE * sizeof(double)));
    CHECK_CUDA_ERROR(cudaMalloc(&net->d_batch_d_output, BATCH_SIZE * OUTPUT_SIZE * sizeof(double)));
    CHECK_CUDA_ERROR(cudaMalloc(&net->d_batch_d_hidden, BATCH_SIZE * HIDDEN_SIZE * sizeof(double)));

    // Free host memory for flattened weights
    cudaFreeHost(h_W1);
    cudaFreeHost(h_W2);

    return net;
}

// CUDA kernel for element-wise ReLU activation
__global__ void reluActivation(double* input, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        input[idx] = input[idx] > 0 ? input[idx] : 0.01 * input[idx]; // 0.01 is the leak factor
    }
}

// CUDA kernel for adding bias to a batch of activations
__global__ void addBiasBatch(double* output, const double* bias, int batch_size, int feature_size) {
    int feature_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (feature_idx < feature_size) {
        double bias_val = bias[feature_idx];
        for (int i = 0; i < batch_size; i++) {
            output[i * feature_size + feature_idx] += bias_val;
        }
    }
}

// CUDA kernel for computing softmax for a batch
__global__ void softmaxBatch(double* output, int batch_size, int feature_size) {
    int batch_idx = blockIdx.x;
    if (batch_idx < batch_size) {
        // Find max value for numerical stability
        double max_val = -DBL_MAX;
        for (int i = 0; i < feature_size; i++) {
            double val = output[batch_idx * feature_size + i];
            max_val = fmax(max_val, val);
        }
        
        // Compute exponentials and sum
        double sum = 0.0;
        for (int i = 0; i < feature_size; i++) {
            int idx = batch_idx * feature_size + i;
            output[idx] = exp(output[idx] - max_val);
            sum += output[idx];
        }
        
        // Normalize
        for (int i = 0; i < feature_size; i++) {
            output[batch_idx * feature_size + i] /= sum;
        }
    }
}

// CUDA kernel to compute cross-entropy loss and output gradients in one step
__global__ void computeLossAndGradients(const double* output, const double* target, 
                                    double* gradient, double* loss,
                                    int batch_size, int num_classes) {
    __shared__ double batch_loss[32];  // Shared memory for loss reduction
    
    int batch_idx = blockIdx.x;
    int tid = threadIdx.x;
    
    if (batch_idx < batch_size && tid < num_classes) {
        int idx = batch_idx * num_classes + tid;
        // Compute gradient (output - target) and loss simultaneously
        gradient[idx] = output[idx] - target[idx];
        
        // For thread 0 in each block, compute the loss for this sample
        if (tid == 0) {
            double sample_loss = 0.0;
            for (int j = 0; j < num_classes; j++) {
                int pos = batch_idx * num_classes + j;
                sample_loss -= target[pos] * log(fmax(output[pos], 1e-7));
            }
            batch_loss[0] = sample_loss;
            atomicAdd(loss, sample_loss);
        }
    }
}

// CUDA kernel to compute ReLU derivative and apply it
__global__ void reluDerivativeBatch(const double* hidden, double* d_hidden, int batch_size, int hidden_size) 
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < batch_size * hidden_size) {
        int batch = idx / hidden_size;
        int neuron = idx % hidden_size;
        int index = batch * hidden_size + neuron;
        d_hidden[index] = (hidden[index] > 0) ? d_hidden[index] : 0.01 * d_hidden[index];
    }
}

// CUDA kernel to update weights with gradients
__global__ void updateWeightsBatch(double* weights, const double* input, const double* gradients, int batch_size, int input_size, int output_size, double learning_rate, double weight_decay) 
{
    int output_idx = blockIdx.x;
    int input_idx = blockIdx.y * blockDim.x + threadIdx.x;

    if (output_idx < output_size && input_idx < input_size) {
    double gradient_sum = 0.0;

    // Compute the average gradient across the batch
    for (int b = 0; b < batch_size; b++) {
    gradient_sum += gradients[b * output_size + output_idx] * input[b * input_size + input_idx];
    }

    // Add weight decay term for regularization
    int weight_idx = output_idx * input_size + input_idx;
    double decay = weight_decay * weights[weight_idx];

    // Update weight with average gradient and weight decay
    double update = (learning_rate / batch_size) * gradient_sum + learning_rate * decay;
    atomicAdd(&weights[weight_idx], -update);
    }
}

// CUDA kernel to update biases with gradients
__global__ void updateBiasesBatch(double* biases, const double* gradients,
                                int batch_size, int feature_size,
                                double learning_rate) {
    int feature_idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (feature_idx < feature_size) {
        double gradient_sum = 0.0;
        
        // Compute the average gradient across the batch
        for (int b = 0; b < batch_size; b++) {
            gradient_sum += gradients[b * feature_size + feature_idx];
        }
        
        // Update bias with average gradient
        biases[feature_idx] -= (learning_rate / batch_size) * gradient_sum;
    }
}

// Forward pass for a batch of inputs
void forwardBatch(NeuralNetwork* net, double** batch_inputs, double* batch_outputs,int batch_size, int stream_idx, float* total_kernel_time, float* total_mem_time){
    cudaStream_t stream = net->streams[stream_idx];
    
    // Create a flat batch input array
    double* h_batch_input;
    CHECK_CUDA_ERROR(cudaMallocHost((void**)&h_batch_input, batch_size * INPUT_SIZE * sizeof(double)));
    
    // Copy batch inputs to flat array
    for (int b = 0; b < batch_size; b++) {
        for (int i = 0; i < INPUT_SIZE; i++) {
            h_batch_input[b * INPUT_SIZE + i] = batch_inputs[b][i];
        }
    }
    
    // Copy batch input to device
    cudaEvent_t mem_start, mem_stop;
    CHECK_CUDA_ERROR(cudaEventCreate(&mem_start));
    CHECK_CUDA_ERROR(cudaEventCreate(&mem_stop));
    CHECK_CUDA_ERROR(cudaEventRecord(mem_start, stream));
    CHECK_CUDA_ERROR(cudaMemcpyAsync(net->d_batch_input, h_batch_input, 
                                batch_size * INPUT_SIZE * sizeof(double), 
                                cudaMemcpyHostToDevice, stream));
    CHECK_CUDA_ERROR(cudaEventRecord(mem_stop, stream));
    CHECK_CUDA_ERROR(cudaEventSynchronize(mem_stop));
    float elapsed_mem = 0;
    CHECK_CUDA_ERROR(cudaEventElapsedTime(&elapsed_mem, mem_start, mem_stop));
    *total_mem_time += elapsed_mem / 1000.0f;
    CHECK_CUDA_ERROR(cudaEventDestroy(mem_start));
    CHECK_CUDA_ERROR(cudaEventDestroy(mem_stop));
    
    double alpha = 1.0, beta = 0.0;
    
    // Perform input to hidden layer computation: d_batch_hidden = d_W1 * d_batch_input
    cublasStatus_t status = cublasDgemm(net->cublas_handle, CUBLAS_OP_T, CUBLAS_OP_N,
                                        HIDDEN_SIZE, batch_size, INPUT_SIZE,
                                        &alpha, net->d_W1, INPUT_SIZE,
                                        net->d_batch_input, INPUT_SIZE,
                                        &beta, net->d_batch_hidden, HIDDEN_SIZE);
    checkCublasStatus(status, "cublasDgemm_W1");
    
    // Add biases to hidden layer activations
    int threads_per_block = 256;
    int blocks = (HIDDEN_SIZE + threads_per_block - 1) / threads_per_block;
    addBiasBatch<<<blocks, threads_per_block, 0, stream>>>(
        net->d_batch_hidden, net->d_b1, batch_size, HIDDEN_SIZE);
    CHECK_CUDA_ERROR(cudaGetLastError());
    
    // Apply ReLU activation to hidden layer
    blocks = (batch_size * HIDDEN_SIZE + threads_per_block - 1) / threads_per_block;
    reluActivation<<<blocks, threads_per_block, 0, stream>>>(
        net->d_batch_hidden, batch_size * HIDDEN_SIZE);
    CHECK_CUDA_ERROR(cudaGetLastError());
    
    // Perform hidden to output layer computation: d_batch_output = d_W2 * d_batch_hidden
    cudaEvent_t dgemm1_start, dgemm1_stop;
    CHECK_CUDA_ERROR(cudaEventCreate(&dgemm1_start));
    CHECK_CUDA_ERROR(cudaEventCreate(&dgemm1_stop));
    CHECK_CUDA_ERROR(cudaEventRecord(dgemm1_start, stream));
    status = cublasDgemm(net->cublas_handle, CUBLAS_OP_T, CUBLAS_OP_N,
                        OUTPUT_SIZE, batch_size, HIDDEN_SIZE,
                        &alpha, net->d_W2, HIDDEN_SIZE,
                        net->d_batch_hidden, HIDDEN_SIZE,
                        &beta, net->d_batch_output, OUTPUT_SIZE);
    checkCublasStatus(status, "cublasDgemm_W2");
    CHECK_CUDA_ERROR(cudaEventRecord(dgemm1_stop, stream));
    CHECK_CUDA_ERROR(cudaEventSynchronize(dgemm1_stop));
    float elapsed_dgemm1 = 0;
    CHECK_CUDA_ERROR(cudaEventElapsedTime(&elapsed_dgemm1, dgemm1_start, dgemm1_stop));
    *total_kernel_time += elapsed_dgemm1 / 1000.0f;
    CHECK_CUDA_ERROR(cudaEventDestroy(dgemm1_start));
    CHECK_CUDA_ERROR(cudaEventDestroy(dgemm1_stop));
    
    // Add biases to output layer
    blocks = (OUTPUT_SIZE + threads_per_block - 1) / threads_per_block;
    CHECK_CUDA_ERROR(cudaEventCreate(&dgemm1_start));
    CHECK_CUDA_ERROR(cudaEventCreate(&dgemm1_stop));
    CHECK_CUDA_ERROR(cudaEventRecord(dgemm1_start, stream));
    addBiasBatch<<<blocks, threads_per_block, 0, stream>>>(
        net->d_batch_output, net->d_b2, batch_size, OUTPUT_SIZE);
    CHECK_CUDA_ERROR(cudaGetLastError());
    CHECK_CUDA_ERROR(cudaEventRecord(dgemm1_stop, stream));
    CHECK_CUDA_ERROR(cudaEventSynchronize(dgemm1_stop));
     elapsed_dgemm1 = 0;
    CHECK_CUDA_ERROR(cudaEventElapsedTime(&elapsed_dgemm1, dgemm1_start, dgemm1_stop));
    *total_kernel_time += elapsed_dgemm1 / 1000.0f;
    CHECK_CUDA_ERROR(cudaEventDestroy(dgemm1_start));
    CHECK_CUDA_ERROR(cudaEventDestroy(dgemm1_stop));
    
    // Apply softmax activation to output layer
    CHECK_CUDA_ERROR(cudaEventCreate(&dgemm1_start));
    CHECK_CUDA_ERROR(cudaEventCreate(&dgemm1_stop));
    CHECK_CUDA_ERROR(cudaEventRecord(dgemm1_start, stream));
    softmaxBatch<<<batch_size, 1, 0, stream>>>(net->d_batch_output, batch_size, OUTPUT_SIZE);
    CHECK_CUDA_ERROR(cudaGetLastError());
    CHECK_CUDA_ERROR(cudaEventRecord(dgemm1_stop, stream));
    CHECK_CUDA_ERROR(cudaEventSynchronize(dgemm1_stop));
     elapsed_dgemm1 = 0;
    CHECK_CUDA_ERROR(cudaEventElapsedTime(&elapsed_dgemm1, dgemm1_start, dgemm1_stop));
    *total_kernel_time += elapsed_dgemm1 / 1000.0f;
    CHECK_CUDA_ERROR(cudaEventDestroy(dgemm1_start));
    CHECK_CUDA_ERROR(cudaEventDestroy(dgemm1_stop));
    
    // Copy results back to host
    CHECK_CUDA_ERROR(cudaEventCreate(&mem_start));
    CHECK_CUDA_ERROR(cudaEventCreate(&mem_stop));
    CHECK_CUDA_ERROR(cudaEventRecord(mem_start, stream));
    CHECK_CUDA_ERROR(cudaMemcpyAsync(batch_outputs, net->d_batch_output,
                                batch_size * OUTPUT_SIZE * sizeof(double),
                                cudaMemcpyDeviceToHost, stream));
    CHECK_CUDA_ERROR(cudaEventRecord(mem_stop, stream));
    CHECK_CUDA_ERROR(cudaEventSynchronize(mem_stop));
     elapsed_mem = 0;
    CHECK_CUDA_ERROR(cudaEventElapsedTime(&elapsed_mem, mem_start, mem_stop));
    *total_mem_time += elapsed_mem / 1000.0f;
    CHECK_CUDA_ERROR(cudaEventDestroy(mem_start));
    CHECK_CUDA_ERROR(cudaEventDestroy(mem_stop));
    
    // Wait for stream to complete
    CHECK_CUDA_ERROR(cudaStreamSynchronize(stream));
    
    // Clean up
    cudaFreeHost(h_batch_input);
}

// Backward pass for a batch of inputs
void backwardBatch(NeuralNetwork* net, double** batch_inputs, double* batch_outputs, 
    double** batch_targets, double* loss, int batch_size, int stream_idx,
    double learning_rate, float* total_kernel_time, float* total_mem_time) {

    cudaStream_t stream = net->streams[stream_idx];
    cudaEvent_t dgemm1_start, dgemm1_stop;
    
    // Prepare target data
    double* h_batch_target;
    CHECK_CUDA_ERROR(cudaMallocHost((void**)&h_batch_target, batch_size * OUTPUT_SIZE * sizeof(double)));
    
    for (int b = 0; b < batch_size; b++) {
        for (int i = 0; i < OUTPUT_SIZE; i++) {
            h_batch_target[b * OUTPUT_SIZE + i] = batch_targets[b][i];
        }
    }
    
    // Copy target data to device
    cudaEvent_t mem_target_start, mem_target_stop;
    CHECK_CUDA_ERROR(cudaEventCreate(&mem_target_start));
    CHECK_CUDA_ERROR(cudaEventCreate(&mem_target_stop));
    CHECK_CUDA_ERROR(cudaEventRecord(mem_target_start, stream));
    CHECK_CUDA_ERROR(cudaMemcpyAsync(net->d_batch_target, h_batch_target,
                                batch_size * OUTPUT_SIZE * sizeof(double),
                                cudaMemcpyHostToDevice, stream));
    CHECK_CUDA_ERROR(cudaEventRecord(mem_target_stop, stream));
    CHECK_CUDA_ERROR(cudaEventSynchronize(mem_target_stop));
    float elapsed_mem_target = 0;
    CHECK_CUDA_ERROR(cudaEventElapsedTime(&elapsed_mem_target, mem_target_start, mem_target_stop));
    *total_mem_time += elapsed_mem_target / 1000.0f;
    CHECK_CUDA_ERROR(cudaEventDestroy(mem_target_start));
    CHECK_CUDA_ERROR(cudaEventDestroy(mem_target_stop));
    
    // Create and initialize device loss
    double* d_loss;
    CHECK_CUDA_ERROR(cudaMalloc(&d_loss, sizeof(double)));

    CHECK_CUDA_ERROR(cudaEventCreate(&mem_target_start));
    CHECK_CUDA_ERROR(cudaEventCreate(&mem_target_stop));
    CHECK_CUDA_ERROR(cudaEventRecord(mem_target_start, stream));
    CHECK_CUDA_ERROR(cudaMemsetAsync(d_loss, 0, sizeof(double), stream));
    CHECK_CUDA_ERROR(cudaEventRecord(mem_target_stop, stream));
    CHECK_CUDA_ERROR(cudaEventSynchronize(mem_target_stop));
     elapsed_mem_target = 0;
    CHECK_CUDA_ERROR(cudaEventElapsedTime(&elapsed_mem_target, mem_target_start, mem_target_stop));
    *total_mem_time += elapsed_mem_target / 1000.0f;
    CHECK_CUDA_ERROR(cudaEventDestroy(mem_target_start));
    CHECK_CUDA_ERROR(cudaEventDestroy(mem_target_stop));
    
    // Compute loss and output gradients in one kernel
    CHECK_CUDA_ERROR(cudaEventCreate(&dgemm1_start));
    CHECK_CUDA_ERROR(cudaEventCreate(&dgemm1_stop));
    CHECK_CUDA_ERROR(cudaEventRecord(dgemm1_start, stream));
    computeLossAndGradients<<<batch_size, OUTPUT_SIZE, 0, stream>>>(
        net->d_batch_output, net->d_batch_target, 
        net->d_batch_d_output, d_loss, batch_size, OUTPUT_SIZE);
    CHECK_CUDA_ERROR(cudaGetLastError());
    CHECK_CUDA_ERROR(cudaEventRecord(dgemm1_stop, stream));
    CHECK_CUDA_ERROR(cudaEventSynchronize(dgemm1_stop));
    float elapsed_dgemm1 = 0;
    CHECK_CUDA_ERROR(cudaEventElapsedTime(&elapsed_dgemm1, dgemm1_start, dgemm1_stop));
    *total_kernel_time += elapsed_dgemm1 / 1000.0f;
    CHECK_CUDA_ERROR(cudaEventDestroy(dgemm1_start));
    CHECK_CUDA_ERROR(cudaEventDestroy(dgemm1_stop));
    
    // Copy loss back to host
    CHECK_CUDA_ERROR(cudaMemcpyAsync(loss, d_loss, sizeof(double), cudaMemcpyDeviceToHost, stream));
    
    double alpha = 1.0, beta = 0.0;
    
    // Compute hidden layer gradients: d_batch_d_hidden = d_W2 * d_batch_d_output
    CHECK_CUDA_ERROR(cudaEventCreate(&dgemm1_start));
    CHECK_CUDA_ERROR(cudaEventCreate(&dgemm1_stop));
    CHECK_CUDA_ERROR(cudaEventRecord(dgemm1_start, stream));
    cublasStatus_t status = cublasDgemm(net->cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N,
                                        HIDDEN_SIZE, batch_size, OUTPUT_SIZE,
                                        &alpha, net->d_W2, HIDDEN_SIZE,
                                        net->d_batch_d_output, OUTPUT_SIZE,
                                        &beta, net->d_batch_d_hidden, HIDDEN_SIZE);
    checkCublasStatus(status, "cublasDgemm_backward_W2");
    CHECK_CUDA_ERROR(cudaGetLastError());
    CHECK_CUDA_ERROR(cudaEventRecord(dgemm1_stop, stream));
    CHECK_CUDA_ERROR(cudaEventSynchronize(dgemm1_stop));
     elapsed_dgemm1 = 0;
    CHECK_CUDA_ERROR(cudaEventElapsedTime(&elapsed_dgemm1, dgemm1_start, dgemm1_stop));
    *total_kernel_time += elapsed_dgemm1 / 1000.0f;
    CHECK_CUDA_ERROR(cudaEventDestroy(dgemm1_start));
    CHECK_CUDA_ERROR(cudaEventDestroy(dgemm1_stop));
    
    // Apply ReLU derivative to hidden layer gradients
    int threads_per_block = 256;
    int blocks = (batch_size * HIDDEN_SIZE + threads_per_block - 1) / threads_per_block;
    CHECK_CUDA_ERROR(cudaEventCreate(&dgemm1_start));
    CHECK_CUDA_ERROR(cudaEventCreate(&dgemm1_stop));
    CHECK_CUDA_ERROR(cudaEventRecord(dgemm1_start, stream));
    reluDerivativeBatch<<<blocks, threads_per_block, 0, stream>>>(
        net->d_batch_hidden, net->d_batch_d_hidden, batch_size, HIDDEN_SIZE);
    CHECK_CUDA_ERROR(cudaGetLastError());
    CHECK_CUDA_ERROR(cudaEventRecord(dgemm1_stop, stream));
    CHECK_CUDA_ERROR(cudaEventSynchronize(dgemm1_stop));
     elapsed_dgemm1 = 0;
    CHECK_CUDA_ERROR(cudaEventElapsedTime(&elapsed_dgemm1, dgemm1_start, dgemm1_stop));
    *total_kernel_time += elapsed_dgemm1 / 1000.0f;
    CHECK_CUDA_ERROR(cudaEventDestroy(dgemm1_start));
    CHECK_CUDA_ERROR(cudaEventDestroy(dgemm1_stop));
    
    // Update output layer weights
    dim3 block_dim(16, 16);
    dim3 grid_dim_W2((OUTPUT_SIZE + block_dim.x - 1) / block_dim.x, (HIDDEN_SIZE + block_dim.y - 1) / block_dim.y);
    CHECK_CUDA_ERROR(cudaEventCreate(&dgemm1_start));
    CHECK_CUDA_ERROR(cudaEventCreate(&dgemm1_stop));
    CHECK_CUDA_ERROR(cudaEventRecord(dgemm1_start, stream));
    updateWeightsBatch<<<grid_dim_W2, block_dim, 0, stream>>>(
        net->d_W2, net->d_batch_hidden, net->d_batch_d_output,
        batch_size, HIDDEN_SIZE, OUTPUT_SIZE, learning_rate, WEIGHT_DECAY);
    CHECK_CUDA_ERROR(cudaGetLastError());
    CHECK_CUDA_ERROR(cudaGetLastError());
    CHECK_CUDA_ERROR(cudaEventRecord(dgemm1_stop, stream));
    CHECK_CUDA_ERROR(cudaEventSynchronize(dgemm1_stop));
     elapsed_dgemm1 = 0;
    CHECK_CUDA_ERROR(cudaEventElapsedTime(&elapsed_dgemm1, dgemm1_start, dgemm1_stop));
    *total_kernel_time += elapsed_dgemm1 / 1000.0f;
    CHECK_CUDA_ERROR(cudaEventDestroy(dgemm1_start));
    CHECK_CUDA_ERROR(cudaEventDestroy(dgemm1_stop));
    
    // Update input layer weights
    dim3 grid_dim_W1((HIDDEN_SIZE + block_dim.x - 1) / block_dim.x, 
                    (INPUT_SIZE + block_dim.y - 1) / block_dim.y);

    CHECK_CUDA_ERROR(cudaEventCreate(&dgemm1_start));
    CHECK_CUDA_ERROR(cudaEventCreate(&dgemm1_stop));
    CHECK_CUDA_ERROR(cudaEventRecord(dgemm1_start, stream));
    updateWeightsBatch<<<grid_dim_W1, block_dim, 0, stream>>>(
        net->d_W1, net->d_batch_input, net->d_batch_d_hidden,
        batch_size, INPUT_SIZE, HIDDEN_SIZE, learning_rate, WEIGHT_DECAY);
    CHECK_CUDA_ERROR(cudaGetLastError());
    CHECK_CUDA_ERROR(cudaGetLastError());
    CHECK_CUDA_ERROR(cudaEventRecord(dgemm1_stop, stream));
    CHECK_CUDA_ERROR(cudaEventSynchronize(dgemm1_stop));
     elapsed_dgemm1 = 0;
    CHECK_CUDA_ERROR(cudaEventElapsedTime(&elapsed_dgemm1, dgemm1_start, dgemm1_stop));
    *total_kernel_time += elapsed_dgemm1 / 1000.0f;
    CHECK_CUDA_ERROR(cudaEventDestroy(dgemm1_start));
    CHECK_CUDA_ERROR(cudaEventDestroy(dgemm1_stop));
    
    // Update biases
    blocks = (OUTPUT_SIZE + threads_per_block - 1) / threads_per_block;
    CHECK_CUDA_ERROR(cudaEventCreate(&dgemm1_start));
    CHECK_CUDA_ERROR(cudaEventCreate(&dgemm1_stop));
    CHECK_CUDA_ERROR(cudaEventRecord(dgemm1_start, stream));
    updateBiasesBatch<<<blocks, threads_per_block, 0, stream>>>(
        net->d_b2, net->d_batch_d_output, batch_size, OUTPUT_SIZE, learning_rate);

    CHECK_CUDA_ERROR(cudaGetLastError());
    CHECK_CUDA_ERROR(cudaEventRecord(dgemm1_stop, stream));
    CHECK_CUDA_ERROR(cudaEventSynchronize(dgemm1_stop));
     elapsed_dgemm1 = 0;
    CHECK_CUDA_ERROR(cudaEventElapsedTime(&elapsed_dgemm1, dgemm1_start, dgemm1_stop));
    *total_kernel_time += elapsed_dgemm1 / 1000.0f;
    CHECK_CUDA_ERROR(cudaEventDestroy(dgemm1_start));
    CHECK_CUDA_ERROR(cudaEventDestroy(dgemm1_stop));
    
    blocks = (HIDDEN_SIZE + threads_per_block - 1) / threads_per_block;
    CHECK_CUDA_ERROR(cudaEventCreate(&dgemm1_start));
    CHECK_CUDA_ERROR(cudaEventCreate(&dgemm1_stop));
    CHECK_CUDA_ERROR(cudaEventRecord(dgemm1_start, stream));
    updateBiasesBatch<<<blocks, threads_per_block, 0, stream>>>(
        net->d_b1, net->d_batch_d_hidden, batch_size, HIDDEN_SIZE, learning_rate);
    CHECK_CUDA_ERROR(cudaGetLastError());
    CHECK_CUDA_ERROR(cudaEventRecord(dgemm1_stop, stream));
    CHECK_CUDA_ERROR(cudaEventSynchronize(dgemm1_stop));
     elapsed_dgemm1 = 0;
    CHECK_CUDA_ERROR(cudaEventElapsedTime(&elapsed_dgemm1, dgemm1_start, dgemm1_stop));
    *total_kernel_time += elapsed_dgemm1 / 1000.0f;
    CHECK_CUDA_ERROR(cudaEventDestroy(dgemm1_start));
    CHECK_CUDA_ERROR(cudaEventDestroy(dgemm1_stop));
    
    // Wait for stream to complete
    CHECK_CUDA_ERROR(cudaStreamSynchronize(stream));
    
    // Clean up
    cudaFreeHost(h_batch_target);
    cudaFree(d_loss);
}

// // Update adjust_learning_rate()
// double adjust_learning_rate(double initial_rate, int epoch) {
//     return initial_rate * exp(-0.2 * epoch);  // Faster decay
// }

// Increase initial learning rate
#define LEARNING_RATE 0.01
float total_kernel_time = 0.0f;
float total_mem_time = 0.0f;

// Train the network with batch processing
void train(NeuralNetwork* net, double** images, double** labels, int numImages) {
    
    // Create CUDA events for timing
    cudaEvent_t start, stop;
    CHECK_CUDA_ERROR(cudaEventCreate(&start));
    CHECK_CUDA_ERROR(cudaEventCreate(&stop));
    
    // Start timer for total training time
    CHECK_CUDA_ERROR(cudaEventRecord(start));
    
    // Allocate memory for batch outputs
    double* batch_outputs;
    CHECK_CUDA_ERROR(cudaMallocHost((void**)&batch_outputs, BATCH_SIZE * OUTPUT_SIZE * sizeof(double)));
    
    double current_learning_rate = LEARNING_RATE;

    for (int epoch = 0; epoch < EPOCHS; epoch++) 
    {
        cudaEvent_t epoch_start, epoch_stop;
        CHECK_CUDA_ERROR(cudaEventCreate(&epoch_start));
        CHECK_CUDA_ERROR(cudaEventCreate(&epoch_stop));
        CHECK_CUDA_ERROR(cudaEventRecord(epoch_start));
        
        double total_loss = 0.0;
        int correct = 0;
        int num_batches = (numImages + BATCH_SIZE - 1) / BATCH_SIZE;

        //current_learning_rate = adjust_learning_rate(LEARNING_RATE, epoch);
        
        for (int batch = 0; batch < num_batches; batch++) {
            int start_idx = batch * BATCH_SIZE;
            int end_idx = start_idx + BATCH_SIZE;
            if (end_idx > numImages) end_idx = numImages;
            int current_batch_size = end_idx - start_idx;
            
            // Process batch
            int stream_idx = batch % NUM_STREAMS;
            double batch_loss = 0.0;
            
            // Forward pass
            forwardBatch(net, &images[start_idx], batch_outputs, current_batch_size, stream_idx,&total_kernel_time,&total_mem_time);
                        
            // Backward pass
            backwardBatch(net, &images[start_idx], batch_outputs, &labels[start_idx], 
                &batch_loss, current_batch_size, stream_idx, current_learning_rate,&total_kernel_time,&total_mem_time);
            
            total_loss += batch_loss;
            
            // Compute accuracy
            for (int i = 0; i < current_batch_size; i++) {
                int pred = 0;
                int actual = 0;
                for (int j = 0; j < OUTPUT_SIZE; j++) {
                    int idx = i * OUTPUT_SIZE + j;
                    if (batch_outputs[idx] > batch_outputs[i * OUTPUT_SIZE + pred])
                        pred = j;
                    if (labels[start_idx + i][j] > labels[start_idx + i][actual])
                        actual = j;
                }
                if (pred == actual)
                    correct++;
            }
        }
        
        CHECK_CUDA_ERROR(cudaEventRecord(epoch_stop));
        CHECK_CUDA_ERROR(cudaEventSynchronize(epoch_stop));
        float epoch_time = get_gpu_time(epoch_start, epoch_stop);
        
        printf("Epoch %d - Loss: %.4f - Train Accuracy: %.2f%% - Time: %.3fs\n",
            epoch + 1, total_loss / numImages, (correct / (double)numImages) * 100, epoch_time);
        
        CHECK_CUDA_ERROR(cudaEventDestroy(epoch_start));
        CHECK_CUDA_ERROR(cudaEventDestroy(epoch_stop));
    }
    
    CHECK_CUDA_ERROR(cudaEventRecord(stop));
    CHECK_CUDA_ERROR(cudaEventSynchronize(stop));
    float total_time = get_gpu_time(start, stop);
    
    printf("Total training time: %.3fs\n", total_time);
    
    // Clean up
    cudaFreeHost(batch_outputs);
    CHECK_CUDA_ERROR(cudaEventDestroy(start));
    CHECK_CUDA_ERROR(cudaEventDestroy(stop));
}
 
// Evaluate model accuracy
void evaluate(NeuralNetwork* net, double** images, double** labels, int numImages) {
    double* batch_outputs;
    CHECK_CUDA_ERROR(cudaMallocHost((void**)&batch_outputs, BATCH_SIZE * OUTPUT_SIZE * sizeof(double)));
    
    int correct = 0;
    int num_batches = (numImages + BATCH_SIZE - 1) / BATCH_SIZE;
    
    cudaEvent_t start, stop;
    CHECK_CUDA_ERROR(cudaEventCreate(&start));
    CHECK_CUDA_ERROR(cudaEventCreate(&stop));
    CHECK_CUDA_ERROR(cudaEventRecord(start));
    
    for (int batch = 0; batch < num_batches; batch++) {
        int start_idx = batch * BATCH_SIZE;
        int end_idx = start_idx + BATCH_SIZE;
        if (end_idx > numImages) end_idx = numImages;
        int current_batch_size = end_idx - start_idx;
        
        // Process batch using stream 0 for evaluation
        forwardBatch(net, &images[start_idx], batch_outputs, current_batch_size, 0,&total_kernel_time,&total_mem_time);
        
        // Compute accuracy
        for (int i = 0; i < current_batch_size; i++) {
            int pred = 0;
            int actual = 0;
            for (int j = 0; j < OUTPUT_SIZE; j++) {
                int idx = i * OUTPUT_SIZE + j;
                if (batch_outputs[idx] > batch_outputs[i * OUTPUT_SIZE + pred])
                    pred = j;
                if (labels[start_idx + i][j] > labels[start_idx + i][actual])
                    actual = j;
            }
            if (pred == actual)
                correct++;
        }
    }
    
    CHECK_CUDA_ERROR(cudaEventRecord(stop));
    CHECK_CUDA_ERROR(cudaEventSynchronize(stop));
    float eval_time = get_gpu_time(start, stop);
    
    printf("Test Accuracy: %.2f%% (Time: %.3fs)\n", 
           (correct / (double)numImages) * 100, eval_time);
    
    // Clean up
    cudaFreeHost(batch_outputs);
    CHECK_CUDA_ERROR(cudaEventDestroy(start));
    CHECK_CUDA_ERROR(cudaEventDestroy(stop));
}

// Read MNIST images with optimized reading
double** loadMNISTImages(const char* filename, int numImages) {
    FILE* file = fopen(filename, "rb");
    if (!file) {
        printf("Error opening %s\n", filename);
        exit(1);
    }
    
    // Skip file header (16 bytes)
    fseek(file, 16, SEEK_SET);
    
    // Use pinned memory for faster host-device transfers
    double** images = allocateHostPinnedMatrix(numImages, INPUT_SIZE);
    
    // Read all pixels at once for better I/O performance
    unsigned char* buffer = (unsigned char*)malloc(numImages * INPUT_SIZE);
    size_t items_read = fread(buffer, 1, numImages * INPUT_SIZE, file);
    if (items_read != numImages * INPUT_SIZE) {
        fprintf(stderr, "Error: Failed to read all image pixels (read %zu, expected %d)\n", 
                items_read, numImages * INPUT_SIZE);
        fclose(file);
        exit(EXIT_FAILURE);
    }
    
    // Convert to double in parallel (could be vectorized further with OpenMP)
    #pragma omp parallel for
    for (int i = 0; i < numImages; i++) {
        for (int j = 0; j < INPUT_SIZE; j++) {
            //images[i][j] = buffer[i * INPUT_SIZE + j] / 255.0;
            // In loadMNISTImages():
            images[i][j] = (buffer[i * INPUT_SIZE + j] / 255.0 - 0.1307) / 0.3081;  // MNIST mean/std
        }
    }
    
    free(buffer);
    fclose(file);
    return images;
}

// Read MNIST labels with optimized reading
double** loadMNISTLabels(const char* filename, int numLabels) {
    FILE* file = fopen(filename, "rb");
    if (!file) {
        printf("Error opening %s\n", filename);
        exit(1);
    }
    
    // Skip file header (8 bytes)
    fseek(file, 8, SEEK_SET);
    
    // Use pinned memory for faster host-device transfers
    double** labels = allocateHostPinnedMatrix(numLabels, OUTPUT_SIZE);
    
    // Read all labels at once
    unsigned char* buffer = (unsigned char*)malloc(numLabels);
    size_t items_read = fread(buffer, 1, numLabels, file);
    if (items_read != numLabels) {
        fprintf(stderr, "Error: Failed to read all labels (read %zu, expected %d)\n", 
                items_read, numLabels);
        fclose(file);
        exit(EXIT_FAILURE);
    }
    
    // Convert to one-hot encoding
    #pragma omp parallel for
    for (int i = 0; i < numLabels; i++) {
        for (int j = 0; j < OUTPUT_SIZE; j++) {
            labels[i][j] = (j == buffer[i]) ? 1.0 : 0.0;
        }
    }
    
    free(buffer);
    fclose(file);
    return labels;
}

// Free network memory and CUDA resources
void freeNetwork(NeuralNetwork* net) {
    // Free host memory
    freePinnedMatrix(net->W1, HIDDEN_SIZE);
    freePinnedMatrix(net->W2, OUTPUT_SIZE);
    cudaFreeHost(net->b1);
    cudaFreeHost(net->b2);
    
    // Free device memory
    CHECK_CUDA_ERROR(cudaFree(net->d_W1));
    CHECK_CUDA_ERROR(cudaFree(net->d_W2));
    CHECK_CUDA_ERROR(cudaFree(net->d_b1));
    CHECK_CUDA_ERROR(cudaFree(net->d_b2));
    CHECK_CUDA_ERROR(cudaFree(net->d_batch_input));
    CHECK_CUDA_ERROR(cudaFree(net->d_batch_hidden));
    CHECK_CUDA_ERROR(cudaFree(net->d_batch_output));
    CHECK_CUDA_ERROR(cudaFree(net->d_batch_target));
    CHECK_CUDA_ERROR(cudaFree(net->d_batch_d_output));
    CHECK_CUDA_ERROR(cudaFree(net->d_batch_d_hidden));
    
    // Destroy CUDA resources
    cublasDestroy(net->cublas_handle);
    for (int i = 0; i < NUM_STREAMS; i++) {
        CHECK_CUDA_ERROR(cudaStreamDestroy(net->streams[i]));
        CHECK_CUDA_ERROR(cudaEventDestroy(net->events[i]));
    }
    
    free(net);
}

// Main function
int main() {
    printf("=== MNIST Neural Network with Optimized CUDA Implementation ===\n\n");
    
    // Set device to first available GPU
    CHECK_CUDA_ERROR(cudaSetDevice(0));
    
    // Load MNIST dataset
    printf("Loading MNIST dataset...\n");
    double** train_images = loadMNISTImages("../../data/train-images.idx3-ubyte", IMAGE_SIZE_TRAIN);
    double** train_labels = loadMNISTLabels("../../data/train-labels.idx1-ubyte", IMAGE_SIZE_TRAIN);
    double** test_images = loadMNISTImages("../../data/t10k-images.idx3-ubyte", IMAGE_SIZE_TEST);
    double** test_labels = loadMNISTLabels("../../data/t10k-labels.idx1-ubyte", IMAGE_SIZE_TEST);
    printf("Dataset loaded successfully.\n\n");

    // Create and train neural network
    printf("Creating neural network...\n");
    NeuralNetwork* net = createNetwork();
    printf("Neural network created. Starting training...\n\n");
    
    // Train the network
    train(net, train_images, train_labels, IMAGE_SIZE_TRAIN);
    
    // Evaluate on test set
    printf("\nEvaluating on test set...\n");
    evaluate(net, test_images, test_labels, IMAGE_SIZE_TEST);

    appEnd = clock();
    double appTimeSec = get_time(appStart); // From your `get_time` function
    printf("\n--- Timing Summary ---\n");
    printf("Total Application Time (s): %.4f\n", appTimeSec);
    printf("Total Kernel Time (including cuBLAS): %.3fs\n", total_kernel_time);
    printf("Total Memory Transfer Time: %.3fs\n", total_mem_time);
    
    // Clean up
    printf("\nCleaning up resources...\n");
    freeNetwork(net);
    freePinnedMatrix(train_images, IMAGE_SIZE_TRAIN);
    freePinnedMatrix(train_labels, IMAGE_SIZE_TRAIN);
    freePinnedMatrix(test_images, IMAGE_SIZE_TEST);
    freePinnedMatrix(test_labels, IMAGE_SIZE_TEST);
    
    printf("Done.\n");
    return 0;
}