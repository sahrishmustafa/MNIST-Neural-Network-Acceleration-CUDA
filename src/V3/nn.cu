#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>  // For half-precision support
#include <device_launch_parameters.h>

#define INPUT_SIZE    784
#define HIDDEN_SIZE   128
#define OUTPUT_SIZE   10
#define LEARNING_RATE 0.01f
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

void softmax(double* x, int size) {
    double sum = 0;
    for (int i = 0; i < size; i++) {
        x[i] = exp(x[i]);
        sum += x[i];
    }
    for (int i = 0; i < size; i++) {
        x[i] /= sum;
    }
}

// -----------------------
// CUDA Kernel Definitions
// -----------------------

// Each thread computes one neuron in the hidden layer.
// Suggested kernel launch: <<<2, 64>>>
__global__ void forwardHidden(half* d_W1, half* d_b1, half* d_input, half* d_hidden) 
{
    // Hidden neuron index
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    // Shared memory tile for input
    __shared__ half s_input[INPUT_SIZE];

    // One-time cooperative load of input into shared memory
    for (int j = threadIdx.x; j < INPUT_SIZE; j += blockDim.x) {
        s_input[j] = d_input[j];
    }
    __syncthreads();

    if (i < HIDDEN_SIZE) {
        float sum = __half2float(d_b1[i]);  // Convert to float for accumulation
        
        // Access TRANSPOSED W1 with better memory coalescing
        // Now threads in a warp access consecutive memory locations for better performance
        for (int j = 0; j < INPUT_SIZE; j++) {
            sum += __half2float(d_W1[j * HIDDEN_SIZE + i]) * __half2float(s_input[j]);
        }
        
        // Convert back to half for storage
        d_hidden[i] = __float2half(fmaxf(0.0f, sum));  // ReLU in float precision
    }
}

// Each thread computes one neuron in the output layer.
__global__ void forwardOutput(half* d_W2, half* d_b2, half* d_hidden, half* d_output) 
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Shared memory for hidden layer activations
    __shared__ half s_hidden[HIDDEN_SIZE];
    
    // Collaborative loading of hidden layer values into shared memory
    // With OUTPUT_SIZE threads (typically 10 for MNIST), each thread loads multiple elements
    for (int j = threadIdx.x; j < HIDDEN_SIZE; j += blockDim.x) {
        s_hidden[j] = d_hidden[j];
    }
    
    // Wait for all threads to finish loading
    __syncthreads();
    
    if(i < OUTPUT_SIZE) {
        float sum = __half2float(d_b2[i]);  // Convert to float for accumulation
        for (int j = 0; j < HIDDEN_SIZE; j++) {
            sum += __half2float(d_W2[i * HIDDEN_SIZE + j]) * __half2float(s_hidden[j]);
        }
        d_output[i] = __float2half(sum);  // Convert back to half
    }
}

// Compute hidden gradient: d_dHidden = (W2^T * d_dOutput) * (hidden > 0)
__global__ void computeHiddenGradients(half* d_W2, half* d_dOutput, half* d_hidden, half* d_dHidden) 
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Create shared memory for d_dOutput (accessed by all threads)
    __shared__ half s_dOutput[OUTPUT_SIZE];
    
    // Collaboratively load d_dOutput into shared memory
    for (int j = threadIdx.x; j < OUTPUT_SIZE; j += blockDim.x) {
        s_dOutput[j] = d_dOutput[j];
    }
    
    // Ensure all threads have loaded the data
    __syncthreads();
    
    if(i < HIDDEN_SIZE) 
    {
        float sum = 0.0f;
        for (int j = 0; j < OUTPUT_SIZE; j++)
        {
            // Use shared memory instead of global memory
            sum += __half2float(d_W2[j * HIDDEN_SIZE + i]) * __half2float(s_dOutput[j]);
        }
        
        // ReLU derivative: 1 if input > 0, else 0
        float relu_deriv = (__half2float(d_hidden[i]) > 0.0f) ? 1.0f : 0.0f;
        d_dHidden[i] = __float2half(sum * relu_deriv);
    }
}

// Update output layer weights and biases
// Launch with OUTPUT_SIZE blocks and HIDDEN_SIZE threads per block.
__global__ void updateOutputLayer(half* d_W2, half* d_b2, half* d_dOutput, half* d_hidden) 
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    int totalWeights = OUTPUT_SIZE * HIDDEN_SIZE;
    if (idx < totalWeights) {
        int i = idx / HIDDEN_SIZE; // output neuron index
        int j = idx % HIDDEN_SIZE; // hidden neuron index
        
        // Compute weight update
        float weight = __half2float(d_W2[idx]);
        float dOutput = __half2float(d_dOutput[i]);
        float hidden = __half2float(d_hidden[j]);
        weight -= LEARNING_RATE * dOutput * hidden;
        d_W2[idx] = __float2half(weight);
    }

    // Optional: Bias update (if combined)
    int biasIdx = blockIdx.x * blockDim.x + threadIdx.x;
    if (biasIdx < OUTPUT_SIZE && threadIdx.x == 0) {
        float bias = __half2float(d_b2[biasIdx]);
        float dOutput = __half2float(d_dOutput[biasIdx]);
        bias -= LEARNING_RATE * dOutput;
        d_b2[biasIdx] = __float2half(bias);
    }
}

// Update hidden layer weights and biases
// Launch with HIDDEN_SIZE blocks and INPUT_SIZE threads per block.
__global__ void updateHiddenLayer(half* d_W1, half* d_b1, half* d_dHidden, half* d_input) 
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;  // Flattened thread index

    int totalWeights = HIDDEN_SIZE * INPUT_SIZE;
    if (idx < totalWeights) {
        // For transposed W1 format, we need to transpose indices
        int j = idx / HIDDEN_SIZE; // input neuron index (first dimension in transposed format)
        int i = idx % HIDDEN_SIZE; // hidden neuron index (second dimension in transposed format)
        
        // Compute weight update for transposed format
        float weight = __half2float(d_W1[idx]);
        float dHidden = __half2float(d_dHidden[i]);
        float input = __half2float(d_input[j]);
        weight -= LEARNING_RATE * dHidden * input;
        d_W1[idx] = __float2half(weight);
    }

    // Bias update remains the same
    int biasIdx = blockIdx.x * blockDim.x + threadIdx.x;
    if (biasIdx < HIDDEN_SIZE && threadIdx.x == 0) {
        float bias = __half2float(d_b1[biasIdx]);
        float dHidden = __half2float(d_dHidden[biasIdx]);
        bias -= LEARNING_RATE * dHidden;
        d_b1[biasIdx] = __float2half(bias);
    }
}

// -----------------------
// Neural Network Structure (Parameters reside on device)
// -----------------------
// Modified network structure to indicate transposed W1
typedef struct 
{
    // Device pointers for parameters (flattened arrays)
    half* d_W1; // [INPUT_SIZE x HIDDEN_SIZE] - TRANSPOSED for better memory access
    half* d_W2; // [OUTPUT_SIZE x HIDDEN_SIZE]
    half* d_b1; // [HIDDEN_SIZE]
    half* d_b2; // [OUTPUT_SIZE]
} NeuralNetwork;

// -----------------------
// Create and Initialize Network (Parameters on Device)
// -----------------------
NeuralNetwork* createNetwork() 
{
    NeuralNetwork* net = (NeuralNetwork*)malloc(sizeof(NeuralNetwork));
    
    // Allocate device memory for weights and biases using half precision
    cudaMalloc((void**)&net->d_W1, HIDDEN_SIZE * INPUT_SIZE * sizeof(half));
    cudaMalloc((void**)&net->d_W2, OUTPUT_SIZE * HIDDEN_SIZE * sizeof(half));
    cudaMalloc((void**)&net->d_b1, HIDDEN_SIZE * sizeof(half));
    cudaMalloc((void**)&net->d_b2, OUTPUT_SIZE * sizeof(half));
    
    // Allocate temporary host arrays to initialize parameters (still in double)
    double* h_W1 = (double*)malloc(HIDDEN_SIZE * INPUT_SIZE * sizeof(double));
    double* h_W2 = (double*)malloc(OUTPUT_SIZE * HIDDEN_SIZE * sizeof(double));
    double* h_b1 = (double*)calloc(HIDDEN_SIZE, sizeof(double));
    double* h_b2 = (double*)calloc(OUTPUT_SIZE, sizeof(double));
    
    // Initialize weights with small random values
    srand(time(NULL));
    for (int i = 0; i < HIDDEN_SIZE * INPUT_SIZE; i++) {
        h_W1[i] = ((double)rand() / RAND_MAX) * 0.01;
    }
    for (int i = 0; i < OUTPUT_SIZE * HIDDEN_SIZE; i++) {
        h_W2[i] = ((double)rand() / RAND_MAX) * 0.01;
    }
    
    // Convert from double to half for device transfer
    half* h_W1_half = (half*)malloc(HIDDEN_SIZE * INPUT_SIZE * sizeof(half));
    half* h_W2_half = (half*)malloc(OUTPUT_SIZE * HIDDEN_SIZE * sizeof(half));
    half* h_b1_half = (half*)malloc(HIDDEN_SIZE * sizeof(half));
    half* h_b2_half = (half*)malloc(OUTPUT_SIZE * sizeof(half));
    
    // TRANSPOSITION: Store W1 in transposed format for better memory access
    for (int i = 0; i < HIDDEN_SIZE; i++) {
        for (int j = 0; j < INPUT_SIZE; j++) {
            // Store in transposed format: [j][i] instead of [i][j]
            h_W1_half[j * HIDDEN_SIZE + i] = __float2half((float)h_W1[i * INPUT_SIZE + j]);
        }
    }
    
    for (int i = 0; i < OUTPUT_SIZE * HIDDEN_SIZE; i++) {
        h_W2_half[i] = __float2half((float)h_W2[i]);
    }
    for (int i = 0; i < HIDDEN_SIZE; i++) {
        h_b1_half[i] = __float2half((float)h_b1[i]);
    }
    for (int i = 0; i < OUTPUT_SIZE; i++) {
        h_b2_half[i] = __float2half((float)h_b2[i]);
    }
    
    // Copy initialized parameters to device
    cudaMemcpy(net->d_W1, h_W1_half, HIDDEN_SIZE * INPUT_SIZE * sizeof(half), cudaMemcpyHostToDevice);
    cudaMemcpy(net->d_W2, h_W2_half, OUTPUT_SIZE * HIDDEN_SIZE * sizeof(half), cudaMemcpyHostToDevice);
    cudaMemcpy(net->d_b1, h_b1_half, HIDDEN_SIZE * sizeof(half), cudaMemcpyHostToDevice);
    cudaMemcpy(net->d_b2, h_b2_half, OUTPUT_SIZE * sizeof(half), cudaMemcpyHostToDevice);
    
    // Free temporary arrays
    free(h_W1); free(h_W2); free(h_b1); free(h_b2);
    free(h_W1_half); free(h_W2_half); free(h_b1_half); free(h_b2_half);
    
    return net;
}

// -----------------------
// Modified Forward Pass on GPU
// -----------------------
// The input (d_input) is a pointer to one image (of size INPUT_SIZE) already on the device.
void forwardGPU(NeuralNetwork* net, half* d_input, double* hidden, double* output, half* d_hidden_temp, half* d_output_temp) 
{
    size_t size_hidden = HIDDEN_SIZE * sizeof(half);
    size_t size_output = OUTPUT_SIZE * sizeof(half);
    
    // Run forward kernels on the provided d_input and temporary d_hidden_temp/d_output_temp.
    forwardHidden<<<4, 32>>>(net->d_W1, net->d_b1, d_input, d_hidden_temp);
    
    cudaDeviceSynchronize();
    
    forwardOutput<<<1, 64>>>(net->d_W2, net->d_b2, d_hidden_temp, d_output_temp);
    cudaDeviceSynchronize();
    
    // Temporary arrays for conversion from half to double
    half* h_hidden_half = (half*)malloc(size_hidden);
    half* h_output_half = (half*)malloc(size_output);
    
    // Copy intermediate and final results back to host for loss calculation and accuracy.
    cudaMemcpy(h_hidden_half, d_hidden_temp, size_hidden, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_output_half, d_output_temp, size_output, cudaMemcpyDeviceToHost);
    
    // Convert from half to double for host-side processing
    for (int i = 0; i < HIDDEN_SIZE; i++) {
        hidden[i] = (double)__half2float(h_hidden_half[i]);
    }
    for (int i = 0; i < OUTPUT_SIZE; i++) {
        output[i] = (double)__half2float(h_output_half[i]);
    }
    
    free(h_hidden_half);
    free(h_output_half);
    
    softmax(output, OUTPUT_SIZE);
}

// -----------------------
// Modified Backward Pass on GPU
// -----------------------
// Instead of expecting the target label on the GPU, we pass the host target label (h_target).
// The function computes the output gradient on the host and then copies it to the device.
void backwardGPU(NeuralNetwork* net, half* d_input, half* d_hidden, half* d_output, double* h_target) 
{
    half *d_dOutput, *d_dHidden;
    size_t size_output = OUTPUT_SIZE * sizeof(half);
    size_t size_hidden = HIDDEN_SIZE * sizeof(half);
    
    cudaMalloc((void**)&d_dOutput, size_output);
    cudaMalloc((void**)&d_dHidden, size_hidden);
    
    // Compute output gradient on host:
    double h_dOutput[OUTPUT_SIZE];
    // Copy the forward pass output from device to host.
    double h_output[OUTPUT_SIZE];
    
    // Temporary array for half-precision output
    half* h_output_half = (half*)malloc(size_output);
    cudaMemcpy(h_output_half, d_output, size_output, cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
    
    // Convert from half to double
    for (int i = 0; i < OUTPUT_SIZE; i++) {
        h_output[i] = (double)__half2float(h_output_half[i]);
    }
    free(h_output_half);
    
    // Compute gradients (still in double)
    for (int i = 0; i < OUTPUT_SIZE; i++) {
        h_dOutput[i] = h_output[i] - h_target[i];
    }
    
    // Convert gradients to half for device transfer
    half* h_dOutput_half = (half*)malloc(size_output);
    for (int i = 0; i < OUTPUT_SIZE; i++) {
        h_dOutput_half[i] = __float2half((float)h_dOutput[i]);
    }
    
    // Copy computed output gradients to device.
    cudaMemcpy(d_dOutput, h_dOutput_half, size_output, cudaMemcpyHostToDevice);
    free(h_dOutput_half);
    
    // Now compute hidden gradient on the device using the computed d_dOutput.
    computeHiddenGradients<<<16, 4>>>(net->d_W2, d_dOutput, d_hidden, d_dHidden);
    cudaDeviceSynchronize();
    
    int totalWeights = OUTPUT_SIZE * HIDDEN_SIZE;
    int threadsPerBlock = 256;
    int blocks = (totalWeights + threadsPerBlock - 1) / threadsPerBlock;
    // Update output layer parameters using d_dOutput.
    updateOutputLayer<<<blocks, threadsPerBlock>>>(net->d_W2, net->d_b2, d_dOutput, d_hidden);
    cudaDeviceSynchronize();
    
    totalWeights = HIDDEN_SIZE * INPUT_SIZE;
    threadsPerBlock = 256;
    blocks = (totalWeights + threadsPerBlock - 1) / threadsPerBlock;
    
    // Update hidden layer parameters using d_dHidden.
    updateHiddenLayer<<<blocks, threadsPerBlock>>>(net->d_W1, net->d_b1, d_dHidden, d_input);
    cudaDeviceSynchronize();
    
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

// -----------------------
// Training and Evaluation (Host Side)
// -----------------------
// For training, we now use the labels stored on the host (h_train_labels) rather than on the GPU.
void train(NeuralNetwork* net, half* d_train_images, double* h_train_labels, int numImages) {
    // Allocate temporary device memory for an image's forward pass computation.
    half *d_hidden_temp, *d_output_temp;
    size_t size_hidden = HIDDEN_SIZE * sizeof(half);
    size_t size_output = OUTPUT_SIZE * sizeof(half);
    
    cudaMalloc((void**)&d_hidden_temp, size_hidden);
    cudaMalloc((void**)&d_output_temp, size_output);
    
    clock_t total_start = clock();
    for (int epoch = 0; epoch < EPOCHS; epoch++) {
        clock_t epoch_start = clock();
        double loss = 0.0;
        int correct = 0;
        double hidden[HIDDEN_SIZE], output[OUTPUT_SIZE];
        
        for (int i = 0; i < numImages; i++) {
            // Get pointer for i-th image from device memory.
            half* d_image_i = d_train_images + i * INPUT_SIZE;
            // Get pointer for i-th label from host (flattened labels remain on host).
            double* h_label_i = h_train_labels + i * OUTPUT_SIZE;
            
            // Forward pass (d_image_i is already on the device)
            forwardGPU(net, d_image_i, hidden, output, d_hidden_temp, d_output_temp);
            
            // Backward pass: use the host label directly.
            backwardGPU(net, d_image_i, d_hidden_temp, d_output_temp, h_label_i);
            
            // Compute loss and accuracy on host
            double sample_loss = 0.0;
            for (int k = 0; k < OUTPUT_SIZE; k++)
                sample_loss -= h_label_i[k] * log(output[k] + 1e-8);  // add epsilon to avoid log(0)
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
    
    cudaFree(d_hidden_temp);
    cudaFree(d_output_temp);
}

// For evaluation, we similarly use the labels on the host.
void evaluate(NeuralNetwork* net, half* d_test_images, double* h_test_labels, int numImages) {
    half *d_hidden_temp, *d_output_temp;
    size_t size_hidden = HIDDEN_SIZE * sizeof(half);
    size_t size_output = OUTPUT_SIZE * sizeof(half);
    
    cudaMalloc((void**)&d_hidden_temp, size_hidden);
    cudaMalloc((void**)&d_output_temp, size_output);
    
    int correct = 0;
    double hidden[HIDDEN_SIZE], output[OUTPUT_SIZE];
    for (int i = 0; i < numImages; i++){
        half* d_image_i = d_test_images + i * INPUT_SIZE;
        forwardGPU(net, d_image_i, hidden, output, d_hidden_temp, d_output_temp);
        
        int pred = 0, actual = 0;
        double* h_label_i = h_test_labels + i * OUTPUT_SIZE;
        for (int j = 0; j < OUTPUT_SIZE; j++){
            if (output[j] > output[pred]) pred = j;
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
void freeNetwork(NeuralNetwork* net) 
{
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
    printf("MNIST Neural Network - Optimized GPU Version (V3)\n(Update: Launch Configurations)\n(Update: Shared Memory Usage)\n(Update: Occupancy Updated)\n(Update: Communication Optimized)\n");
    printf("(Update: Using half-precision (FP16) for increased performance)\n(Update: Coalesced the strided memory accesses)\n\n");

    // Load the entire dataset on the host (2D arrays)
    double** train_images = loadMNISTImages("../../data/train-images.idx3-ubyte", NUM_TRAIN);
    double** train_labels = loadMNISTLabels("../../data/train-labels.idx1-ubyte", NUM_TRAIN);
    double** test_images  = loadMNISTImages("../../data/t10k-images.idx3-ubyte", NUM_TEST);
    double** test_labels  = loadMNISTLabels("../../data/t10k-labels.idx1-ubyte", NUM_TEST);

    // Flatten the host arrays into contiguous 1D arrays (still in double)
    double* h_train_images_flat = flatten2D(train_images, NUM_TRAIN, INPUT_SIZE);
    double* h_train_labels_flat = flatten2D(train_labels,  NUM_TRAIN, OUTPUT_SIZE);
    double* h_test_images_flat  = flatten2D(test_images,  NUM_TEST, INPUT_SIZE);
    double* h_test_labels_flat  = flatten2D(test_labels,  NUM_TEST, OUTPUT_SIZE);

    // Convert images from double to half for GPU processing
    half* h_train_images_half = (half*)malloc(NUM_TRAIN * INPUT_SIZE * sizeof(half));
    half* h_test_images_half = (half*)malloc(NUM_TEST * INPUT_SIZE * sizeof(half));
    
    for (int i = 0; i < NUM_TRAIN * INPUT_SIZE; i++) {
        h_train_images_half[i] = __float2half((float)h_train_images_flat[i]);
    }
    for (int i = 0; i < NUM_TEST * INPUT_SIZE; i++) {
        h_test_images_half[i] = __float2half((float)h_test_images_flat[i]);
    }

    // Allocate device memory for the half-precision image dataset
    half *d_train_images, *d_test_images;
    cudaMalloc((void**)&d_train_images, NUM_TRAIN * INPUT_SIZE * sizeof(half));
    cudaMalloc((void**)&d_test_images,  NUM_TEST  * INPUT_SIZE * sizeof(half));

    // Copy half-precision data to device
    cudaMemcpy(d_train_images, h_train_images_half, NUM_TRAIN * INPUT_SIZE * sizeof(half), cudaMemcpyHostToDevice);
    cudaMemcpy(d_test_images,  h_test_images_half,  NUM_TEST  * INPUT_SIZE * sizeof(half), cudaMemcpyHostToDevice);

    // Free the flattened image arrays and original 2D arrays if no longer needed.
    free(h_train_images_flat);
    free(h_test_images_flat);
    free(h_train_images_half);
    free(h_test_images_half);
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