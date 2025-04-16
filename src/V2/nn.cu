#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#define INPUT_SIZE    784
#define HIDDEN_SIZE   128
#define OUTPUT_SIZE   10
#define LEARNING_RATE 0.01
#define EPOCHS        3
#define BATCH_SIZE    64
#define NUM_CLASSES   10  // Digits 0-9
#define NUM_TRAIN     60000
#define NUM_TEST      10000

float kernelTime = 0.0f;
float memcpyTime = 0.0f;
clock_t appStart, appEnd;
cudaEvent_t kStart, kStop, mStart, mStop;


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
__global__ void forwardHidden(double* d_W1, double* d_b1, double* d_input, double* d_hidden) {
    int i = threadIdx.x;
    if(i < HIDDEN_SIZE) {
        double sum = d_b1[i];
        for (int j = 0; j < INPUT_SIZE; j++){
            sum += d_W1[i * INPUT_SIZE + j] * d_input[j];
        }
        d_hidden[i] = (sum > 0) ? sum : 0;  // ReLU activation
    }
}

// Each thread computes one neuron in the output layer.
__global__ void forwardOutput(double* d_W2, double* d_b2, double* d_hidden, double* d_output) {
    int i = threadIdx.x;
    if(i < OUTPUT_SIZE) {
        double sum = d_b2[i];
        for (int j = 0; j < HIDDEN_SIZE; j++){
            sum += d_W2[i * HIDDEN_SIZE + j] * d_hidden[j];
        }
        d_output[i] = sum;  // Softmax will be applied on host later
    }
}

// Compute hidden gradient: d_dHidden = (W2^T * d_dOutput) * (hidden > 0)
// (This kernel remains unchanged.)
__global__ void computeHiddenGradients(double* d_W2, double* d_dOutput, double* d_hidden, double* d_dHidden) {
    int i = threadIdx.x;
    if(i < HIDDEN_SIZE) {
        double sum = 0.0;
        for (int j = 0; j < OUTPUT_SIZE; j++){
            sum += d_W2[j * HIDDEN_SIZE + i] * d_dOutput[j];
        }
        d_dHidden[i] = (d_hidden[i] > 0) ? sum : 0.0;
    }
}

// Update output layer weights and biases
// Launch with OUTPUT_SIZE blocks and HIDDEN_SIZE threads per block.
__global__ void updateOutputLayer(double* d_W2, double* d_b2, double* d_dOutput, double* d_hidden) {
    int i = blockIdx.x;      // output neuron index
    int j = threadIdx.x;     // hidden neuron index
    if(i < OUTPUT_SIZE && j < HIDDEN_SIZE) {
        d_W2[i * HIDDEN_SIZE + j] -= LEARNING_RATE * d_dOutput[i] * d_hidden[j];
    }
    if(j == 0 && i < OUTPUT_SIZE) {
        d_b2[i] -= LEARNING_RATE * d_dOutput[i];
    }
}

// Update hidden layer weights and biases
// Launch with HIDDEN_SIZE blocks and INPUT_SIZE threads per block.
__global__ void updateHiddenLayer(double* d_W1, double* d_b1, double* d_dHidden, double* d_input) {
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
typedef struct {
    // Device pointers for parameters (flattened arrays)
    double* d_W1; // [HIDDEN_SIZE x INPUT_SIZE]
    double* d_W2; // [OUTPUT_SIZE x HIDDEN_SIZE]
    double* d_b1; // [HIDDEN_SIZE]
    double* d_b2; // [OUTPUT_SIZE]
} NeuralNetwork;

// -----------------------
// Create and Initialize Network (Parameters on Device)
// -----------------------
NeuralNetwork* createNetwork() {
    NeuralNetwork* net = (NeuralNetwork*)malloc(sizeof(NeuralNetwork));
    
    // Allocate device memory for weights and biases
    cudaMalloc((void**)&net->d_W1, HIDDEN_SIZE * INPUT_SIZE * sizeof(double));
    cudaMalloc((void**)&net->d_W2, OUTPUT_SIZE * HIDDEN_SIZE * sizeof(double));
    cudaMalloc((void**)&net->d_b1, HIDDEN_SIZE * sizeof(double));
    cudaMalloc((void**)&net->d_b2, OUTPUT_SIZE * sizeof(double));
    
    // Allocate temporary host arrays to initialize parameters
    double* h_W1 = (double*)malloc(HIDDEN_SIZE * INPUT_SIZE * sizeof(double));
    double* h_W2 = (double*)malloc(OUTPUT_SIZE * HIDDEN_SIZE * sizeof(double));
    double* h_b1 = (double*)calloc(HIDDEN_SIZE, sizeof(double));
    double* h_b2 = (double*)calloc(OUTPUT_SIZE, sizeof(double));
    
    srand(time(NULL));
    for (int i = 0; i < HIDDEN_SIZE * INPUT_SIZE; i++) {
        h_W1[i] = ((double)rand() / RAND_MAX) * 0.01;
    }
    for (int i = 0; i < OUTPUT_SIZE * HIDDEN_SIZE; i++) {
        h_W2[i] = ((double)rand() / RAND_MAX) * 0.01;
    }
    
    cudaEventRecord(mStart);
    // Copy initialized parameters to device
    cudaMemcpy(net->d_W1, h_W1, HIDDEN_SIZE * INPUT_SIZE * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(net->d_W2, h_W2, OUTPUT_SIZE * HIDDEN_SIZE * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(net->d_b1, h_b1, HIDDEN_SIZE * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(net->d_b2, h_b2, OUTPUT_SIZE * sizeof(double), cudaMemcpyHostToDevice);
    cudaEventRecord(mStop);
    cudaEventSynchronize(mStop);
    float tempMemcpyTime;
    cudaEventElapsedTime(&tempMemcpyTime, mStart, mStop);
    memcpyTime += tempMemcpyTime;
    
    free(h_W1); free(h_W2); free(h_b1); free(h_b2);
    
    return net;
}

// -----------------------
// Modified Forward Pass on GPU
// -----------------------
// The input (d_input) is a pointer to one image (of size INPUT_SIZE) already on the device.
void forwardGPU(NeuralNetwork* net, double* d_input, double* hidden, double* output, 
                double* d_hidden_temp, double* d_output_temp) {
    size_t size_hidden = HIDDEN_SIZE * sizeof(double);
    size_t size_output = OUTPUT_SIZE * sizeof(double);
    
    // Run forward kernels on the provided d_input and temporary d_hidden_temp/d_output_temp.
    cudaEventRecord(kStart);
    forwardHidden<<<1, HIDDEN_SIZE>>>(net->d_W1, net->d_b1, d_input, d_hidden_temp);
    cudaDeviceSynchronize();
    cudaEventRecord(kStop);
    cudaEventSynchronize(kStop);
    float tempKernelTime;   
    cudaEventElapsedTime(&tempKernelTime, kStart, kStop);
    kernelTime += tempKernelTime;
    
    cudaEventRecord(kStart);
    forwardOutput<<<1, OUTPUT_SIZE>>>(net->d_W2, net->d_b2, d_hidden_temp, d_output_temp);
    cudaDeviceSynchronize();
    cudaEventRecord(kStop);
    cudaEventSynchronize(kStop);
    cudaEventElapsedTime(&tempKernelTime, kStart, kStop);
    kernelTime += tempKernelTime;
    
    // Copy intermediate and final results back to host for loss calculation and accuracy.
    cudaEventRecord(mStart);
    cudaMemcpy(hidden, d_hidden_temp, size_hidden, cudaMemcpyDeviceToHost);
    cudaMemcpy(output, d_output_temp, size_output, cudaMemcpyDeviceToHost);
    cudaEventRecord(mStop);
    cudaEventSynchronize(mStop);
    float tempMemcpyTime;
    cudaEventElapsedTime(&tempMemcpyTime, mStart, mStop);
    memcpyTime += tempMemcpyTime;
    
    softmax(output, OUTPUT_SIZE);
}

// -----------------------
// Modified Backward Pass on GPU
// -----------------------
// Instead of expecting the target label on the GPU, we pass the host target label (h_target).
// The function computes the output gradient on the host and then copies it to the device.
void backwardGPU(NeuralNetwork* net, double* d_input, double* d_hidden, double* h_output_softmax, double* h_target) {
    double *d_dOutput, *d_dHidden;
    size_t size_output = OUTPUT_SIZE * sizeof(double);
    size_t size_hidden = HIDDEN_SIZE * sizeof(double);
    
    cudaMalloc((void**)&d_dOutput, size_output);
    cudaMalloc((void**)&d_dHidden, size_hidden);
    
    // Compute output gradient on host using softmax outputs.
    double h_dOutput[OUTPUT_SIZE];
    for (int i = 0; i < OUTPUT_SIZE; i++) {
        h_dOutput[i] = h_output_softmax[i] - h_target[i];
    }
    // Copy computed output gradients to device.
    cudaEventRecord(mStart);
    cudaMemcpy(d_dOutput, h_dOutput, size_output, cudaMemcpyHostToDevice);
    cudaDeviceSynchronize();
    cudaEventRecord(mStop);
    cudaEventSynchronize(mStop);
    float tempMemcpyTime;
    cudaEventElapsedTime(&tempMemcpyTime, mStart, mStop);
    memcpyTime += tempMemcpyTime;

    // Now compute hidden gradient on the device using the computed d_dOutput.
    cudaEventRecord(kStart);
    computeHiddenGradients<<<1, HIDDEN_SIZE>>>(net->d_W2, d_dOutput, d_hidden, d_dHidden);
    cudaDeviceSynchronize();
    cudaEventRecord(kStop);
    cudaEventSynchronize(kStop);  
    float tempKernelTime;
    cudaEventElapsedTime(&tempKernelTime, kStart, kStop);
    kernelTime += tempKernelTime;
    
    
    // Update output layer parameters using d_dOutput.
    cudaEventRecord(kStart);
    updateOutputLayer<<<OUTPUT_SIZE, HIDDEN_SIZE>>>(net->d_W2, net->d_b2, d_dOutput, d_hidden);
    cudaDeviceSynchronize();
    cudaEventRecord(kStop);
    cudaEventSynchronize(kStop);  
    cudaEventElapsedTime(&tempKernelTime, kStart, kStop);
    kernelTime += tempKernelTime;
    
    // Update hidden layer parameters using d_dHidden.
    cudaEventRecord(kStart);
    updateHiddenLayer<<<HIDDEN_SIZE, INPUT_SIZE>>>(net->d_W1, net->d_b1, d_dHidden, d_input);
    cudaDeviceSynchronize();
    cudaEventRecord(kStop);
    cudaEventSynchronize(kStop);  
    cudaEventElapsedTime(&tempKernelTime, kStart, kStop);
    kernelTime += tempKernelTime;
    
    cudaFree(d_dOutput);
    cudaFree(d_dHidden);
}

// -----------------------
// Utility: Flatten 2D host matrix into contiguous 1D array
// -----------------------
double* flatten2D(double** mat, int rows, int cols) {
    double* flat = (double*)malloc(rows * cols * sizeof(double));
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            flat[i * cols + j] = mat[i][j];
        }
    }
    return flat;
}

// -----------------------
// Training and Evaluation (Host Side)
// -----------------------
// For training, we now use the labels stored on the host (h_train_labels) rather than on the GPU.
void train(NeuralNetwork* net, double* d_train_images, double* h_train_labels, int numImages) {
    // Allocate temporary device memory for an image's forward pass computation.
    double *d_input_temp, *d_hidden_temp, *d_output_temp;
    size_t size_input = INPUT_SIZE * sizeof(double);
    size_t size_hidden = HIDDEN_SIZE * sizeof(double);
    size_t size_output = OUTPUT_SIZE * sizeof(double);
    
    cudaMalloc((void**)&d_input_temp, size_input);
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
            double* d_image_i = d_train_images + i * INPUT_SIZE;
            // Get pointer for i-th label from host (flattened labels remain on host).
            double* h_label_i = h_train_labels + i * OUTPUT_SIZE;
            
            // Forward pass (d_image_i is already on the device)
            forwardGPU(net, d_image_i, hidden, output, d_hidden_temp, d_output_temp);
            
            // Backward pass: use the host label directly.
            backwardGPU(net, d_image_i, d_hidden_temp, output, h_label_i);
            
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
    
    cudaFree(d_input_temp);
    cudaFree(d_hidden_temp);
    cudaFree(d_output_temp);
}

// For evaluation, we similarly use the labels on the host.
void evaluate(NeuralNetwork* net, double* d_test_images, double* h_test_labels, int numImages) {
    double *d_hidden_temp, *d_output_temp;
    size_t size_hidden = HIDDEN_SIZE * sizeof(double);
    size_t size_output = OUTPUT_SIZE * sizeof(double);
    
    cudaMalloc((void**)&d_hidden_temp, size_hidden);
    cudaMalloc((void**)&d_output_temp, size_output);
    
    int correct = 0;
    double hidden[HIDDEN_SIZE], output[OUTPUT_SIZE];
    for (int i = 0; i < numImages; i++){
        double* d_image_i = d_test_images + i * INPUT_SIZE;
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
int main() {
    cudaEventCreate(&kStart); cudaEventCreate(&kStop);
    cudaEventCreate(&mStart); cudaEventCreate(&mStop);

    printf("MNIST Neural Network - Naive GPU Version (V2)\n\n");

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

    // Allocate device memory for the entire image dataset only (labels remain on host)
    double *d_train_images, *d_test_images;
    cudaMalloc((void**)&d_train_images, NUM_TRAIN * INPUT_SIZE * sizeof(double));
    cudaMalloc((void**)&d_test_images,  NUM_TEST  * INPUT_SIZE * sizeof(double));

    cudaEventRecord(mStart);
    cudaMemcpy(d_train_images, h_train_images_flat, NUM_TRAIN * INPUT_SIZE * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_test_images,  h_test_images_flat,  NUM_TEST  * INPUT_SIZE * sizeof(double), cudaMemcpyHostToDevice);
    cudaEventRecord(mStop);
    cudaEventSynchronize(mStop);
    float tempMemcpyTime;
    cudaEventElapsedTime(&tempMemcpyTime, mStart, mStop);
    memcpyTime += tempMemcpyTime;

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

    appEnd = clock();
    double appTimeSec = get_time(appStart); // From your `get_time` function
    
    printf("\n--- Timing Summary ---\n");
    printf("Total Application Time (s): %.4f\n", appTimeSec);
    printf("Total Memcpy Time (ms): %.4f\n", memcpyTime);
    printf("Total Kernel Time (ms): %.4f\n", kernelTime);
    printf("Application-Level Time (s) = Kernel + Memcpy: %.4f\n", (kernelTime + memcpyTime) / 1000.0);
    
    return 0;
}
