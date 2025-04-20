#include <cstring>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

#define INPUT_SIZE 784
#define HIDDEN_SIZE 128
#define OUTPUT_SIZE 10
#define LEARNING_RATE 0.01
#define EPOCHS 3
#define BATCH_SIZE 64
#define NUM_CLASSES 10

double kernel_time = 0.0;

double get_time(clock_t start) {
    return (double)(clock() - start) / CLOCKS_PER_SEC;
}

double* allocateMatrix(int rows, int cols) {
    double* mat = (double*)malloc(rows * cols * sizeof(double));
    return mat;
}

void freeMatrix(double** mat, int rows) {
    for (int i = 0; i < rows; i++) {
        free(mat[i]);
    }
    free(mat);
}

void freeMatrix(double* mat) {
    free(mat);
}

void relu(double* x, int size) {
    #pragma acc parallel loop present(x)
    for (int i = 0; i < size; i++) {
        x[i] = (x[i] > 0) ? x[i] : 0;
    }
}

void softmax(double* x, int size) {
    double sum = 0;
    #pragma acc parallel loop present(x) reduction(+:sum)
    for (int i = 0; i < size; i++) {
        x[i] = exp(x[i]);
        sum += x[i];
    }
    #pragma acc parallel loop present(x)
    for (int i = 0; i < size; i++) {
        x[i] /= sum;
    }
}

typedef struct {
    double* W1;
    double* W2;
    double* b1;
    double* b2;
} NeuralNetwork;

NeuralNetwork* createNetwork() {
    NeuralNetwork* net = (NeuralNetwork*)malloc(sizeof(NeuralNetwork));
    net->W1 = allocateMatrix(HIDDEN_SIZE, INPUT_SIZE);
    net->W2 = allocateMatrix(OUTPUT_SIZE, HIDDEN_SIZE);
    net->b1 = (double*)calloc(HIDDEN_SIZE, sizeof(double));
    net->b2 = (double*)calloc(OUTPUT_SIZE, sizeof(double));

    srand(time(NULL));
    for (int i = 0; i < HIDDEN_SIZE; i++)
        for (int j = 0; j < INPUT_SIZE; j++)
            net->W1[i * INPUT_SIZE + j] = ((double)rand() / RAND_MAX) * 0.01;

    for (int i = 0; i < OUTPUT_SIZE; i++)
        for (int j = 0; j < HIDDEN_SIZE; j++)
            net->W2[i * HIDDEN_SIZE + j] = ((double)rand() / RAND_MAX) * 0.01;

    #pragma acc enter data copyin(net[0:1], net->W1[0:HIDDEN_SIZE*INPUT_SIZE], \
        net->W2[0:OUTPUT_SIZE*HIDDEN_SIZE], net->b1[0:HIDDEN_SIZE], net->b2[0:OUTPUT_SIZE])

    return net;
}

void forward(NeuralNetwork* net, double* input, double* hidden, double* output) {
    clock_t kernel_start = clock();

    #pragma acc parallel loop present(net, net->W1, net->b1, input) copyout(hidden[0:HIDDEN_SIZE])
    for (int i = 0; i < HIDDEN_SIZE; i++) {
        hidden[i] = net->b1[i];
        for (int j = 0; j < INPUT_SIZE; j++)
            hidden[i] += net->W1[i * INPUT_SIZE + j] * input[j];
    }
    relu(hidden, HIDDEN_SIZE);

    #pragma acc parallel loop present(net, net->W2, net->b2, hidden) copyout(output[0:OUTPUT_SIZE])
    for (int i = 0; i < OUTPUT_SIZE; i++) {
        output[i] = net->b2[i];
        for (int j = 0; j < HIDDEN_SIZE; j++)
            output[i] += net->W2[i * HIDDEN_SIZE + j] * hidden[j];
    }
    softmax(output, OUTPUT_SIZE);
    kernel_time += get_time(kernel_start);
}

void backward(NeuralNetwork* net, double* input, double* hidden, double* output, double* target, double* d_output, double* d_hidden) {
    clock_t kernel_start = clock();

    #pragma acc parallel loop present(output, target, d_output)
    for (int i = 0; i < OUTPUT_SIZE; i++)
    d_output[i] = output[i] - target[i];

    #pragma acc parallel loop present(net, d_output, hidden, d_hidden)
    for (int i = 0; i < HIDDEN_SIZE; i++) {
    double sum = 0.0;
    for (int j = 0; j < OUTPUT_SIZE; j++)
    sum += net->W2[j * HIDDEN_SIZE + i] * d_output[j];
    d_hidden[i] = sum * (hidden[i] > 0);
    }

    #pragma acc parallel loop collapse(2) present(net, d_output, hidden)
    for (int i = 0; i < OUTPUT_SIZE; i++) {
    for (int j = 0; j < HIDDEN_SIZE; j++)
    net->W2[i * HIDDEN_SIZE + j] -= LEARNING_RATE * d_output[i] * hidden[j];
    }

    #pragma acc parallel loop collapse(2) present(net, d_hidden, input)
    for (int i = 0; i < HIDDEN_SIZE; i++) {
    for (int j = 0; j < INPUT_SIZE; j++)
    net->W1[i * INPUT_SIZE + j] -= LEARNING_RATE * d_hidden[i] * input[j];
    }

    #pragma acc parallel loop present(net, d_output)
    for (int i = 0; i < OUTPUT_SIZE; i++)
    net->b2[i] -= LEARNING_RATE * d_output[i];

    #pragma acc parallel loop present(net, d_hidden)
    for (int i = 0; i < HIDDEN_SIZE; i++)
    net->b1[i] -= LEARNING_RATE * d_hidden[i];

    kernel_time += get_time(kernel_start);
}


void train(NeuralNetwork* net, double* images, double* labels, int numImages) {
    clock_t total_start = clock();
    #pragma acc enter data copyin(images[0:numImages*INPUT_SIZE], labels[0:numImages*OUTPUT_SIZE])

    // Allocate memory for d_output and d_hidden on the host
    double* d_output = (double*) malloc(sizeof(double) * OUTPUT_SIZE);
    double* d_hidden = (double*) malloc(sizeof(double) * HIDDEN_SIZE);

    // Enter data into OpenACC for copying to the device
    #pragma acc enter data copyin(d_output[0:OUTPUT_SIZE], d_hidden[0:HIDDEN_SIZE])
    
    
    for (int epoch = 0; epoch < EPOCHS; epoch++) {
        clock_t epoch_start = clock();
        double loss = 0.0;
        int correct = 0;
        
        for (int i = 0; i < numImages; i++) {
            double hidden[HIDDEN_SIZE], output[OUTPUT_SIZE];
            double* input = &images[i * INPUT_SIZE];
            double* target = &labels[i * OUTPUT_SIZE];
            
            #pragma acc enter data copyin(hidden[0:HIDDEN_SIZE], output[0:OUTPUT_SIZE])
            #pragma acc data copyin(input[0:INPUT_SIZE], target[0:OUTPUT_SIZE]) \
                copyout(hidden[0:HIDDEN_SIZE], output[0:OUTPUT_SIZE])
            {
                forward(net, input, hidden, output);
                backward(net, input, hidden, output, target, d_output, d_hidden);
            }

            #pragma acc update host(output[0:OUTPUT_SIZE], target[0:OUTPUT_SIZE])
            for (int k = 0; k < OUTPUT_SIZE; k++)
                loss -= target[k] * log(output[k]);
            int pred = 0, actual = 0;
            for (int j = 0; j < OUTPUT_SIZE; j++) {
                if (output[j] > output[pred]) pred = j;
                if (target[j] > target[actual]) actual = j;
            }
            if (pred == actual) correct++;
        }

        printf("Epoch %d - Loss: %.4f - Train Accuracy: %.2f%% - Time: %.3fs\n",
               epoch + 1, loss / numImages, (correct / (double)numImages) * 100, get_time(epoch_start));
    }

    // Free allocated memory
    free(d_output);
    free(d_hidden);

    printf("Total training time: %.3fs\n", get_time(total_start));
    #pragma acc exit data delete(images[0:numImages*INPUT_SIZE], labels[0:numImages*OUTPUT_SIZE])
    #pragma acc exit data delete(d_output[0:OUTPUT_SIZE], d_hidden[0:HIDDEN_SIZE])
}



void evaluate(NeuralNetwork* net, double* images, double* labels, int numImages) {
    int correct = 0;
    #pragma acc enter data copyin(images[0:numImages*INPUT_SIZE], labels[0:numImages*OUTPUT_SIZE])

    for (int i = 0; i < numImages; i++) {
        double hidden[HIDDEN_SIZE], output[OUTPUT_SIZE];
        double* input = &images[i * INPUT_SIZE];
        double* target = &labels[i * OUTPUT_SIZE];

        #pragma acc data copyin(input[0:INPUT_SIZE]) copyout(hidden[0:HIDDEN_SIZE], output[0:OUTPUT_SIZE])
        forward(net, input, hidden, output);

        #pragma acc update host(output[0:OUTPUT_SIZE], target[0:OUTPUT_SIZE])
        int pred = 0, actual = 0;
        for (int j = 0; j < OUTPUT_SIZE; j++) {
            if (output[j] > output[pred]) pred = j;
            if (target[j] > target[actual]) actual = j;
        }
        if (pred == actual) correct++;
    }
    printf("Test Accuracy: %.2f%%\n", (correct / (double)numImages) * 100);
    #pragma acc exit data delete(images[0:numImages*INPUT_SIZE], labels[0:numImages*OUTPUT_SIZE])
}

double** loadMNISTImages(const char* filename, int numImages) {
    FILE* file = fopen(filename, "rb");
    fseek(file, 16, SEEK_SET);
    double* data = (double*)malloc(numImages * INPUT_SIZE * sizeof(double));
    for (int i = 0; i < numImages; i++) {
        for (int j = 0; j < INPUT_SIZE; j++) {
            unsigned char pixel;
            fread(&pixel, 1, 1, file);
            data[i * INPUT_SIZE + j] = pixel / 255.0;
        }
    }
    fclose(file);
    double** images = (double**)malloc(numImages * sizeof(double*));
    for (int i = 0; i < numImages; i++)
        images[i] = &data[i * INPUT_SIZE];
    return images;
}

double** loadMNISTLabels(const char* filename, int numLabels) {
    FILE* file = fopen(filename, "rb");
    fseek(file, 8, SEEK_SET);
    double* data = (double*)malloc(numLabels * OUTPUT_SIZE * sizeof(double));
    for (int i = 0; i < numLabels; i++) {
        unsigned char label;
        fread(&label, 1, 1, file);
        for (int j = 0; j < OUTPUT_SIZE; j++)
            data[i * OUTPUT_SIZE + j] = (j == label) ? 1.0 : 0.0;
    }
    fclose(file);
    double** labels = (double**)malloc(numLabels * sizeof(double*));
    for (int i = 0; i < numLabels; i++)
        labels[i] = &data[i * OUTPUT_SIZE];
    return labels;
}

void freeNetwork(NeuralNetwork* net) {
    #pragma acc exit data delete(net->W1[0:HIDDEN_SIZE*INPUT_SIZE], \
        net->W2[0:OUTPUT_SIZE*HIDDEN_SIZE], net->b1[0:HIDDEN_SIZE], net->b2[0:OUTPUT_SIZE])
    freeMatrix(net->W1);
    freeMatrix(net->W2);
    free(net->b1);
    free(net->b2);
    free(net);
}

int main() {
    printf("MNIST Neural Network\n\n");

    double** train_images = loadMNISTImages("../data/train-images.idx3-ubyte", 60000);
    double** train_labels = loadMNISTLabels("../data/train-labels.idx1-ubyte", 60000);
    double** test_images = loadMNISTImages("../data/t10k-images.idx3-ubyte", 10000);
    double** test_labels = loadMNISTLabels("../data/t10k-labels.idx1-ubyte", 10000);

    NeuralNetwork* net = createNetwork();
    clock_t app_start = clock();

    train(net, train_images[0], train_labels[0], 60000);
    evaluate(net, test_images[0], test_labels[0], 10000);

    double app_time = get_time(app_start);
    printf("\n--- Final Report ---\n");
    printf("Application-Level Time: %.3fs\n", app_time);
    printf("Kernel Time: %.3fs\n", kernel_time);
    printf("---------------------\n");

    freeNetwork(net);
    //freeMatrix(train_images, 60000);
    free(train_images);
    //freeMatrix(train_labels, 60000);
    free(train_labels);
    //freeMatrix(test_images, 10000);
    free(test_images);
    //freeMatrix(test_labels, 10000);
    free(test_labels);
    return 0;
}