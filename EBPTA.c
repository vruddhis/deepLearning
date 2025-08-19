#include <stdio.h>
#include <math.h> //for exp


double sigmoid(double x) 
{return 1.0 / (1.0 + exp(-x));}

double derivative_sigmoid(double y) 
{return y * (1.0 - y); }

// C = A multiplied by B
void matrixMultiply(int rowsA, int colsA, int colsB, double A[rowsA][colsA], double B[colsA][colsB], double C[rowsA][colsB]) 
{
	for (int i = 0; i < rowsA; i++) {
		for (int j = 0; j < colsB; j++) {
			C[i][j] = 0;
			for (int k = 0; k < colsA; k++) {
				C[i][j] += A[i][k] * B[k][j];
			}
		}
	}
}

// x: input vector (input_size)
// W: weights matrix (num_neurons X input_size) FLAT ARRAY
// b: bias vector (num_neurons)
// output: output vector (num_neurons)
void fullyConnectedLayer(const double *x, int input_size, const double *W, int num_neurons, const double *b, double *output) 
{
	for (int i = 0; i < num_neurons; i++) {
		double z = b[i];
		for (int j = 0; j < input_size; j++) {
			z += W[i * input_size + j] * x[j];
		}
		output[i] = sigmoid(z);
	}
}

// half MSE
double computeError(const double *target, const double *output, int size) 
{
    double error = 0.0;
    for (int i = 0; i < size; i++) {
        double diff = target[i] - output[i];
        error += diff * diff;
    }
    return error / 2.0;  
}

// output layer delta: (target - y) * sigmoid'(y)
void computeOutputDelta(const double *target, const double *output, double *delta, int size) 
{
    for (int i = 0; i < size; i++) {
        double error = target[i] - output[i];
        delta[i] = error * derivative_sigmoid(output[i]);
    }
}

// hidden layer delta: (W^T * delta_next) * sigmoid'(y_hidden)
void computeHiddenDelta(const double *hidden_output, const double *delta_next, const double *W_next, int hidden_size, int output_size, double *delta_hidden) 
{
    for (int i = 0; i < hidden_size; i++) {
        double sum = 0.0;
        for (int j = 0; j < output_size; j++) {
            sum += W_next[j * hidden_size + i] * delta_next[j];
        }
        delta_hidden[i] = sum * derivative_sigmoid(hidden_output[i]);
    }
}

// W[i][j] += alpha * delta[i] * input[j]
// b[i]   += alpha * delta[i]
void updateWeights(double *W, double *b, const double *input, const double *delta, int num_neurons, int input_size, double alpha) 
{
    for (int i = 0; i < num_neurons; i++) {
        for (int j = 0; j < input_size; j++) {
            W[i * input_size + j] += alpha * delta[i] * input[j];
        }
        b[i] += alpha * delta[i];
    }
}

// forward pass for n fully connected layers
// x: input vector (input_size)
// W: array of pointers to weight matrices
// b: array of pointers to bias vectors
// layer_sizes: array of neuron counts per layer
// num_layers: number of layers
// outputs: array of pointers to output vectors for each layer
void forwardPassN(const double *x, int input_size, const double **W, const double **b, const int *layer_sizes, int num_layers, double **outputs) 
{
    const double *current_input = x;
    int current_input_size = input_size;
    for (int l = 0; l < num_layers; l++) {
        fullyConnectedLayer(current_input, current_input_size,
                           W[l], layer_sizes[l], b[l], outputs[l]);
        current_input = outputs[l];
        current_input_size = layer_sizes[l];
    }
}

// x: input vector (input_size)
// target: target output vector (layer_sizes[num_layers-1])
// W: array of pointers to weight matrices (flat matrices)
// b: array of pointers to bias vectors
// layer_sizes: array of neuron counts per layer
// num_layers: number of layers
// outputs: array of pointers to output vectors for each layer
// alpha: learning rate
void backwardPassN(const double *x, const double *target, double **W, double **b, const int *layer_sizes, int num_layers, double **outputs, double alpha, int input_size) 
{
    double *deltas[num_layers]; // we are gonna store the derivatives for each layer here
    for (int l = 0; l < num_layers; l++) {
        deltas[l] = (double*)malloc(layer_sizes[l] * sizeof(double));
    }

    int output_layer = num_layers - 1;
    computeOutputDelta(target, outputs[output_layer], deltas[output_layer], layer_sizes[output_layer]);


    for (int l = num_layers - 2; l >= 0; l--) 
    {
        computeHiddenDelta(outputs[l], deltas[l+1], W[l+1], layer_sizes[l], layer_sizes[l+1], deltas[l]);
    }


    for (int l = 0; l < num_layers; l++) {
        const double *input_vec;
        int input_size_val;
        if (l == 0) {
            input_vec = x;
            input_size_val = input_size;
        } else {
            input_vec = outputs[l-1];
            input_size_val = layer_sizes[l-1];
        }
        updateWeights(W[l], b[l], input_vec, deltas[l], layer_sizes[l], input_size_val, alpha);
    }

    // free deltas
    for (int l = 0; l < num_layers; l++) {
        free(deltas[l]);
    }
}

//T is targets
void train(double **W, double **b, const double **X, const double **T, int num_samples, int input_size, const int *layer_sizes, int num_layers, double alpha, int epochs)
{
    double *outputs[num_layers];
    for (int l = 0; l < num_layers; l++) {
        outputs[l] = (double*)malloc(layer_sizes[l] * sizeof(double));
    }

    for (int epoch = 0; epoch < epochs; epoch++) {
        double total_error = 0.0;

        for (int s = 0; s < num_samples; s++) {
            forwardPassN(X[s], input_size, (const double**)W, (const double**)b, layer_sizes, num_layers, outputs);
            total_error = computeError(T[s], outputs[num_layers - 1], layer_sizes[num_layers - 1]);
            backwardPassN(X[s], T[s], W, b, layer_sizes, num_layers, outputs, alpha, input_size);
        }

        printf("Epoch %d, Error = %f\n", epoch, total_error);
    }

    for (int l = 0; l < num_layers; l++) {
        free(outputs[l]);
    }
}
