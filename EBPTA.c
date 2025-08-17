#include <stdio.h>

double sigmoid(double x) 
{return 1.0 / (1.0 + exp(-x));}

double derivative_sigmoid(double y) 
{return y * (1.0 - y); }

//C = A multiplied by B
void matrixMultiply(int rowsA, int colsA, int colsB, 
					double A[rowsA][colsA],
					double B[colsA][colsB],
					double C[rowsA][colsB]) {
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
// W: weights matrix (num_neurons X input_size)
// b: bias vector (num_neurons)
// output: output vector (num_neurons)
void fullyConnectedLayer(const double *x, int input_size,
						 const double *W, int num_neurons,
						 const double *b,
						 double *output) {
	for (int i = 0; i < num_neurons; i++) {
		double z = b[i];
		for (int j = 0; j < input_size; j++) {
			z += W[i * input_size + j] * x[j];
		}
		output[i] = sigmoid(z);
	}
                         }

// half MSE
double computeError(const double *target, const double *output, int size) {
    double error = 0.0;
    for (int i = 0; i < size; i++) {
        double diff = target[i] - output[i];
        error += diff * diff;
    }
    return error / 2.0;  
}

//output layer delta: (target - y) * sigmoid'(y)
void computeOutputDelta(const double *target, const double *output,
                        double *delta, int size) {
    for (int i = 0; i < size; i++) {
        double error = target[i] - output[i];
        delta[i] = error * derivative_sigmoid(output[i]);
    }
}

//hidden layer delta: (W^T * delta_next) * sigmoid'(y_hidden)
void computeHiddenDelta(const double *hidden_output, const double *delta_next,
                        const double *W_next, int hidden_size, int output_size,
                        double *delta_hidden) {
    for (int i = 0; i < hidden_size; i++) {
        double sum = 0.0;
        for (int j = 0; j < output_size; j++) {
            sum += W_next[j * hidden_size + i] * delta_next[j];
        }
        delta_hidden[i] = sum * derivative_sigmoid(hidden_output[i]);
    }
}

//W[i][j] += alpha * delta[i] * input[j]
//b[i]   += alpha * delta[i]
void updateWeights(double *W, double *b, const double *input,
                   const double *delta, int num_neurons, int input_size,
                   double alpha) {
    for (int i = 0; i < num_neurons; i++) {
        for (int j = 0; j < input_size; j++) {
            W[i * input_size + j] += alpha * delta[i] * input[j];
        }
        b[i] += alpha * delta[i];
    }
}


