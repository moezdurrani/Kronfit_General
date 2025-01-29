import numpy as np
import networkx as nx
import random
import time
import argparse


class KronFit:
    def __init__(self, graph, init_matrix, learning_rate, iterations, scale_matrix=True):
        self.graph = graph
        self.init_matrix = init_matrix
        self.learning_rate = learning_rate
        self.iterations = iterations

        self.scale_matrix = scale_matrix
        self.num_nodes = graph.number_of_nodes()
        self.num_edges = graph.number_of_edges()
        self.edges_list = list(graph.edges)

    def scale_initiator(self):
        scale_factor = self.num_edges / np.sum(self.init_matrix)
        self.init_matrix *= scale_factor
        self.init_matrix /= np.max(self.init_matrix)  # Normalize to prevent overflow
        self.init_matrix = np.clip(self.init_matrix, 0, 1)  # Keep values in [0, 1]

    def compute_log_likelihood(self):
        likelihood = 0
        for edge in self.edges_list:
            src, dst = edge
            prob = self.init_matrix[src % self.init_matrix.shape[0], dst % self.init_matrix.shape[1]]
            if prob > 0:
                likelihood += np.log(prob)
        return likelihood

    def compute_gradient(self, samples=100000):
        gradient = np.zeros_like(self.init_matrix)
        sampled_edges = random.sample(self.edges_list, min(samples, len(self.edges_list)))

        # Detailed trace data
        grad_details = []

        for src, dst in sampled_edges:
            # mapping nodes from the graph onto the initiator matrix
            row = src % self.init_matrix.shape[0]
            col = dst % self.init_matrix.shape[1]
            prob = self.init_matrix[row, col]

            if prob > 0:
                grad_contrib = 1 / prob
                gradient[row, col] += grad_contrib
                # Collect detailed gradient contribution data
                grad_details.append((row, col, grad_contrib))

        gradient /= samples
        return gradient, grad_details

    def gradient_descent(self, learning_rate, min_step, max_step, warmup, samples, threshold=1e-4):
        prev_log_likelihood = -np.inf
        for i in range(self.iterations):
            print(f"{i + 1:03d}] SampleGradient: {samples}:")

            gradient, grad_details = self.compute_gradient(samples)

            for row in range(self.init_matrix.shape[0]):
                for col in range(self.init_matrix.shape[1]):
                    adaptive_rate = learning_rate / (1.0 + abs(gradient[row, col]))
                    update = np.clip(adaptive_rate * gradient[row, col], -max_step, max_step)
                    old_value = self.init_matrix[row, col]
                    self.init_matrix[row, col] += update
                    print(
                        f"    {row * self.init_matrix.shape[1] + col}]  {self.init_matrix[row, col]:.6f}  <--  {old_value:.6f} +  {update:.6f}   Grad: {gradient[row, col]:.4f}   Rate: {adaptive_rate:.8f}")

            self.scale_initiator()
            current_log_likelihood = self.compute_log_likelihood()
            print(f"  Current Log-Likelihood: {current_log_likelihood:.4f}")

            # Check for convergence using the change in log likelihood
            if abs(current_log_likelihood - prev_log_likelihood) < threshold:
                print("Convergence reached based on log likelihood.")
                break
            prev_log_likelihood = current_log_likelihood

        return self.init_matrix

    def fit(self):
        # Main fitting loop
        start_time = time.time()

        if self.scale_matrix:
            self.scale_initiator()

        print("Initial Parameters:\n", self.init_matrix)

        # Capture the optimized initiator matrix returned from gradient descent
        fitted_parameters = self.gradient_descent(self.learning_rate, min_step=0.005, max_step=0.05, warmup=10000,
                                                  samples=100000)

        end_time = time.time()
        print(f"Fitted parameters:\n{fitted_parameters}")  # Now should display the fitted matrix
        print(f"Run Time: {end_time - start_time:.2f} seconds")


def main():
    parser = argparse.ArgumentParser(description="Run Kronecker graph fitting.")
    parser.add_argument("file_path", type=str, help="Path to the edge list file")
    parser.add_argument("init_matrix", nargs=4, type=float,
                        help="Four parameters for the initiator matrix as space-separated values")
    parser.add_argument("iterations", type=int, help="Number of iterations for fitting the model")

    args = parser.parse_args()

    G = nx.read_edgelist(args.file_path, nodetype=int, create_using=nx.DiGraph())
    init_matrix = np.array([[args.init_matrix[0], args.init_matrix[1]], [args.init_matrix[2], args.init_matrix[3]]])
    learning_rate, iterations = 1e-5, args.iterations
    model = KronFit(G, init_matrix, learning_rate, iterations)
    model.fit()


if __name__ == "__main__":
    main()
