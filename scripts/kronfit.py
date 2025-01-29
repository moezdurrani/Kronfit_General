import numpy as np
import networkx as nx
import random
import time

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

    def kronecker_product(self, A, B):
        return np.kron(A, B)

    def compute_log_likelihood(self):
        """
        Compute the log-likelihood of the graph given the current Kronecker matrix.
        """
        likelihood = 0
        for edge in self.edges_list:
            src, dst = edge
            prob = self.init_matrix[src % self.init_matrix.shape[0], dst % self.init_matrix.shape[1]]
            if prob > 0:
                likelihood += np.log(prob)
        return likelihood

    def compute_gradient(self, learning_rate, min_step, max_step=0.05, warmup=10000, samples=100000):
        # Initialize the gradient
        gradient = np.zeros_like(self.init_matrix)

        # Sample edges for gradient computation
        sampled_edges = random.sample(self.edges_list, min(samples, len(self.edges_list)))

        # Compute gradient for each sampled edge
        for src, dst in sampled_edges:
            row = src % self.init_matrix.shape[0]
            col = dst % self.init_matrix.shape[1]
            prob = self.init_matrix[row, col]

            if prob > 0:
                gradient[row, col] += 1 / prob

        # Normalize the gradient
        gradient /= samples

        # Scale the gradient using an adaptive learning rate
        adaptive_rate = learning_rate / (1.0 + np.abs(gradient))
        gradient = np.clip(gradient * adaptive_rate, -max_step, max_step)

        # Re-scale the initiator matrix to preserve edge counts
        self.scale_initiator()

        return gradient

    def gradient_descent(self, learning_rate, min_step, max_step, warmup, samples):
        # Apply a gradient descent step
        for i in range(self.iterations):
            # Warm-up sampling (only print, not used)
            if i == 0:
                print(f"Warm-up: Sampling {warmup} edges")

            # Compute the gradient
            grad = self.compute_gradient(learning_rate, min_step, max_step, warmup, samples)

            # Update the initiator matrix
            self.init_matrix -= self.learning_rate * grad

            # Compute log-likelihood
            log_likelihood = self.compute_log_likelihood()

            # Print progress
            print(f"Iteration {i + 1}/{self.iterations}")
            print("Current matrix:\n", self.init_matrix)
            print(f"Log-likelihood: {log_likelihood}\n")

        # Return the optimized initiator matrix
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
    G = nx.read_edgelist("../data/graph.txt", nodetype=int, create_using=nx.DiGraph())
    init_matrix = np.array([[0.9, 0.7], [0.5, 0.2]])
    learning_rate, iterations = 1e-5, 100
    model = KronFit(G, init_matrix, learning_rate, iterations)
    model.fit()

if __name__ == "__main__":
    main()
