import numpy as np
import networkx as nx
import random
import time


class KronFit:
    def __init__(self, graph, init_matrix, scale_matrix=True):
        """
        Initialize the KronFit algorithm.
        :param graph: NetworkX graph object.
        :param init_matrix: Initial initiator matrix (NumPy array).
        :param scale_matrix: Whether to scale the initiator matrix to match the graph's edges.
        """
        self.graph = graph
        self.init_matrix = np.array(init_matrix, dtype=np.float64)
        self.scale_matrix = scale_matrix
        self.num_nodes = graph.number_of_nodes()
        self.num_edges = graph.number_of_edges()
        self.edges_list = list(graph.edges)

    def scale_initiator(self):
        """
        Scale the initiator matrix to match the number of edges in the graph.
        Ensures that values remain between 0 and 1 after scaling.
        """
        scale_factor = self.num_edges / np.sum(self.init_matrix)
        self.init_matrix *= scale_factor
        self.init_matrix /= np.max(self.init_matrix)  # Normalize to prevent overflow
        self.init_matrix = np.clip(self.init_matrix, 0, 1)  # Keep values in [0, 1]

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

    def gradient_descent(self, iterations=50, learning_rate=1e-5, min_step=0.005, max_step=0.05, warmup=10000, samples=100000):
        """
        Perform gradient descent to optimize the initiator matrix.
        """
        for i in range(iterations):
            # Initialize the gradient
            gradient = np.zeros_like(self.init_matrix)

            # Warm-up sampling
            if i == 0:
                print(f"Warm-up: Sampling {warmup} edges")
                _ = random.sample(self.edges_list, min(len(self.edges_list), warmup))

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

            # Apply gradient updates
            self.init_matrix += gradient

            # Ensure matrix values remain in a valid range
            self.init_matrix = np.clip(self.init_matrix, 0, 1)

            # Re-scale the initiator matrix to preserve edge counts
            self.scale_initiator()

            # Compute log-likelihood
            log_likelihood = self.compute_log_likelihood()

            # Print progress
            print(f"Iteration {i + 1}/{iterations}")
            print("Current matrix:\n", self.init_matrix)
            print(f"Log-likelihood: {log_likelihood}\n")

        return self.init_matrix

    def fit(self, iterations=50, learning_rate=1e-5, min_step=0.005, max_step=0.05, warmup=10000, samples=100000):
        """
        Fit the Kronecker model to the graph.
        """
        start_time = time.time()

        if self.scale_matrix:
            self.scale_initiator()

        print("Initial matrix:\n", self.init_matrix)

        # Perform gradient descent
        optimized_matrix = self.gradient_descent(iterations, learning_rate, min_step, max_step, warmup, samples)

        end_time = time.time()
        print(f"Optimized matrix:\n{optimized_matrix}")
        print(f"RunTime: {end_time - start_time:.2f} seconds")
        return optimized_matrix


# Example usage
if __name__ == "__main__":
    # Load graph
    G = nx.read_edgelist("../data/graph.txt", nodetype=int, create_using=nx.DiGraph())

    # Initial initiator matrix
    init_matrix = [
        [0.9, 0.7],
        [0.5, 0.2]
    ]

    # Run KronFit
    kronfit = KronFit(G, init_matrix)
    optimized_matrix = kronfit.fit(iterations=50, learning_rate=1e-5, samples=5000, warmup=10000)
