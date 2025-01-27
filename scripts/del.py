import numpy as np
import networkx as nx
import random


class KronFit:
    def __init__(self, graph, init_matrix, perm_method='d', scale_matrix=True):
        """
        Initialize the KronFit algorithm.
        :param graph: NetworkX graph object.
        :param init_matrix: Initial initiator matrix (NumPy array).
        :param perm_method: Node permutation method ('d'=degree, 'r'=random, 'o'=order).
        :param scale_matrix: Whether to scale the initiator matrix to match the graph's edges.
        """
        self.graph = graph
        self.init_matrix = init_matrix
        self.perm_method = perm_method
        self.scale_matrix = scale_matrix
        self.num_nodes = graph.number_of_nodes()
        self.num_edges = graph.number_of_edges()

    def scale_initiator(self):
        """
        Scale the initiator matrix to match the number of edges in the graph.
        """
        scale_factor = self.num_edges / np.sum(self.init_matrix)
        self.init_matrix *= scale_factor

    def compute_log_likelihood(self, matrix):
        """
        Compute the log-likelihood of the graph given the current Kronecker matrix.
        """
        likelihood = 0
        for edge in self.graph.edges:
            src, dst = edge
            prob = matrix[src % len(matrix)][dst % len(matrix)]
            likelihood += np.log(prob)
        return likelihood

    def gradient_descent(self, iterations=50, learning_rate=1e-5, min_step=0.005, max_step=0.05, warmup=10000,
                         samples=100000):
        """
        Perform gradient descent to optimize the initiator matrix and print progress at each iteration.
        """
        for i in range(iterations):
            gradient = np.zeros_like(self.init_matrix)
            for _ in range(samples):
                # Sample an edge from the graph
                src, dst = random.choice(list(self.graph.edges))
                # Compute the Kronecker probability for the sampled edge
                prob = self.init_matrix[src % len(self.init_matrix)][dst % len(self.init_matrix)]
                # Compute the gradient for the sampled edge
                gradient[src % len(self.init_matrix)][dst % len(self.init_matrix)] += (1 / prob)

            # Normalize the gradient
            gradient /= samples

            # Update the matrix using gradient descent
            self.init_matrix += learning_rate * gradient

            # Clip the matrix to ensure valid probabilities
            self.init_matrix = np.clip(self.init_matrix, 0.01, 0.99)

            # Calculate log-likelihood for this iteration
            log_likelihood = self.compute_log_likelihood(self.init_matrix)

            # Print progress
            print(f"Iteration {i + 1}/{iterations}")
            print("Current matrix:\n", self.init_matrix)
            print(f"Log-likelihood: {log_likelihood}\n")

        return self.init_matrix

    def fit(self, iterations=50, learning_rate=1e-5, min_step=0.005, max_step=0.05, warmup=10000, samples=100000):
        """
        Fit the Kronecker model to the graph.
        """
        if self.scale_matrix:
            self.scale_initiator()

        print("Initial matrix:\n", self.init_matrix)

        # Gradient Descent Optimization
        optimized_matrix = self.gradient_descent(
            iterations, learning_rate, min_step, max_step, warmup, samples
        )

        print("Optimized matrix:\n", optimized_matrix)
        return optimized_matrix


# Example usage
if __name__ == "__main__":
    # Load graph
    G = nx.read_edgelist("../data/graph2.txt", nodetype=int, create_using=nx.DiGraph())

    # Initial initiator matrix
    init_matrix = np.array([
        [0.9, 0.7],
        [0.5, 0.2]
    ])

    # Initialize and run KronFit
    kronfit = KronFit(G, init_matrix)
    optimized_matrix = kronfit.fit()