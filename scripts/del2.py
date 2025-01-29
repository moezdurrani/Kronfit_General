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
        k = int(np.ceil(np.log2(self.num_nodes)))  # Number of Kronecker iterations
        scale_factor = (self.num_edges / (np.sum(self.init_matrix) ** k)) ** (1 / k)
        self.init_matrix *= scale_factor

    def compute_log_likelihood(self, matrix):
        """
        Compute the log-likelihood of the graph given the current Kronecker matrix.
        """
        k = int(np.ceil(np.log2(self.num_nodes)))  # Number of Kronecker iterations
        likelihood = 0

        # Log-likelihood for existing edges
        for edge in self.graph.edges():
            src, dst = edge
            prob = 1.0
            for i in range(k):
                src_idx = (src >> (k - i - 1)) & 1
                dst_idx = (dst >> (k - i - 1)) & 1
                prob *= matrix[src_idx][dst_idx]
            likelihood += np.log(prob)

        # Log-likelihood for non-edges (sampling-based approximation)
        non_edge_sample_size = min(len(self.graph.edges()), 1000)
        non_edges = set()
        while len(non_edges) < non_edge_sample_size:
            src = random.randint(0, self.num_nodes - 1)
            dst = random.randint(0, self.num_nodes - 1)
            if not self.graph.has_edge(src, dst):
                non_edges.add((src, dst))

        for src, dst in non_edges:
            prob = 1.0
            for i in range(k):
                src_idx = (src >> (k - i - 1)) & 1
                dst_idx = (dst >> (k - i - 1)) & 1
                prob *= matrix[src_idx][dst_idx]
            likelihood += np.log(1 - prob)

        return likelihood

    def gradient_descent(self, iterations=50, learning_rate=0.01, min_step=0.005, max_step=0.05,
                         warmup=10000, samples=100000):
        """
        Perform gradient descent to optimize the initiator matrix.
        """
        k = int(np.ceil(np.log2(self.num_nodes)))
        best_likelihood = float('-inf')
        best_matrix = None

        for i in range(iterations):
            gradient = np.zeros_like(self.init_matrix)

            # Compute gradient for existing edges
            for edge in self.graph.edges():
                src, dst = edge
                for l in range(k):
                    src_idx = (src >> (k - l - 1)) & 1
                    dst_idx = (dst >> (k - l - 1)) & 1
                    prob = self.init_matrix[src_idx][dst_idx]
                    gradient[src_idx][dst_idx] += 1.0 / (prob * k)

            # Compute gradient for non-edges (sampling-based)
            non_edge_sample_size = min(len(self.graph.edges()), 1000)
            sampled_non_edges = set()
            while len(sampled_non_edges) < non_edge_sample_size:
                src = random.randint(0, self.num_nodes - 1)
                dst = random.randint(0, self.num_nodes - 1)
                if not self.graph.has_edge(src, dst):
                    sampled_non_edges.add((src, dst))

            for src, dst in sampled_non_edges:
                for l in range(k):
                    src_idx = (src >> (k - l - 1)) & 1
                    dst_idx = (dst >> (k - l - 1)) & 1
                    prob = self.init_matrix[src_idx][dst_idx]
                    gradient[src_idx][dst_idx] -= 1.0 / ((1 - prob) * k)

            # Update matrix using gradient descent
            step = learning_rate * gradient
            step = np.clip(step, -max_step, max_step)
            self.init_matrix += step

            # Ensure matrix values stay in valid probability range
            self.init_matrix = np.clip(self.init_matrix, 0.01, 0.99)

            # Calculate log-likelihood
            current_likelihood = self.compute_log_likelihood(self.init_matrix)

            # Keep track of best matrix
            if current_likelihood > best_likelihood:
                best_likelihood = current_likelihood
                best_matrix = self.init_matrix.copy()

            # Print progress
            if (i + 1) % 5 == 0:
                print(f"Iteration {i + 1}/{iterations}")
                print("Current matrix:\n", self.init_matrix)
                print(f"Log-likelihood: {current_likelihood}\n")

        self.init_matrix = best_matrix
        return self.init_matrix

    def fit(self, iterations=50, learning_rate=0.01, min_step=0.005, max_step=0.05,
            warmup=10000, samples=100000):
        """
        Fit the Kronecker model to the graph.
        """
        if self.scale_matrix:
            self.scale_initiator()

        print("Initial matrix:\n", self.init_matrix)
        print("Graph info:")
        print(f"Nodes: {self.num_nodes}")
        print(f"Edges: {self.num_edges}")
        print(f"Kronecker iterations: {int(np.ceil(np.log2(self.num_nodes)))}\n")

        optimized_matrix = self.gradient_descent(
            iterations, learning_rate, min_step, max_step, warmup, samples
        )

        print("\nOptimized matrix:\n", optimized_matrix)
        return optimized_matrix

if __name__ == "__main__":
    # Load graph
    G = nx.read_edgelist("../data/graph.txt", nodetype=int, create_using=nx.DiGraph())

    # Initial initiator matrix
    init_matrix = np.array([
        [0.9, 0.7],
        [0.5, 0.2]
    ])

    # Initialize and run KronFit
    kronfit = KronFit(G, init_matrix)
    optimized_matrix = kronfit.fit()