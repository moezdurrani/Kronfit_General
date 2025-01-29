import torch
import networkx as nx
import random
import time
import argparse


class KronFit:
    def __init__(self, graph, init_matrix, learning_rate, iterations, scale_matrix=True):
        self.graph = graph
        self.init_matrix = torch.tensor(init_matrix, dtype=torch.float64, requires_grad=True)
        self.learning_rate = learning_rate
        self.iterations = iterations
        self.scale_matrix = scale_matrix
        self.num_nodes = graph.number_of_nodes()
        self.num_edges = graph.number_of_edges()
        self.edges_list = list(graph.edges)

    def scale_initiator(self):
        """ Scale the initiator matrix to ensure it remains within a valid probability range. """
        with torch.no_grad():
            scale_factor = self.num_edges / self.init_matrix.sum()
            self.init_matrix *= scale_factor
            self.init_matrix /= self.init_matrix.max()
            self.init_matrix.clamp_(0, 1)  # Keep values in [0, 1]

    def compute_log_likelihood(self):
        """ Compute the log-likelihood of the graph given the current initiator matrix. """
        log_likelihood = torch.tensor(0.0, dtype=torch.float64, requires_grad=True)
        for src, dst in self.edges_list:
            row = src % self.init_matrix.shape[0]
            col = dst % self.init_matrix.shape[1]
            prob = torch.sigmoid(self.init_matrix[row, col])  # Ensure values are in probability range
            log_likelihood = log_likelihood + torch.log(prob)
        return log_likelihood

    def gradient_descent(self, min_step=0.005, max_step=0.05, samples=100000, threshold=1e-4):
        """ Perform gradient descent using PyTorch's autograd. """
        prev_log_likelihood = -torch.inf

        for i in range(self.iterations):
            print(f"{i + 1:03d}] SampleGradient: {samples}:")

            # Compute loss (negative log-likelihood for gradient calculation)
            loss = -self.compute_log_likelihood()  # Negate because we maximize likelihood
            loss.backward()  # Compute gradients using autograd

            # Parameter update using gradients
            with torch.no_grad():
                for row in range(self.init_matrix.shape[0]):
                    for col in range(self.init_matrix.shape[1]):
                        grad = self.init_matrix.grad[row, col]
                        adaptive_rate = self.learning_rate / (1.0 + abs(grad))
                        update = torch.clamp(adaptive_rate * grad, -max_step, max_step)
                        old_value = self.init_matrix[row, col].item()
                        self.init_matrix[row, col] -= update
                        print(
                            f"    {row * self.init_matrix.shape[1] + col}]  {self.init_matrix[row, col]:.6f}  <--  {old_value:.6f} +  {update:.6f}   Grad: {grad:.4f}   Rate: {adaptive_rate:.8f}")

                # Scale and normalize the matrix
                self.scale_initiator()
                current_log_likelihood = self.compute_log_likelihood().item()
                print(f"  Current Log-Likelihood: {current_log_likelihood:.4f}\n")

                # Check for convergence
                if abs(current_log_likelihood - prev_log_likelihood) < threshold:
                    print("Convergence reached based on log likelihood.")
                    break
                prev_log_likelihood = current_log_likelihood

                # Zero gradients for the next iteration
                self.init_matrix.grad.zero_()

        return self.init_matrix

    def fit(self):
        """ Fit the model using gradient descent and print the results. """
        start_time = time.time()
        self.scale_initiator()
        print("Initial Parameters:\n", self.init_matrix)

        fitted_parameters = self.gradient_descent()
        end_time = time.time()

        print("Fitted parameters:\n", fitted_parameters)
        print("Run Time: {:.2f} seconds".format(end_time - start_time))


def main():
    parser = argparse.ArgumentParser(description="Run Kronecker graph fitting.")
    parser.add_argument("file_path", type=str, help="Path to the edge list file")
    parser.add_argument("init_matrix", nargs=4, type=float, help="Four parameters for the initiator matrix separated with space")
    parser.add_argument("iterations", type=int, help="Number of iterations for fitting the model")
    args = parser.parse_args()

    G = nx.read_edgelist(args.file_path, nodetype=int, create_using=nx.DiGraph())
    init_matrix = [[args.init_matrix[0], args.init_matrix[1]], [args.init_matrix[2], args.init_matrix[3]]]
    learning_rate, iterations = 1e-5, args.iterations

    model = KronFit(G, init_matrix, learning_rate, iterations)
    model.fit()


if __name__ == "__main__":
    main()
