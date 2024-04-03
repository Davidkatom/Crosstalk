class QubitSystem:
    def __init__(self, num_qubits):
        self.num_qubits = num_qubits
        self.edges = [(i, i+1) for i in range(num_qubits - 1)]  # Linear chain

    def find_min_experiments(self):
        experiments = []  # List to store experiments (flip, measure)

        # Greedy algorithm: iterate over edges, selecting experiments
        # that cover new edges without violating constraints.
        for edge in self.edges:
            # Check if edge is already covered by existing experiments
            if not self.is_edge_covered(edge, experiments):
                # Add an experiment that covers the edge without violating constraints
                possible_experiments = self.possible_experiments_for_edge(edge)
                for exp in possible_experiments:
                    if self.is_experiment_valid(exp, experiments):
                        experiments.append(exp)
                        break

        return experiments

    def is_edge_covered(self, edge, experiments):
        # An edge is covered if there's an experiment flipping one end and measuring the other
        return any(exp[0] == edge[0] and exp[1] == edge[1] or exp[0] == edge[1] and exp[1] == edge[0] for exp in experiments)

    def possible_experiments_for_edge(self, edge):
        # For a given edge, return both possible experiments (directions of flip-measure)
        return [(edge[0], edge[1]), (edge[1], edge[0])]

    def is_experiment_valid(self, new_experiment, existing_experiments):
        # Check if adding the new_experiment violates the constraint
        # A violation occurs if a qubit being measured is adjacent to more than one active qubit
        for exp in existing_experiments:
            if exp[1] == new_experiment[1]:  # Same qubit being measured
                return False
            if abs(exp[0] - new_experiment[0]) == 1:  # Adjacent qubit being flipped
                return False
        return True

# Example usage
num_qubits = 5  # Example: 5 qubits in a linear arrangement
system = QubitSystem(num_qubits)
min_experiments = system.find_min_experiments()
print("Minimum set of experiments:", min_experiments)
