import itertools


class GridChainGenerator:
    def __init__(self, n, m):
        self.n = n  # Grid rows
        self.m = m  # Grid columns
        self.grid = [[(i, j) for j in range(m)] for i in range(n)]
        self.all_chains = []
        self.visited = [[False] * m for _ in range(n)]

    def is_valid_move(self, x, y):
        return 0 <= x < self.n and 0 <= y < self.m and not self.visited[x][y]

    def generate_all_chains_helper(self, chain):
        if len(chain) > 1:  # Save the chain if it has more than one vertex
            self.all_chains.append(list(chain))
        if len(chain) == self.n * self.m:  # Stop if the chain covers all cells
            return
        x, y = chain[-1]
        for dx, dy in [(0, 1), (1, 0), (0, -1), (-1, 0)]:  # N, E, S, W directions
            new_x, new_y = x + dx, y + dy
            if self.is_valid_move(new_x, new_y):
                self.visited[new_x][new_y] = True
                chain.append((new_x, new_y))
                self.generate_all_chains_helper(chain)
                chain.pop()  # Backtrack
                self.visited[new_x][new_y] = False

    def generate_all_chains(self):
        self.all_chains = []
        for i in range(self.n):
            for j in range(self.m):
                self.visited[i][j] = True
                self.generate_all_chains_helper([(i, j)])
                self.visited[i][j] = False
        return self.all_chains

    def covers_all_arcs(self, chains):
        edge_set = set()
        for i in range(self.n):
            for j in range(self.m):
                for dx, dy in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                    new_x, new_y = i + dx, j + dy
                    if 0 <= new_x < self.n and 0 <= new_y < self.m:
                        edge_set.add((min((i, j), (new_x, new_y)), max((i, j), (new_x, new_y))))
        for chain in chains:
            for i in range(len(chain) - 1):
                edge_set.discard((min(chain[i], chain[i + 1]), max(chain[i], chain[i + 1])))
        return len(edge_set) == 0

    def find_minimal_chains_for_arcs(self):
        # This function finds the minimal set of chains that covers all arcs in the grid.
        all_chains = self.generate_all_chains()
        all_edges = self._generate_all_edges()
        min_chains = []
        min_chain_len = float('inf')

        for r in range(1, len(all_chains) + 1):
            for subset in itertools.combinations(all_chains, r):
                covered_edges = set()
                for chain in subset:
                    for i in range(len(chain) - 1):
                        edge = (min(chain[i], chain[i + 1]), max(chain[i], chain[i + 1]))
                        covered_edges.add(edge)
                if covered_edges == all_edges:
                    if r < min_chain_len:
                        min_chain_len = r
                        min_chains = list(subset)
                    break  # Break the loop if all edges are covered
            if min_chains:  # Stop early if we have found a covering set
                break
        return min_chains

    def _generate_all_edges(self):
        # This helper function generates all possible edges in the grid.
        edges = set()
        for i in range(self.n):
            for j in range(self.m):
                if i + 1 < self.n:
                    edges.add(((i, j), (i + 1, j)))  # Vertical edge
                if j + 1 < self.m:
                    edges.add(((i, j), (i, j + 1)))  # Horizontal edge
        return edges

    def draw_chains(self, chains):
        grid = [['.' for _ in range(m)] for _ in range(n)]

        # Assign different characters to different chains
        chain_chars = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
        for i, chain in enumerate(chains):
            char = chain_chars[i % len(chain_chars)]  # Cycle through characters if there are many chains
            for x, y in chain:
                grid[x][y] = char
            print('\n'.join(''.join(row) for row in grid))
            grid = [['.' for _ in range(m)] for _ in range(n)]

# Example Usage
if __name__ == "__main__":
    m, n = 4, 4
    if __name__ == "__main__":
        generator = GridChainGenerator(m, n)
        minimal_chains = generator.find_minimal_chains_for_arcs()
        print("Minimal chains that cover all arcs:")
        print(minimal_chains)
        print(len(minimal_chains))
        print("Visualization:")
        print(generator.draw_chains(minimal_chains))

