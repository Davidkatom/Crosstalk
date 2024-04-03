from itertools import combinations, product

# Function to generate all valid chains for the grid
def generate_all_chains(n, m):
    def backtrack(chain, x, y):
        if 0 <= x < n and 0 <= y < m:
            new_chain = chain + [(x, y)]
            if all(abs(x - xc) + abs(y - yc) != 1 for (xc, yc) in chain[:-1]):
                chains.add(tuple(new_chain))
                for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                    nx, ny = x + dx, y + dy
                    if (nx, ny) not in chain:
                        backtrack(new_chain, nx, ny)

    chains = set()
    for i, j in product(range(n), range(m)):
        backtrack([], i, j)
    return [list(chain) for chain in chains]

# Function to check if a set of chains covers all arcs in the grid
def covers_all_arcs(chains, n, m):
    arcs = set()
    for chain in chains:
        for i in range(len(chain) - 1):
            arcs.add((chain[i], chain[i + 1]))
            arcs.add((chain[i + 1], chain[i]))  # Add both directions to ensure undirected coverage

    all_arcs = set()
    for i in range(n):
        for j in range(m):
            for di, dj in [(0, 1), (1, 0), (0, -1), (-1, 0)]:  # Check all adjacent cells
                if 0 <= i + di < n and 0 <= j + dj < m:
                    all_arcs.add(((i, j), (i + di, j + dj)))

    return all_arcs.issubset(arcs)

# Function to find the minimal number of chains to cover all arcs in the grid
def find_minimal_chains_for_arcs(n, m):
    all_chains = generate_all_chains(n, m)
    for num_chains in range(1, len(all_chains) + 1):
        for chain_set in combinations(all_chains, num_chains):
            if covers_all_arcs(chain_set, n, m):
                return chain_set
    return None

# Example usage
n, m = 4, 6  # Adjust based on your grid size
minimal_chains = find_minimal_chains_for_arcs(n, m)
print(minimal_chains)
print(len(minimal_chains))

def draw_chains(chains, n, m):
    # Create an empty grid
    grid = [['.' for _ in range(m)] for _ in range(n)]

    # Assign different characters to different chains
    chain_chars = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
    for i, chain in enumerate(chains):
        char = chain_chars[i % len(chain_chars)]  # Cycle through characters if there are many chains
        for x, y in chain:
            grid[x][y] = char
        print('\n'.join(''.join(row) for row in grid))
        grid = [['.' for _ in range(m)] for _ in range(n)]

    # Convert the grid to a string and print it
   # return '\n'.join(''.join(row) for row in grid)

ascii_art = draw_chains(minimal_chains, n, m)
#print(ascii_art)
