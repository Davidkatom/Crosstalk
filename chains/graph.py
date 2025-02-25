import random
import copy
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np


class Graph:
    def __init__(self, edges=None, flipped_nodes=None):
        if edges is None:
            self.edges = {}  # Represents the edges as a dictionary
            self.flipped_nodes = set()  # Added to keep track of flipped nodes
        else:
            self.edges = copy.deepcopy(edges)
            self.flipped_nodes = copy.deepcopy(flipped_nodes)

    def get_nodes(self):
        return self.edges.keys()

    def create_grid(self, n, m):
        # Create vertices and edges for an n x m grid
        for row in range(n):
            for col in range(m):
                # Compute the number for the current node based on its position
                node = row * m + col

                # Add an edge from this node to the node directly below it (if it exists)
                if row + 1 < n:
                    self.add_edge(node, (row + 1) * m + col)

                # Add an edge from this node to the node directly to the right of it (if it exists)
                if col + 1 < m:
                    self.add_edge(node, row * m + (col + 1))

    def add_edge(self, node1, node2):
        # Ensure that both nodes are represented in the graph, even if they don't yet have any edges
        if node1 not in self.edges:
            self.edges[node1] = []
        if node2 not in self.edges:
            self.edges[node2] = []

        # Add each node to the other's list of connected nodes (since this is an undirected graph)
        self.edges[node1].append(node2)
        self.edges[node2].append(node1)

    def remove_node(self, node):
        # Remove the specified node from the graph
        if node in self.edges:
            # Remove all edges connected to this node
            for neighbor in self.edges[node]:
                self.edges[neighbor].remove(node)
            del self.edges[node]

    def remove_edge(self, node1, node2):
        # Remove the edge between node1 and node2
        self.edges[node1].remove(node2)
        self.edges[node2].remove(node1)

        # If either node no longer has any edges, remove it from the graph
        if not self.edges[node1]:
            del self.edges[node1]
        if not self.edges[node2]:
            del self.edges[node2]

    def check_if_edge_exists(self, node1, node2):
        # Check if an edge exists between node1 and node2
        return node1 in self.edges and node2 in self.edges[node1]

    def get_vertices(self):
        # Returns all nodes in the graph
        return list(self.edges.keys())

    def get_graph(self):
        # Returns the entire graph as an adjacency list
        return self.edges

    def duplicate(self):
        # Returns a deep copy of the graph
        new_graph = Graph()
        for node, neighbors in self.edges.items():
            for neighbor in neighbors:
                new_graph.add_edge(node, neighbor)
        return new_graph

    def max_neighbors(self):
        max_node = None
        max_neighbors = -1

        for node, neighbors in self.edges.items():
            if len(neighbors) > max_neighbors:
                max_neighbors = len(neighbors)
                max_node = node

        return max_node, max_neighbors

    def two_color(self):
        color = {}  # This dictionary will hold the color assigned to each vertex
        for vertex in self.edges:
            if vertex not in color:  # Start coloring process for unvisited vertices
                if not self._color_graph(vertex, True, color):
                    return None  # If coloring was unsuccessful, the graph cannot be two-colored
        return color

    def _color_graph(self, start, current_color, color):
        if start in color:
            return color[start] == current_color  # Check for coloring conflict

        color[start] = current_color  # Color the vertex

        # Recursively attempt to color adjacent vertices with the opposite color
        for neighbor in self.edges[start]:
            if not self._color_graph(neighbor, not current_color, color):
                return False  # Return False if coloring adjacent vertices fails
        return True

    def draw(self, n=None, m=None,colored_graph=None, subgraph=None):
        G = nx.Graph()
        for node, neighbors in self.edges.items():
            for neighbor in neighbors:
                G.add_edge(node, neighbor)



        # Check if grid dimensions are provided and set positions accordingly
        placed_nodes = set()
        pos = {}
        if n is not None and m is not None:
            # Scaled for better visibility
            x = 0
            y = 0
            plt.figure(figsize=(m, n))
            nodes = list(self.edges.keys())
            nodes.sort()
            i = 0
            while i < len(nodes):
                # node = G.nodes.get(nodes[i])

                if len(self.edges[nodes[i]]) == 1:
                    if x == 0:
                        x += 1
                    else:
                        pos[i] = (x, y)
                        placed_nodes.add(nodes[i])
                        x = 0
                        y += 2
                        i += 5
                        continue
                if x > m:
                    x = 0
                    y += 2
                    i += 4
                    pos[i] = (x, y)
                    placed_nodes.add(nodes[i])
                    continue
                pos[i] = (x, y)
                placed_nodes.add(nodes[i])
                x += 1
                i += 1
            x = 0
            y = 1
            for i in range(len(nodes)):
                # node = G.nodes[nodes[i]]
                if nodes[i] not in placed_nodes:
                    # pos[nodes[i]] = (x, y)
                    # if x == m:
                    #     x = 0
                    #     y += 2
                    if x > m:
                        x = (x - m) % 4
                        y += 2
                    pos[i] = (x, y)
                    x += 4

            # Generate the position of each node based on its grid position
        else:
            plt.figure()  # Default figure size
            pos = None  # Use NetworkX's default positioning

        # Prepare the colors for each node based on whether they are flipped or not
        node_colors = []
        if subgraph is not None:
            for node in G.nodes():
                if node in subgraph.flipped_nodes:
                    node_colors.append('yellow')  # Color for flipped nodes
                elif node in subgraph.get_vertices():
                    node_colors.append('red')
                else:
                    node_colors.append('skyblue')  # Default color
        else:
            node_colors = "skyblue"


        # pos = nx.circular_ladder_graph(G)  # Increase iterations for better placement
        nx.draw(G, pos=pos, with_labels=True, node_color=node_colors, node_size=700, edge_color='k')
        plt.show()

    def divide_graph(self):
        temp_graph = Graph(self.edges, self.flipped_nodes)
        # temp_graph = self.duplicate()
        edges = dict(temp_graph.get_graph())
        subgraphs = []

        def create_subgraph(graph, subgraph):
            # graph.draw()
            node, count = graph.max_neighbors()
            measured = set()
            if count <= 0:
                return subgraph
            for neighbor in edges[node]:
                subgraph.add_edge(node, neighbor)
                measured.add(neighbor)
                # graph.remove_edge(node, neighbor)
            for n in measured:
                next_neighbors = list(edges[n])
                next_neighbors.remove(node)
                for neighbor in next_neighbors:
                    graph.remove_node(neighbor)
                graph.remove_node(n)
            graph.remove_node(node)
            subgraph.flipped_nodes.add(node)
            # subgraph.draw()
            # graph.draw()
            return create_subgraph(graph, subgraph)

        while edges:
            subgraph = Graph()
            subgraph = create_subgraph(temp_graph, subgraph)
            subgraphs.append(subgraph)
            temp_graph = Graph(self.edges, self.flipped_nodes)
            for subgraph in subgraphs:
                for flipped_node in subgraph.flipped_nodes:
                    for nodes in subgraph.edges[flipped_node]:
                        temp_graph.remove_edge(flipped_node, nodes)

            edges = dict(temp_graph.get_graph())
        return subgraphs

def tuplate_edges(subgraph):
    flipped = subgraph.flipped_nodes
    tuples = []
    for node in flipped:
        for neighbor in subgraph.edges[node]:
            tuples.append((node, neighbor))
    return tuples

def print_subgraphs(subgraphs):
    for i, subgraph in enumerate(subgraphs):
        print(f"Subgraph {i + 1}:")
        print(f"  Flipped Nodes: {subgraph.flipped_nodes}")
        print(f"  Edges: {subgraph.edges}")
#
#
# n = 2  # Number of rows
# m = 3  # Number of columns
# graph = Graph()
# graph.create_grid(n, m)  # Define your grid dimensions n and m
#
# g = 0
# for i in range(g):
#     # remove random edge
#     n = len(graph.get_nodes())
#     node = random.randint(0, n - 1)
#     neighbors = graph.edges[node]
#     if len(neighbors) > 0:
#         neighbor = random.choice(neighbors)
#         graph.remove_edge(node, neighbor)
#     else:
#         graph.remove_node(node)
#
# subgraphs = graph.divide_graph()
# subgraphs = graph.divide_graph()
#
# graph.draw()
# # # Create a test graph
# print_subgraphs(subgraphs)
# for sgraph in subgraphs:
#     sgraph.draw()

# graph.draw_with_subgraphs(subgraphs, n, m)  # Draw the original graph with colored subgraphs
