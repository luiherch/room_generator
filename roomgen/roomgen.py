from queue import Queue
import random
import itertools
from pyvis.network import Network
import numpy as np
from sklearn.cluster import SpectralClustering
import copy
import logging

class MaxIterationsReached(Exception):
    pass


class Room:
    def __init__(self, nodes: list, min_length: int) -> None:
        self.start = None
        self.end = None
        self.nodes = nodes
        self.network = Network()
        self.max_it = 1000
        self.min_length = min_length
        self.tol = 1
        self.prufer_code = []
        self.logger = logging.getLogger(__name__)

    def __repr__(self) -> str:
        return f"<Room> N={len(self.nodes)}"

    @staticmethod
    def least_distance(graph: dict, source):
        q = Queue()
        distance = {k: 9999999 for k in graph.keys()}
        visited_vertices = set()
        q.put(source)
        visited_vertices.update({0})
        while not q.empty():
            vertex = q.get()
            if vertex == source:
                distance[vertex] = 0
            for u in graph[vertex]:
                if u not in visited_vertices:
                    if distance[u] > distance[vertex] + 1:
                        distance[u] = distance[vertex] + 1
                    q.put(u)
                    visited_vertices.update({u})
        return distance

    def show(self, filename):
        self.network.show(name=filename, notebook=False)

    def degrees(self):
        degrees = {}
        for node in self.nodes:
            deg = len(self.network.neighbors(node))
            if deg not in degrees:
                degrees[deg] = []
            degrees[deg].append(node)
        self.logger.debug(degrees)
        return degrees

    def adjacency_matrix(self):
        adj_list = self.network.get_adj_list()
        keys = sorted(adj_list.keys())
        size = len(keys)
        M = [[0] * size for i in range(size)]
        for a, b in [
            (keys.index(a), keys.index(b)) for a, row in adj_list.items() for b in row
        ]:
            M[a][b] = 2 if (a == b) else 1

        adj_mat = np.array(M)
        return adj_mat

    def adjacency_list(self):
        return self.network.get_adj_list()

    def pick_entrypoints(self, group_degrees: dict) -> None:
        start = None
        end = None
        for it in range(self.max_it):
            it = it + 1
            start = None
            end = None
            for cluster, items in group_degrees.items():
                if start is None:
                    items.pop(1, None)  # remove nodes with degree 1
                    items = [j for i in list(items.values()) for j in i]
                    start = random.choice(items)
                else:
                    k = items.keys()
                    minimums = items[min(k)]
                    end = random.choice(minimums)

            if start is not None and end is not None:
                distances = Room.least_distance(self.adjacency_list(), start)
                if distances[end] >= self.min_length:
                    self.logger.info(f"Shortest distance: {distances[end]}")
                    self.logger.debug(f"N of iterations: {it}")
                    break
                else:
                    self.logger.debug(f"Start to end is too short: {distances[end]}")

            if it >= self.max_it:
                if distances[end] < self.min_length - self.tol:
                    raise MaxIterationsReached("Unable to find an appropriate graph")
                self.logger.warning(f"Caution, this graph has node {self.min_length - self.tol}")

        self.start = start
        self.end = end
        self.network.node_map[self.start]["shape"] = "triangle"
        self.network.node_map[self.end]["shape"] = "diamond"


class RoomGenerator:
    def __init__(self, n_nodes: int = 15, n_clusters: int = 2, min_length: int = 4, tol: int = 1) -> None:
        self.graph = None
        self.n_nodes = n_nodes
        self.n_clusters = n_clusters
        self.spectral_clustering = SpectralClustering(
            n_clusters=self.n_clusters, affinity="precomputed", n_init=100
        )
        self.min_length = min_length
        self.tol = tol
        self.extra_edges = random.randint(5, 7)
        self.nodes = list(range(1, self.n_nodes + 1))
        self.labels = [str(n) for n in self.nodes]
        self.edge_comb = list(itertools.combinations(self.labels, 2))
        self.prufer_code = []
        self.color_map = {0: "#5992b0", 1: "#e9bd54", 2: "#d4728e", 3: "#48A64D"}
        self.graph_iter = 20
        self.logger = logging.getLogger(__name__)

    def __repr__(self):
        return f"<RoomGenerator> N={self.n_nodes}"

    def generate_room(self) -> Room:
        for _ in range(self.graph_iter):

            room = Room(self.nodes,self.min_length)
            self.gen_prufer_code()
            edges = self.prufer_to_graph()
            room.prufer_code = copy.deepcopy(self.prufer_code)
            room.network.add_nodes(self.nodes, label=self.labels)
            room.network.add_edges(edges)
            for extra in range(self.extra_edges):
                while True:
                    a, b = random.choice(self.edge_comb)
                    a = int(a)
                    b = int(b)
                    if (a, b) not in edges and (b, a) not in edges:
                        if (
                                len(room.network.neighbors(a)) < 4
                                and len(room.network.neighbors(b)) < 4
                        ):  # check degree count
                            break
                        self.logger.debug(f"{(a, b)} already has degree 4")
                self.logger.debug(f"Extra edge: {(a, b)}")
                room.network.add_edge(a, b)

            room.degrees()
            adj_mat = room.adjacency_matrix()
            self.spectral_clustering.fit(adj_mat)
            for ind, label in enumerate(self.spectral_clustering.labels_):
                room.network.node_map[ind + 1]["color"] = self.color_map[label]

            group_degrees = {}
            for node in self.nodes:
                deg = len(room.network.neighbors(node))
                cluster = self.spectral_clustering.labels_[node - 1] + 1
                if cluster not in group_degrees:
                    group_degrees[cluster] = {}
                if deg not in group_degrees[cluster]:
                    group_degrees[cluster][deg] = []
                group_degrees[cluster][deg].append(node)

            try:
                room.pick_entrypoints(group_degrees)
                self.prufer_code.clear()
                return room
            except MaxIterationsReached as e:
                self.logger.warning(e)
                self.logger.warning("Building a new graph")
                self.prufer_code.clear()
                continue

    def gen_prufer_code(self):
        for _ in range(self.n_nodes - 2):
            self.prufer_code.append(random.randint(1, self.n_nodes))

    def prufer_to_graph(self):
        m = len(self.prufer_code)
        out_str = []
        vertices = m + 2

        # Initialize the array of vertices
        vertex_set = [0] * vertices

        # Number of occurrences of vertex in code
        for i in range(vertices - 2):
            vertex_set[self.prufer_code[i] - 1] += 1

        # Find the smallest label not present in prufer
        j = 0
        for i in range(vertices - 2):
            for j in range(vertices):
                # If j+1 is not present in prufer set
                if vertex_set[j] == 0:
                    # Remove from Prufer set and print pair
                    vertex_set[j] = -1
                    out_str.append((j + 1, self.prufer_code[i]))
                    vertex_set[self.prufer_code[i] - 1] -= 1
                    break

        j = 0

        # For the last element
        aux1 = None
        for i in range(vertices):
            if vertex_set[i] == 0 and j == 0:
                aux1 = i + 1
                j += 1
            elif vertex_set[i] == 0 and j == 1:
                out_str.append((aux1, i + 1))
        self.logger.debug(out_str)
        return list(out_str)
