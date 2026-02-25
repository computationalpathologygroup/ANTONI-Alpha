import yaml

# --- Node and Tree Construction ---


class Node:
    def __init__(self, name, synonyms=None, parent=None):
        self.name = name
        self.synonyms = synonyms or []
        self.parent = parent
        self.children = []

    def __repr__(self):
        return f"Node({self.name})"


def build_tree(data, parent=None):
    """
    Recursively builds a list of nodes from the taxonomy YAML data.
    """
    nodes = []
    for key, value in data.items():
        synonyms = value.get("synonyms", [])
        node = Node(name=key, synonyms=synonyms, parent=parent)
        if parent:
            parent.children.append(node)
        nodes.append(node)
        if "parts" in value:
            child_nodes = build_tree(value["parts"], parent=node)
            nodes.extend(child_nodes)
    return nodes


def build_lookup(nodes):
    """
    Returns a lookup dictionary mapping each canonical name and synonym (lowercase) to the node.
    """
    lookup = {}
    for node in nodes:
        lookup[node.name.lower()] = node
        for syn in node.synonyms:
            lookup[syn.lower()] = node
    return lookup


def build_graph(nodes):
    """
    Builds an undirected graph of nodes where edges are added for parent-child and sibling relationships.
    """
    graph = {node: set() for node in nodes}
    for node in nodes:
        if node.parent:
            # Parent-child bidirectional edges.
            graph[node].add(node.parent)
            graph[node.parent].add(node)
            # Sibling edges.
            for sibling in node.parent.children:
                if sibling is not node:
                    graph[node].add(sibling)
                    graph[sibling].add(node)
    return graph


def shortest_path(graph, start, goal):
    """
    Compute the shortest path distance between start and goal nodes.
    """
    from collections import deque

    queue = deque([(start, 0)])
    visited = {start}
    while queue:
        current, dist = queue.popleft()
        if current == goal:
            return dist
        for neighbor in graph[current]:
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append((neighbor, dist + 1))
    return float("inf")


def load_taxonomy(taxonomy_file):
    """
    Loads the YAML taxonomy and returns nodes, lookup and graph.
    """
    with open(taxonomy_file, "r") as f:
        taxonomy_data = yaml.safe_load(f)
    nodes = build_tree(taxonomy_data)
    lookup = build_lookup(nodes)
    graph = build_graph(nodes)
    return nodes, lookup, graph
