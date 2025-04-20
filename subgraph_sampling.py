import os
import csv
import random
import networkx as nx


def load_mappings(mapping_path):
    """
    Load tab-separated mapping file to a dict.
    """
    mapping = {}
    with open(mapping_path, mode="r", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="\t")
        for row in reader:
            if len(row) >= 2:
                key = row[0].strip()
                value = row[1].strip()
                mapping[key] = value
    return mapping


def load_graph_from_folder(folder_path):
    """
    Construct a directed multigraph from TSV files in the given folder.
    """
    G = nx.MultiDiGraph()
    tsv_files = ["train.tsv", "dev.tsv", "test.tsv"]
    for filename in tsv_files:
        filepath = os.path.join(folder_path, filename)
        if os.path.exists(filepath):
            with open(filepath, mode="r", encoding="utf-8") as f:
                reader = csv.reader(f, delimiter="\t")
                for row in reader:
                    if len(row) < 3:
                        continue
                    subj, relation, obj = row[0].strip(), row[1].strip(), row[2].strip()
                    G.add_edge(subj, obj, relation=relation)
    return G


def clean_entity(name: str) -> str:
    """
    Trim entity name at the first comma and return only the name portion.
    """
    return name.split(",", 1)[0]


def sample_subgraph_by_degree_expanding(G: nx.MultiDiGraph,
                                       start_entity: str,
                                       degrees: dict,
                                       max_nodes: int = 5) -> set:
    """
    Greedily expand the subgraph from start_entity by repeatedly
    selecting the neighbor with the highest degree until max_nodes are reached.
    """
    visited = {start_entity}
    frontier = set(G.predecessors(start_entity)) | set(G.successors(start_entity))
    frontier.discard(start_entity)

    while frontier and len(visited) < max_nodes:
        next_node = max(frontier, key=lambda n: degrees.get(n, 0))
        frontier.remove(next_node)
        if next_node in visited:
            continue
        visited.add(next_node)
        nbrs = set(G.predecessors(next_node)) | set(G.successors(next_node))
        frontier.update(n for n in nbrs if n not in visited)
    return visited


def subgraph_to_natural_text(G: nx.MultiDiGraph,
                             nodes: set,
                             entity2text: dict,
                             relation2text: dict) -> str:
    """
    Convert edges within the sampled subgraph to natural English text.
    """
    lines = []
    subG = G.subgraph(nodes)
    for u, v, key, data in subG.edges(keys=True, data=True):
        head = clean_entity(entity2text.get(u, u))
        rel = relation2text.get(data.get("relation", ""), data.get("relation", ""))
        tail = clean_entity(entity2text.get(v, v))
        lines.append(f"Head: {head}; Relation: {rel}; Tail: {tail}")
    return "\n".join(lines)


def sample_random_entities(G: nx.MultiDiGraph, sample_size: int = 10) -> list:
    """
    Randomly sample a list of entities from the graph.
    """
    all_entities = list(G.nodes())
    return random.sample(all_entities, min(sample_size, len(all_entities)))


def save_results(results: dict, output_file: str) -> None:
    """
    Save the dictionary of start_entity->description to a text file.
    """
    with open(output_file, "w", encoding="utf-8") as f:
        for start, description in results.items():
            f.write(description)
            f.write("\n\n")
    print(f"Results saved to {output_file}")


if __name__ == "__main__":
    # Path to dataset folder
    folder_path = "./data/WN18RR"

    # Load entity and relation text mappings
    entity2text = load_mappings(os.path.join(folder_path, "entity2text.txt"))
    relation2text = load_mappings(os.path.join(folder_path, "relation2text.txt"))

    # Build the multi-directed graph
    G = load_graph_from_folder(folder_path)
    print(f"Loaded graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges.")

    # Precompute global degrees once
    degrees = dict(G.degree())

    # Sample starting entities
    # sample_size = 10
    starts = G.nodes
    print("Sampled start entities:", starts)

    # For each start entity, expand subgraph and generate description
    max_subgraph_nodes = 5
    results = {}
    for start in starts:
        center = clean_entity(entity2text.get(start, start))
        nodes = sample_subgraph_by_degree_expanding(G, start, degrees, max_subgraph_nodes)
        triples_text = subgraph_to_natural_text(G, nodes, entity2text, relation2text)
        # Add center node label in English
        description = f"Center node: {center}\n{triples_text}"
        results[start] = description

    # Save all descriptions
    save_results(results, "entity_relations.txt")
