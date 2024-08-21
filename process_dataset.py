import networkx as nx
import pickle
import os

def read_dataset(file_path, node_mapping=lambda x: x):
    """ Reads the dataset and returns a graph. """
    G = nx.DiGraph()
    with open(file_path, 'r') as file:
        for line in file:
            if line.startswith('%'):
                continue
            edge_data = line.strip().split()

            source = node_mapping(edge_data[0])
            target = node_mapping(edge_data[1])

            G.add_edge(source, target)
    return G

# Wikipedia data source: http://konect.cc/networks/
dataset_paths = [
    '../data/wikipedia_link_fi/out.wikipedia_link_fi', # nodes: 684613, edges: 14658697, average degree: 42.82
    '../data/wikipedia_link_zh/out.wikipedia_link_zh', # nodes: 1786381, edges: 72614837, average degree: 81.30
    '../data/wikipedia_link_de/out.wikipedia_link_de', # nodes: 3603726, edges: 96865851, average degree: 53.76
    '../data/wikipedia_link_en/out.wikipedia_link_en', # nodes: 13593032, edges: 437217424, average degree: 64.33
]

for file_path in dataset_paths:
    print(f"Reading dataset: {file_path}")
    # Check if the file already has been processed
    if os.path.exists(file_path + '.pickle'):
        print("Dataset already processed, skipping...")
        continue
    # relabel nodes to be integers starting from index 0
    node_mapping = lambda node: int(node) - 1
    G = read_dataset(file_path, node_mapping=node_mapping)
    print(f"Number of nodes: {G.number_of_nodes()}")
    print(f"Number of edges: {G.number_of_edges()}")
    print(f"Average degree: {2 * G.number_of_edges() / G.number_of_nodes()}")
    print("------")
    # Save the graph to a pickle file
    with open(file_path + '.pickle', 'wb') as file:
        pickle.dump(G, file)
