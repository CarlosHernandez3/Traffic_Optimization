import torch
import xml.etree.ElementTree as ET
from torch_geometric.data import Data
import os

def convert_sumo_to_pyg(edges_xml, nodes_xml):
    # Parse XML
    edges_root = ET.fromstring(edges_xml)
    nodes_root = ET.fromstring(nodes_xml)

    # Create a mapping from node id to index (0-based)
    node_map = {}
    for i, node in enumerate(nodes_root.findall('node')):
        node_map[node.get('id')] = i

    # Extract node features: x and y coordinates
    num_nodes = len(node_map)
    node_features = torch.zeros((num_nodes, 2))  # [x, y] coordinates as features
    for node in nodes_root.findall('node'):
        idx = node_map[node.get('id')]
        x = float(node.get('x'))
        y = float(node.get('y'))
        node_features[idx] = torch.tensor([x, y])

    # Extract edge indices and edge features
    edge_indices = []
    edge_attrs = []

    for edge in edges_root.findall('edge'):
        from_node = edge.get('from')
        to_node = edge.get('to')
        from_idx = node_map[from_node]
        to_idx = node_map[to_node]

        # Add edge to edge_index
        edge_indices.append([from_idx, to_idx])

        # Extract edge attributes (priority, numLanes, speed)
        priority = int(edge.get('priority'))
        num_lanes = int(edge.get('numLanes'))
        speed = float(edge.get('speed'))

        edge_attrs.append([priority, num_lanes, speed])

    # Convert to PyTorch tensors
    edge_index = torch.tensor(edge_indices).t().contiguous()
    edge_attr = torch.tensor(edge_attrs, dtype=torch.float)

    # Create PyTorch Geometric Data object
    data = Data(
        x=node_features,
        edge_index=edge_index,
        edge_attr=edge_attr,
        num_nodes=num_nodes
    )

    return data


def load_and_convert_sumo_files(edges_file_path, nodes_file_path):

    with open(edges_file_path, 'r') as f:
        edges_xml = f.read()

    with open(nodes_file_path, 'r') as f:
        nodes_xml = f.read()

    return convert_sumo_to_pyg(edges_xml, nodes_xml)


def load_from_config_dir(edges_filename, nodes_filename):

    current_dir = os.path.dirname(os.path.abspath(__file__))

    config_dir = os.path.join(os.path.dirname(current_dir), 'config')

    edges_path = os.path.join(config_dir, edges_filename)
    nodes_path = os.path.join(config_dir, nodes_filename)

    if not os.path.exists(edges_path):
        raise FileNotFoundError(f"Edges file not found: {edges_path}")

    if not os.path.exists(nodes_path):
        raise FileNotFoundError(f"Nodes file not found: {nodes_path}")

    return load_and_convert_sumo_files(edges_path, nodes_path)


if __name__ == "__main__":
    try:
        data = load_from_config_dir('edges.xml', 'nodes.xml')

        print("Successfully loaded data from config directory!")
        print("Number of nodes:", data.num_nodes)
        print("Node features shape:", data.x.shape)
        print("Edge index shape:", data.edge_index.shape)
        print("Edge attributes shape:", data.edge_attr.shape)

    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Using stored data instead...")

        edges_xml = """<edges>
        <edge id="3to4" from="3" to="4" priority="5" numLanes="4" speed="40" />
        <edge id="4to3" from="4" to="3" priority="5" numLanes="4" speed="40" />
         
        <edge id="1to6" from="1" to="6" priority="4" numLanes="2" speed="40" />
        <edge id="6to1" from="6" to="1" priority="4" numLanes="2" speed="40" />
       
        <edge id="2to5" from="2" to="5" priority="3" numLanes="2" speed="40" />
        <edge id="5to2" from="5" to="2" priority="3" numLanes="2" speed="40" />

        <edge id="1to2" from="1" to="2" priority="2" numLanes="3" speed="40" />
        <edge id="2to1" from="2" to="1" priority="2" numLanes="3" speed="40" />
        <edge id="2to3" from="2" to="3" priority="2" numLanes="3" speed="40" />
        <edge id="3to2" from="3" to="2" priority="2" numLanes="3" speed="45" />
        <edge id="4to5" from="4" to="5" priority="1" numLanes="2" speed="45" />
        <edge id="5to4" from="5" to="4" priority="1" numLanes="2" speed="35" />
        <edge id="5to6" from="5" to="6" priority="1" numLanes="2" speed="30" />
        <edge id="6to5" from="6" to="5" priority="1" numLanes="2" speed="30" />
            </edges>"""

        nodes_xml = """<nodes>
        <node id="1" x="0.0" y="200.0" type="priority" />
        <node id="2" x="0.0" y="100.0" type="priority" />
        <node id="3" x="100.0" y="200.0" type="priority" />
        <node id="4" x="200.0" y="200.0" type="priority" />
        <node id="5" x="200.0" y="100.0" type="priority" />
        <node id="6" x="200.0" y="0.0" type="priority" />
            </nodes>"""

        data = convert_sumo_to_pyg(edges_xml, nodes_xml)

        print("Number of nodes:", data.num_nodes)
        print("Node features shape:", data.x.shape)
        print("Edge index shape:", data.edge_index.shape)
        print("Edge attributes shape:", data.edge_attr.shape)



torch.save(data, "../data/data.pt")