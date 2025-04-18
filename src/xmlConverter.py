
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