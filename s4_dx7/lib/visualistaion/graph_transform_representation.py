from typing import Mapping
from pyvis.network import Network
import networkx as nx

def draw_transform_graph(G: nx.Graph):
    # Step 2: Initialize a Pyvis Network
    nt = Network(
        height="1080", 
        width="100%", 
        bgcolor="#222222", 
        font_color="white", 
        # select_menu=True,
        # filter_menu=True,
        directed=True,
        layout='spring'
        
        )
    
    # [G.nodes[e].update({'image': image.get(e, 'https://www.w3schools.com/w3css/img_lights.jpg'), 'shape': 'image'}) for e in G.nodes]
    # [G.nodes[e].update({'shape': 'square', 'title': image.get(e), 'font_size': 1}) for e in G.nodes]
    nt.from_nx(
        G, 
        # node_size_transf=lambda x: x.get('node_size', 2),
        # node_size_transf=lambda x: print(x)


        
    )
    # # Step 3 & 4: Add nodes and edges to the Pyvis Network
    # for node in G.nodes:
        # Customize as per your image; adjust width and height as needed
        # nt.add_node(node, shape='image', image=image.get(node))
    #     # Use the title parameter for hover effect

    # for edge in G.edges:
    #     nt.add_edge(edge[0], edge[1])

    # Customize the network's settings if needed, e.g., turn on the physics simulation

    # nt.toggle_physics(True)

    # Step 5: Generate and show or save the interactive graph
    nt.set_options(''' var options = { "nodes": { "size": 20, "shape": "triangle", "width":15, "font.size":"2" }, "edges":{ "width":5, "font.size":"20" } } ''')
    # nt.set_options( '{ "nodes": { "size": 20, "shape": "triangle", "width":15, "font.size":"10" }, "edges":{ "width":5, "font.size":"20" } }')
    # nt.options.
    # .nodes.font.size = 4
    nt.show("graph.html", notebook=False)