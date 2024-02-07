
# %%
from s4_dx7.lib.visualistaion.graph_transform_representation import draw_transform_graph
from matplotlib import pyplot as plt
import numpy as np

from s4_dx7.lib.visualistaion.utils import fig_to_png_data_uri

G=nx.Graph()

G.add_edge('1', '2')
G.add_edge('1', '3')

fig, axs = plt.subplots(2, 2)
for ax in axs.flat:
    ax.plot(np.random.rand(10))
im = fig_to_png_data_uri(fig)

images = {
    '1': im,
    '2': im,
    '3': im,
}
draw_transform_graph(G,images)


# %%
from pyvis.network import Network
import networkx as nx

G=nx.Graph()

G.add_edge('1', '2')
G.add_edge('1', '3')

nx.draw(G, with_labels = True)
nt = Network('500px', '500px')
nt.from_nx(G)
nt.show('nx.html')
nt.show('nx.html', notebook=False)

# %%
from pyvis.network import Network
import networkx as nx

# Step 1: Create a NetworkX graph
G = nx.Graph()
G.add_node(1)
G.add_edge(1, 2)

# Step 2: Initialize a Pyvis Network
nt = Network(notebook=False)
# Step 3 & 4: Add nodes and edges to the Pyvis Network
for node in G.nodes:
    # Customize as per your image; adjust width and height as needed
    nt.add_node(node, shape='image', image=image_url_to_base64('https://www.hrpartners.com.au/career-advice/archives/images/0e371a38-ba45-4610-a738-e22237d2aee2/'))
      # Use the title parameter for hover effect

for edge in G.edges:
    nt.add_edge(edge[0], edge[1])

# Customize the network's settings if needed, e.g., turn on the physics simulation
# nt.toggle_physics(True)

# Step 5: Generate and show or save the interactive graph
nt.show("graph.html", notebook=False)