""" Workflow-Parser + Graph-Renderer  
    Beispiel mit networkx + matplotlib """

import json
import networkx as nx
import matplotlib.pyplot as plt

with open("workflow.json") as f:
    data = json.load(f)

G = nx.DiGraph()

for node_id, node in data["nodes"].items():
    G.add_node(node_id, label=node["class_type"])
    for input_list in node.get("inputs", {}).values():
        for input_node in input_list:
            if isinstance(input_node, list) and len(input_node) == 2:
                G.add_edge(input_node[0], node_id)

nx.draw(G, with_labels=True)
plt.show()
