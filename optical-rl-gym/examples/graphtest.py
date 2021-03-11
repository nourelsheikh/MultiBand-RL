import matplotlib.pyplot as plt
import pickle
import networkx as nx
with open(f'./topologies/nsfnet_chen_eon_5-paths.h5', 'rb') as f:
    topology = pickle.load(f)
    G = nx.Graph(topology)
    #%matplotlib inline
    pos = nx.spring_layout(G)
    nx.draw(G, with_labels=True, node_color='skyblue', edge_cmap=plt.cm.Blues, pos = pos)
    edge_labels = nx.get_edge_attributes(G,'length') # key is edge, pls check for your case
    formatted_edge_labels = {(elem[0],elem[1]):edge_labels[elem] for elem in edge_labels} # use this to modify the tuple keyed dict if it has > 2 elements, else ignore
    nx.draw_networkx_edge_labels(G,pos,edge_labels=formatted_edge_labels,font_color='blue')
    plt.show()