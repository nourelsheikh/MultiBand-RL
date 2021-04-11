import optical_rl_gym
from optical_rl_gym.utils import Path

import pandas as pd
from itertools import islice
import numpy as np
import networkx as nx
import math
import pickle
from xml.dom.minidom import parse
import xml.dom.minidom

import matplotlib.pyplot as plt

from graph_utils import read_sndlib_topology, read_txt_file, get_k_shortest_paths, get_path_weight


def get_modulation_formatC(length, modulations):
    for i in range(len(modulations) - 1):
        if modulations[i]['band']=='C':
            if length > modulations[i + 1]['maximum_length'] and length <= modulations[i]['maximum_length']:
                             #print(length, i, modulations[i]['modulation'])
                             return modulations[i]
    # if length <= modulations[-1]['maximum_length']:
        #         print(length, len(modulations) - 1, modulations[len(modulations) - 1]['modulation'])
    return modulations[5]

def get_modulation_formatL(length, modulations):
    for i in range(len(modulations) - 1):
        if modulations[i]['band']=='L':
            if length > modulations[i + 1]['maximum_length'] and length <= modulations[i]['maximum_length']:
                #print(length, i, modulations[i]['modulation'])
                return modulations[i]
    # if length <= modulations[-1]['maximum_length']:
        #         print(length, len(modulations) - 1, modulations[len(modulations) - 1]['modulation'])
    return modulations[len(modulations) - 1]

def get_topology(file_name, topology_name, modulations, k_paths=5):
    k_shortest_paths = {}
    excel = pd.DataFrame([])
    if file_name.endswith('.xml'):
        topology = read_sndlib_topology(file_name)
    elif file_name.endswith('.txt'):
        topology = read_txt_file(file_name)
    else:
        raise ValueError('Supplied topology is unknown')
    idp = 0
    for idn1, n1 in enumerate(topology.nodes()):
        for idn2, n2 in enumerate(topology.nodes()):
            if idn1 < idn2:
                paths = get_k_shortest_paths(topology, n1, n2, k_paths, weight='length')
                #print(n1, n2, len(paths))
                lengths = [get_path_weight(topology, path, weight='length') for path in paths]
                selected_modulationsC = [get_modulation_formatC(length, modulations) for length in lengths]
                selected_modulationsL = [get_modulation_formatL(length, modulations) for length in lengths]
                objs = []
                for path, length, modulationC, modulationL in zip(paths, lengths, selected_modulationsC, selected_modulationsL):
                        #if (modulationC['band'] == 'L'):
                          #print("*&^%^&*((&*&%$^&")
                        #print(idp, modulationC, "MOD C")
                        #print(idp, modulationL, "MOD L")
                        objs.append(Path(idp, path, length, modulationC, modulationL))      # Add to path class best_modulationC
                        #print('\t', idp, length, modulationC, modulationL, path)
                        #print(idn1, idn2, len(path))
                        excel = excel.append(pd.DataFrame({'Source': idn1, 'Destination': idn2, 'Path (hops)': len(path)-1}, index=[0]), ignore_index=True)
                        #print(Path.path)
                        idp += 1

                k_shortest_paths[n1, n2] = objs
                k_shortest_paths[n2, n1] = objs
    topology.graph['name'] = topology_name
    topology.graph['ksp'] = k_shortest_paths
    topology.graph['modulations'] = modulations
    topology.graph['k_paths'] = k_paths
    topology.graph['node_indices'] = []
    for idx, node in enumerate(topology.nodes()):
        topology.graph['node_indices'].append(node)
        topology.nodes[node]['index'] = idx
    print(excel.head())
    excel.to_excel("topology_bounds.xlsx")
    #print("&&&&&&&&&&&&&&&")
    #for i in range(len(selected_modulationsC) - 1):
      #print(selected_modulationsC[i])
    return topology


# defining the EON parameters
# definitions according to : https://github.com/xiaoliangchenUCD/DeepRMSA/blob/eb2f2442acc25574e9efb4104ea245e9e05d9821/K-SP-FF%20benchmark_NSFNET.py#L268
modulations = list()
# modulation: string description
# capacity: Gbps
# maximum_distance: km
#C- Band
modulations.append({'modulation': 'C-BPSK', 'capacity': 23., 'maximum_length': 19700, 'band': 'C'})  #added item in list to denote band
modulations.append({'modulation': 'C-QPSK', 'capacity': 46., 'maximum_length': 9900, 'band': 'C'})
modulations.append({'modulation': 'C-8QAM', 'capacity': 69., 'maximum_length': 5400, 'band': 'C'})
modulations.append({'modulation': 'C-16QAM', 'capacity': 92., 'maximum_length': 1400, 'band': 'C'})
modulations.append({'modulation': 'C-32QAM', 'capacity': 115., 'maximum_length': 1300, 'band': 'C'})
modulations.append({'modulation': 'C-64QAM', 'capacity': 140., 'maximum_length': 700, 'band': 'C'})
modulations.append({'modulation': 'C-256QAM', 'capacity': 186., 'maximum_length': 100, 'band': 'C'})
#L- Band
modulations.append({'modulation': 'L-BPSK', 'capacity': 23., 'maximum_length': 16700, 'band': 'L'})    #L band modulations
modulations.append({'modulation': 'L-QPSK', 'capacity': 46., 'maximum_length': 8400, 'band': 'L'})
modulations.append({'modulation': 'L-8QAM', 'capacity': 69., 'maximum_length': 4600, 'band': 'L'})
modulations.append({'modulation': 'L-16QAM', 'capacity': 92., 'maximum_length': 2200, 'band': 'L'})
modulations.append({'modulation': 'L-32QAM', 'capacity': 115., 'maximum_length': 1100, 'band': 'L'})
modulations.append({'modulation': 'L-64QAM', 'capacity': 140., 'maximum_length': 600, 'band': 'L'})
modulations.append({'modulation': 'L-256QAM', 'capacity': 186., 'maximum_length': 100, 'band': 'L'})
#S- Band
#modulations.append({'modulation': 'S-BPSK', 'capacity': 12.5, 'maximum_length': 8000})
#modulations.append({'modulation': 'S-QPSK', 'capacity': 25., 'maximum_length': 5400})
#modulations.append({'modulation': 'S-8QAM', 'capacity': 37.5, 'maximum_length': 2100})
#modulations.append({'modulation': 'S-16QAM', 'capacity': 50., 'maximum_length': 1200})
#modulations.append({'modulation': 'S-32QAM', 'capacity': 37.5, 'maximum_length': 600})
#modulations.append({'modulation': 'S-64QAM', 'capacity': 50., 'maximum_length': 300})
#E- Band
#modulations.append({'modulation': 'E-BPSK', 'capacity': 12.5, 'maximum_length': 5900})
#modulations.append({'modulation': 'E-QPSK', 'capacity': 25., 'maximum_length': 2900})
#modulations.append({'modulation': 'E-8QAM', 'capacity': 37.5, 'maximum_length': 1100})
#modulations.append({'modulation': 'E-16QAM', 'capacity': 50., 'maximum_length': 600})
#modulations.append({'modulation': 'E-32QAM', 'capacity': 37.5, 'maximum_length': 300})
#modulations.append({'modulation': 'E-64QAM', 'capacity': 50., 'maximum_length': 100})

# other setup:
# modulations.append({'modulation': 'BPSK', 'capacity': 12.5, 'maximum_length': 4000})
# modulations.append({'modulation': 'QPSK', 'capacity': 25., 'maximum_length': 2000})
# modulations.append({'modulation': '8QAM', 'capacity': 37.5, 'maximum_length': 1000})
# modulations.append({'modulation': '16QAM', 'capacity': 50., 'maximum_length': 500})
# modulations.append({'modulation': '32QAM', 'capacity': 62.5, 'maximum_length': 250})
# modulations.append({'modulation': '64QAM', 'capacity': 75., 'maximum_length': 125})

k_paths = 5

# The paper uses K=5 and J=1
topology = get_topology('/home/zceened/optical-rl-gym/examples/topologies/nsfnet_chen.txt', 'NSFNET', modulations, k_paths=k_paths)

with open(f'./topologies/NSFNET_chen_eon_{k_paths}-paths_CL.h5', 'wb') as f:
    pickle.dump(topology, f)

print('done for', topology)
