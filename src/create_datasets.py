import rdkit
from rdkit import Chem
from tqdm.auto import tqdm
import numpy as np
import pandas as pd
import networkx as nx
import karateclub
from karateclub import DeepWalk
import torch
from torch.utils.data import Dataset, DataLoader
# from karateclub import DeepWalk
from node2vec import Node2Vec
from node2vec.edges import HadamardEmbedder, AverageEmbedder

def generate_fingerprints(final_edges, fingerprint='RDKIT', graphembed='Deepwalk'):
    
#     finger prints using normal rdkit
    data = []
    for edge in tqdm(final_edges):
        mol1 = Chem.MolFromSmiles(edge[0])
        mol2 = Chem.MolFromSmiles(edge[1])
        fp1 = np.array(rdkit.Chem.RDKFingerprint(mol1, maxPath=7, fpSize=512), dtype=object)
        fp2 = np.array(rdkit.Chem.RDKFingerprint(mol2, maxPath=7, fpSize=512), dtype=object)
        data.append(np.array([fp1,fp2,edge[2]], dtype=object))
    
    finger_print_data = np.array(data)
    

    smiles = []
    for i in final_edges:
        smiles.append(i[0])
        smiles.append(i[1])
    smiles = np.array(list(set(smiles)))
    obj = []
    for i in tqdm(final_edges):
        if i[2] == '1':
            obj.append((np.where(smiles == i[0])[0][0],np.where(smiles == i[1])[0][0]) )

#     graph embeds using deepwalk

    G = nx.Graph()
    G.add_edges_from(obj)
    graph_embeds_data = []
    
    if graphembed == 'Deepwalk':

        model = DeepWalk()
        model.fit(G)
        embedding = model.get_embedding()

        for i in tqdm(final_edges):
            graph_embeds_data.append(np.array([ embedding[np.where(smiles == i[0])[0][0]], 
                                        embedding[np.where(smiles == i[1])[0][0]], 
                                        i[2] ]))

# graph embeds using node2vec
    edge_embeds = []
    if graphembed == 'node2vec':
        node2vec = Node2Vec(G, dimensions=128, walk_length=30, num_walks=200, workers=4)
        model = node2vec.fit(window=10, min_count=1, batch_words=4) 
        edges_embs = HadamardEmbedder(keyed_vectors=model.wv)
        edges_kv = edges_embs.as_keyed_vectors()

        for edge in final_edges:
            smi1 = np.where(smiles == edge[0])[0][0]
            smi2 = np.where(smiles == edge[1])[0][0]
            edge_embeds.append(edges_embs[(str(smi1),str(smi2))])


#     final data   
    data = {}
    for i, edge in tqdm(enumerate(final_edges)):
        if fingerprint == 'RDKIT':
            val = finger_print_data[i][:2]
        
        if graphembed == 'Deepwalk':
            val2 = graph_embeds_data[i][:2]
        
        if graphembed == 'node2vec':
            val2 = edge_embeds[i]
            
        sample = {'smiles':final_edges[i][:2],
                  'fingerprint': val,
                  'graphembedding':val2 ,
                  'link':final_edges[i][2]}
        
        data[i] = sample
    
    return data     

class LinkDataset(Dataset):
    def __init__(self, data,  graphembed='Deepwalk', transform=None):
        self.data = data
        self.graphembed = graphembed
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        fingerprints = np.concatenate((item['fingerprint'][0], item['fingerprint'][1]), axis = 0)
        if self.graphembed == 'Deepwalk':
            graphembeds = np.concatenate((item['graphembedding'][0], item['graphembedding'][1]), axis = 0)
        
        if self.graphembed == 'node2vec':
            graphembeds = item['graphembedding']
                
        label = int(item['link'])
        fp, ge= torch.tensor(fingerprints.astype(float)), torch.tensor(graphembeds.astype(float))
        l= torch.zeros(2)
        l[label] = 1
        return fp.unsqueeze(0), ge.unsqueeze(0), l