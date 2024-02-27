import networkx as nx 
import random
import matplotlib.pyplot as plt
import numpy as np 
import math 
import copy 
import graphviz
from scipy.optimize import fsolve
from typing import List
import sympy as sp
from sympy import Symbol
from collections import deque

class Arrival():
    
    def __init__(self,n:int):
        self.n = n # number of nodes
        self.vertices = [v for v in range(n)] # zero is origin and n is the destination 
        self.s_0 = np.array([random.choice([i for i in range(n) if i != v]) for v in self.vertices])  # even successors
        self.s_1 = np.array([random.choice([i for i in range(n) if i != v]) for v in self.vertices])  # odd successors
      
        # Some edges can have both successor as themselves, acting as sinks. should this be allowed??
        self.start_node = 0 # current node 
        self.target_node = n-1
        
        # Construct Graph Structure and Equations for node visit counts
        self.raw_graph = self.get_network_graph()
        # self.draw_graph(self.raw_graph)
        
        self.graph = self.combine_unreachable_nodes()
        self.s_0, self.s_1 = self.update_successor_list()
        self.X,self.equations = self.get_equations()
        
        self.draw_graph(self.graph)
         
    
    def __repr__(self):
        return f"Vertices: {self.vertices} \nEven Successors: {self.s_0}\nOdd Successors: {self.s_1}\n"
    
    def update_successor_list(self):
        ## now the successor lists is 0 indexed with last element being the successors for -1
        new_s0, new_s1 = [-1]*self.n, [-1]*self.n
        for node in self.graph.nodes:
            if node == -1:
                continue
            
            for source,target,attr in self.graph.out_edges(node,data=True):
                # print(edge)
                label = attr['label']
                # print(node ,edge, label)
                
                if label == '1':
                    new_s1[node] = target
                elif label == '0':
                    new_s0[node] = target     
        # print(new_s1) 
        return new_s0,new_s1
        
    def get_network_graph(self):
        G = nx.MultiDiGraph()
        G.add_nodes_from(self.vertices, d_dash=False)
        
        for v in self.vertices:
            G.add_edge(v,self.s_0[v],label='0')
            G.add_edge(v,self.s_1[v],label='1')
            
        return G
        
    def combine_unreachable_nodes_old(self):
        # reachable_nodes = nx.descendants(self.raw_graph, self.start_node) | {self.start_node}
        reachable_nodes = nx.ancestors(self.raw_graph, self.target_node) | {self.target_node}

        new_G = nx.MultiDiGraph()
        mapping = {}
        counter = 0
        for node in self.raw_graph.nodes():
            if node in reachable_nodes:
                mapping[node] = counter ### to reset the node number we add counter
                counter += 1
            else:
                # print(f'removing node {node}')
                mapping[node] = -1
        # print(mapping)
        new_G.add_node(-1)

        for source, target, attr in self.raw_graph.edges(data=True):

            if source in reachable_nodes and target in reachable_nodes:
                new_G.add_edge(mapping[source], mapping[target],label=attr['label'])
            elif source in reachable_nodes:
                new_G.add_edge(mapping[source], -1,label=attr['label'])
                

        new_G.add_edge(-1, -1,label = '1')
        new_G.add_edge(-1, -1,label = '0')
        
        ## Changing the vertices
        self.vertices = list(new_G.nodes)
        self.n = len(self.vertices)
        self.target_node = self.n-2 
        
        # print(n)
        return new_G

    def combine_unreachable_nodes(self):
        # reachable_nodes = nx.descendants(self.raw_graph, self.start_node) | {self.start_node}
        reachable_nodes = nx.ancestors(self.raw_graph, self.target_node) | {self.target_node}

        new_G = nx.MultiDiGraph()
        new_G.add_edges_from([(-1, -1,{"label" : '1'}),(-1, -1,{"label" : '0'})])
        
        for s,t,attr in self.raw_graph.edges(data=True):
            if s not in reachable_nodes:
                new_G.add_edge(s, -1,label=attr['label'])
            else:
                new_G.add_edge(s, t,label=attr['label'])
                
        ## Changing the vertices
        self.vertices = list(new_G.nodes)
        self.vertices.sort()
        
        self.n = len(self.vertices)
        self.target_node = self.n-2 
        
        # print(n)
        return new_G

    def get_equations(self):
        symbol_nodes = list(self.graph.nodes)
        symbol_nodes.sort()
        
        symbols = sp.symbols(' '.join([f"X{i}" for i in symbol_nodes]),positive=True)
        s_mappings = {n:s for n,s in zip(symbol_nodes,symbols)}
        
        
        equations = []  
        for v in symbol_nodes:
            odd_parents = []
            even_parents = []
            for s, t, attr in self.graph.in_edges(v,data=True):
                
                label = attr['label']
                if label == '1':
                    odd_parents.append(s)
                elif label == '0':
                    even_parents.append(s)
            
            parent_sum = sp.sympify(0)
            for p in odd_parents:
                parent_sum += sp.floor(s_mappings[p]/2)
            for p in even_parents:
                parent_sum += sp.ceiling(s_mappings[p]/2)
            # parent_sum = sum([math.floor(X[p]/2) for p in odd_parents]) + sum([math.ceil(X[p]/2) for p in even_parents])
            # eq = X[v] - (parent_sum + 1) if v == 0 else X[v] - parent_sum # origin is visited one more time
            total_sum = (parent_sum + 1) if v == 0 else parent_sum # origin is visited one more time
            eq = sp.Min(total_sum,self.n*(2**self.n))
            equations.append(eq)
            
        return list(s_mappings.values()),equations
    
    def evaluate(self,x):
        assert len(x) == self.n
        # make sympy assignments of given values 
        assignment = {self.X[i]: value_i for i, value_i in enumerate(x)}
        
        results = []
        for eq in self.equations:
            # substitute assignments in equations
            result = eq.subs(assignment)
            results.append(result)
        
        return np.array(results)
    
    def draw_graph(self,G):
        pos = nx.spring_layout(G)
        edge_labels = {(i, j): attr['label'] for i, j, attr in G.edges(data=True)}
        # node_labels = nx.get_node_attributes(self.graph, 'd_dash')
        nx.draw(G,pos, with_labels=True, node_size=200)
        nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)
        # nx.draw_networkx_labels(self.raw_graph, pos, labels=node_labels)
        
        plt.show()
    
    def save_graph(self,filename):
        ## TODO save graph from self.graph in pdf as well as .npy file
        with open(filename, 'wb') as f:
            pickle.dump(a.graph, f, pickle.HIGHEST_PROTOCOL)
            
            
        g = graphviz.Digraph('G', filename=filename)
        # g.edges(self.vertices)
        for v in range(self.n):
            g.edge(str(v),str(self.s_0[v]),label='0')
            g.edge(str(v),str(self.s_1[v]),label='1')
            
        g.view()
        
    def load_graph(self,filename:str):
        
        with open(filename, 'rb') as f:
            self.raw_graph = pickle.load(f)
            
        self.vertices = list(self.raw_graph.nodes) # zero is origin and n is the destination 
        self.n = len(self.vertices) # number of nodes
        self.draw_graph(self.raw_graph)
        
        # Some edges can have both successor as themselves, acting as sinks. should this be allowed??
        self.start_node = 0 # current node 
        self.target_node = n-1
        self.graph = self.combine_unreachable_nodes()
        self.draw_graph(self.graph)
        
        self.s_0, self.s_1 = self.update_successor_list(self.s_0,self.s_1)
        self.X,self.equations = self.get_equations()
        
            
    def next_node(self,v,s_curr,s_next):
        assert v < self.n
        next = s_curr[v]
        self.s_curr[v] = self.s_next[v]
        self.s_next[v] = next
        return next    
        
    def run_procedure(self):
        v = self.start_node
        s_curr = np.copy(self.s_0) # current switches for each node
        s_next = np.copy(self.s_1) # next switch for each node
        counter = 0
        while v != self.target_node and v != -1:
            counter += 1
            print(f'move {counter} :{v}')
            
            w = s_curr[v]
            s_curr[v] = s_next[v]
            s_next[v] = w  
            v = w
            
        return v
        
    
##### driver code 
a = Arrival(30)
a