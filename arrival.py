import numpy as np 
import math 
import random as rand
import copy 
import graphviz
from scipy.optimize import fsolve
from typing import List
import sympy as sp
from sympy import Symbol
from collections import deque



class Arrival():
    
    def __init__(self,n):
        self.n = n # number of nodes
        self.vertices = [v for v in range(n)] # zero is origin and n is the destination 
        self.s_0 = np.array([rand.choice([i for i in range(n) if i != v]) for v in self.vertices])  # even successors
        self.s_1 = np.array([rand.choice([i for i in range(n) if i != v]) for v in self.vertices])  # odd successors
      
        # Some edges can have both successor as themselves, acting as sinks. should this be allowed??
        self.s_curr = np.copy(self.s_0) # current switches for each node
        self.s_next = np.copy(self.s_1) # next switch for each node
        self.v = 0 # current node 
        self.plot_graph('untrimmed.gv')
        self.trim_dead_ends()
        self.get_equations()
        
    def __repr__(self):
        return f"Even Successors: {self.s_0}\nOdd Successors: {self.s_1}\nCurrent Switches: {self.s_curr}\nNext Switches: {self.s_next}\nCurrent Node: {self.v}"
    
    def next_node(self,v):
        assert v < n
        next = self.s_curr[v]
        self.s_curr[v] = self.s_next[v]
        self.next[v] = next
        return next
    
    def plot_graph(self,filename):
        g = graphviz.Digraph('G', filename=filename)
        # g.edges(self.vertices)
        for v in range(self.n):
            g.edge(str(v),str(self.s_0[v]),label='0')
            g.edge(str(v),str(self.s_1[v]),label='1')
            
        g.view()
            
    def system_of_equations(self,x : np.array(int)):
        # number of times the node was visited
        F = np.empty((len(self.vertices)))
        
        for v in self.vertices:
            odd_parents = np.where(self.s_1 == v)[0]
            even_parents = np.where(self.s_0 == v)[0]
            parent_sum = sum([math.floor(x[p]/2) for p in odd_parents]) + sum([math.ceil(x[p]/2) for p in even_parents])

            # origin is visited one extra time 
            total_sum = parent_sum + 1 if v==0 else parent_sum
            F[v] = sp.Min(total_sum,self.n*(2**self.n))
                    
        return F
    
    def trim_dead_ends(self):
        while True:
            print(self.vertices)
            print(self.s_0)
            print(self.s_1)
            dead_ends = self.find_dead_ends()
            if dead_ends == []:
                break 
            
            print(dead_ends)
            for dead_end in dead_ends:
                self.vertices.remove(dead_end)
                self.n -= 1
                self.s_0 = np.delete(self.s_0, dead_end, axis=0)
                self.s_1 = np.delete(self.s_1, dead_end, axis=0)
                self.s_curr = np.delete(self.s_curr, dead_end, axis=0)
                self.s_next = np.delete(self.s_next, dead_end, axis=0)
    
    def find_dead_ends(self):
        visited = set()
        dead_ends = set()

        def bfs(node):
            queue = deque([node])
            visited.add(node)

            while queue:
                current_node = queue.popleft()
                # print(current_node)
                successors = np.array([self.vertices[v] for v in range(self.n) if self.s_0[v] == current_node] + 
                                            [self.vertices[v] for v in range(self.n) if self.s_1[v] == current_node])
                
                # successors = np.array()
                
                # Filter out successors that are -1 (indicating no successor)
                successors = successors[successors != -1]

                for successor in successors:
                    if successor not in visited:
                        visited.add(successor)
                        queue.append(successor)

                if current_node not in np.concatenate([self.s_0, self.s_1]):
                    dead_ends.add(current_node)

        for vertex in self.vertices:
            if vertex not in visited:
                bfs(vertex)

        return list(dead_ends)

    def find_dead_ends1(self,origin, destination):
        # Initialize the queue for BFS
        queue = deque([destination])

        # Initialize switching behavior for each node
        switching_behavior = {}

        # Initialize switching to even for all nodes
        for node in graph.nodes:
            switching_behavior[node] = 'even'

        # Initialize dead ends list
        dead_ends = []

        while queue:
            current_node = queue.popleft()

            # Switch the successor type for the current node
            current_successor_type = switching_behavior[current_node]
            next_successor_type = 'odd' if current_successor_type == 'even' else 'even'
            switching_behavior[current_node] = next_successor_type

            # Check if the current node is a dead end
            if not graph.successors(current_node, successor_type=current_successor_type):
                dead_ends.append(current_node)

            # Add unvisited neighbors to the queue
            for neighbor in graph.neighbors(current_node):
                if neighbor not in switching_behavior:
                    queue.append(neighbor)
                    switching_behavior[neighbor] = next_successor_type

        return dead_ends

              
    def get_equations(self):
        self.X = sp.symbols(' '.join([f"X{i}" for i in range(self.n)]),positive=True)
        
        self.equations = []
        for v in range(self.n):
            odd_parents = np.where(self.s_1 == v)[0]
            even_parents = np.where(self.s_0 == v)[0]
            
            parent_sum = sp.sympify(0)
            for p in odd_parents:
                parent_sum += sp.floor(self.X[p]/2)
            for p in even_parents:
                parent_sum += sp.ceiling(self.X[p]/2)
            # parent_sum = sum([math.floor(X[p]/2) for p in odd_parents]) + sum([math.ceil(X[p]/2) for p in even_parents])
            # eq = X[v] - (parent_sum + 1) if v == 0 else X[v] - parent_sum # origin is visited one more time
            total_sum = (parent_sum + 1) if v == 0 else parent_sum # origin is visited one more time
            eq = sp.Min(total_sum,self.n*(2**self.n))
            self.equations.append(eq)
            
        return self.X,self.equations
    
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
            
        
        
    
# Example usage:
# n = 5 # Size of the instance
# game = Arrival(n)
# print(game)
# game.plot_graph()
# print(game.equations)
# result = game.evaluate(np.ones(n))
# print(result)

    
    
##### driver code 
a = Arrival(25)

# # print(a.s_0)
a.plot_graph('arrival.gv')
# x_guess = np.ones(a.n)
# # F = a.system_of_equations(x_guess)
# sol = fsolve(a.system_of_equations,x_guess)
# print(a.system_of_equations(sol))