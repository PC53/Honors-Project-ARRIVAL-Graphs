import numpy as np
import math
from arrival import Arrival

def is_converged(x_new, x_old, tol=0):
    """
    Component-wise check.
    """
    return np.max(np.abs(x_new - x_old)) <= tol

def calculate_fixed_point(f, x0, max_iterations=100000, tol=0):
    """
    Calculate the fixed point of the monotone function f using iteration.
    """
    x_old = x0.copy()
    
    for i in range(max_iterations):
        x_new = f(x_old)
        
        if is_converged(x_new, x_old, tol):
            return x_new
        
        x_old = x_new
        print(f"iteration {i} : {x_new}")
    
    raise Exception("Fixed point calculation did not converge within the maximum iterations.")

# Define the monotone function
def monotone_function(x):
    return np.sqrt(x)

# Define the initial guess for the fixed point calculation
n = 3
game = Arrival(n)

game.plot_graph()
print(game.equations)
x0 = np.zeros(n)
x0[0] = 1

# Calculate the fixed point
fixed_point = calculate_fixed_point(game.evaluate, x0)

print("Fixed point:", fixed_point)



def is_converged(x_new, x_old, tol=0):
    """
    Component-wise check.
    """
    return np.max(np.abs(x_new - x_old)) <= tol

def calculate_fixed_point(f, x0, max_iterations=1000, tol=0):
    """
    Calculate the fixed point of the monotone function f using iteration.
    """
    x_old = x0.copy()
    
    for i in range(max_iterations):
        x_new = f(x_old)
        ### both x[0] (sink node count) and x[-1] (target node count) cannot be > 0 at the same time,
        ### since our graph contruction that if a node has path to target node, it will 
        ### be in reachable_nodes and not point to sink node. 
        print(f"iteration {i} : {x_new}")
        if x_new[0] > 0:
            return False
        if x_new[-1] > 0:
            return True
        ### CHANGES MADE : if X-1 becomes >= 1, that means -1 would be visited and since the 
        ###                 function is monotone X-1 would not decrease so we can stop there and 
        ###                 conclude on the ARRIVAL problem early.
        ###                 This is true because each iteration is a run profile, 
        
        ### WHat type of bound can we expect on this method
        
        # if is_converged(x_new, x_old, tol):
        #     return x_new
        
        x_old = x_new
    
    raise Exception("Fixed point calculation did not converge within the maximum iterations.")

example = Arrival(10)
example.save_graph('example.gpickle')
# Define the initial guess for the fixed point calculation
n = 1000
game = Arrival(n)
# print(len(game.equations))
x0 = np.zeros(game.n)
x0[1] = 1

game.run_procedure()
game.graph.edges

# Calculate the fixed point
fixed_point = calculate_fixed_point(game.evaluate, x0)

print("Fixed point:", fixed_point)