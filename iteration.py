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
