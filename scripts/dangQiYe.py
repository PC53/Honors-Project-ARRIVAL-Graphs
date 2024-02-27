import math as m
import numpy as np

def binary_search(xL, xR, f):
    """
    Binary search to find a fixed point using the given function f.
    """
    if f(xL) == 0:
        return xL
    elif f(xR) == 0:
        return xR
    else:
        bisection_point = (xL + xR) // 2

        if f(bisection_point) < bisection_point:
            return binary_search(xL, bisection_point, f)
        elif f(bisection_point) > bisection_point:
            return binary_search(bisection_point, xR, f)
        else:
            return bisection_point
        

def get_center_point(Ld):
    """
    Get the center point of the lattice Ld.
    """
    last_coordinate_mean = m.floor(np.mean(Ld[:, -1]))
    center_point = [x for x in Ld if x[-1] == last_coordinate_mean][0]
    return center_point

def FixedPoint(Ld, fd):
    """
    Find the fixed point in the d-dimensional lattice Ld using order preserving function fd and center point x0d.
    """
    d = Ld.shape[1]
    if d > 1:
        x0 = get_center_point(Ld)
        x0d = x0[-1]
        Ld_minus_1 = Ld[Ld[:, -1] == x0d, :-1]
        
        fd_minus_1 = lambda x: fd(x, x0d)[:, :-1]


        # Recur for d-1 dimensions
        x_star = FixedPoint(Ld_minus_1, fd_minus_1)
        
        fd_x_star = fd(x_star, x0d)
        if fd_x_star[:, -1] > x0d:
            Ld = [x for x in Ld if x >= (x_star, x0d)]
            return FixedPoint(Ld, fd)
        elif fd_x_star[:, -1] < x0d:
            Ld = [x for x in Ld if x <= (x_star, x0d)]
            return FixedPoint(Ld, fd)
        else:
            return (x_star, x0d)
    elif d == 1:
        # d = 1, perform binary search
        xL, xR = Ld[0][0], Ld[0][-1]
        return binary_search(xL, xR, fd)


# Define the initial guess for the fixed point calculation
n = 5
game = Arrival(n)

# Calculate the fixed point
fixed_point = FixedPoint(,game.evaluate)

print("Fixed point:", fixed_point)

