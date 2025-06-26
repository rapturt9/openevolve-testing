"""
A simple function minimization program.
This program attempts to find the minimum value of a mathematical function.
"""

import random
import math

# Define the function to minimize (Rosenbrock function)
def rosenbrock(x, y):
    """
    Rosenbrock function: f(x,y) = (1-x)^2 + 100(y-x^2)^2
    Global minimum at (1,1) with value 0
    """
    return (1 - x)**2 + 100 * (y - x**2)**2

# EVOLVE-BLOCK-START
def minimize_function(iterations=1000):
    """
    A simple random search algorithm to minimize the Rosenbrock function.
    
    Args:
        iterations: Number of iterations to run
        
    Returns:
        tuple: (x, y, min_value) - The coordinates and minimum value found
    """
    # Initialize with random values in the range [-5, 5]
    best_x = random.uniform(-5, 5)
    best_y = random.uniform(-5, 5)
    best_value = rosenbrock(best_x, best_y)
    
    # Simple random search
    for _ in range(iterations):
        # Generate a random point
        x = random.uniform(-5, 5)
        y = random.uniform(-5, 5)
        
        # Evaluate the function
        value = rosenbrock(x, y)
        
        # Update if better
        if value < best_value:
            best_x = x
            best_y = y
            best_value = value
    
    return best_x, best_y, best_value
# EVOLVE-BLOCK-END

# Test the function
if __name__ == "__main__":
    # Set random seed for reproducibility
    random.seed(42)
    
    # Run the minimization
    x, y, min_val = minimize_function(iterations=10000)
    
    print(f"Minimum found at: ({x:.6f}, {y:.6f})")
    print(f"Function value: {min_val:.10f}")
    print(f"Distance from true minimum (1,1): {math.sqrt((x-1)**2 + (y-1)**2):.10f}")