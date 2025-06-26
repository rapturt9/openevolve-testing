"""
Evaluator for the function minimization problem.
This evaluator runs the minimize_function and scores it based on how close it gets to the global minimum.
"""

import math
import random
import importlib.util
import sys
from pathlib import Path
import traceback

from openevolve.evaluation_result import EvaluationResult

# Known global minimum of the Rosenbrock function
GLOBAL_MINIMUM = (1.0, 1.0)
GLOBAL_MINIMUM_VALUE = 0.0

def evaluate(program_path):
    """
    Evaluate the function minimization program.
    
    Args:
        program_path: Path to the program file
        
    Returns:
        EvaluationResult: Evaluation metrics and artifacts
    """
    try:
        # Set random seed for reproducibility
        random.seed(42)
        
        # Import the program as a module
        program_path = Path(program_path)
        spec = importlib.util.spec_from_file_location("program", program_path)
        program = importlib.util.module_from_spec(spec)
        sys.modules["program"] = program
        spec.loader.exec_module(program)
        
        # Run the minimization function with a fixed number of iterations
        iterations = 5000
        x, y, min_val = program.minimize_function(iterations=iterations)
        
        # Calculate distance from the global minimum
        distance = math.sqrt((x - GLOBAL_MINIMUM[0])**2 + (y - GLOBAL_MINIMUM[1])**2)
        
        # Calculate normalized scores (lower is better for both)
        # For value score: 1.0 means perfect (0.0), 0.0 means very bad (100.0 or worse)
        value_score = max(0.0, 1.0 - min_val / 100.0)
        
        # For distance score: 1.0 means perfect (0.0), 0.0 means very bad (5.0 or worse)
        distance_score = max(0.0, 1.0 - distance / 5.0)
        
        # Combined score (weighted average)
        combined_score = 0.7 * value_score + 0.3 * distance_score
        
        # Return the evaluation result
        return EvaluationResult(
            metrics={
                "value_score": value_score,
                "distance_score": distance_score,
                "combined_score": combined_score,
            },
            artifacts={
                "iterations": iterations,
                "min_x": x,
                "min_y": y,
                "min_value": min_val,
                "distance": distance,
                "notes": f"Found minimum at ({x:.6f}, {y:.6f}) with value {min_val:.10f}"
            }
        )
    
    except Exception as e:
        # Return error information
        return EvaluationResult(
            metrics={
                "value_score": 0.0,
                "distance_score": 0.0,
                "combined_score": 0.0,
            },
            artifacts={
                "error": str(e),
                "traceback": traceback.format_exc(),
                "failure_stage": "execution"
            }
        )