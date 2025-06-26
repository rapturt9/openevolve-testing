"""
Script to view the programs at each checkpoint.
This allows you to see how the program evolved over time.
"""

import os
import sys
import json
from pathlib import Path
import argparse

def print_separator(title):
    """Print a separator with a title"""
    width = 80
    print("\n" + "=" * width)
    print(f"{title:^{width}}")
    print("=" * width + "\n")

def view_checkpoints(output_dir=None):
    """View the programs at each checkpoint"""
    # Get the output directory
    if output_dir is None:
        output_dir = Path(__file__).parent / "output"
    else:
        output_dir = Path(output_dir)
    
    # Check if the output directory exists
    if not output_dir.exists():
        print(f"Error: Output directory '{output_dir}' does not exist.")
        return
    
    # Get the checkpoints directory
    checkpoints_dir = output_dir / "checkpoints"
    if not checkpoints_dir.exists():
        print(f"Error: Checkpoints directory '{checkpoints_dir}' does not exist.")
        return
    
    # Get all checkpoint directories
    checkpoint_dirs = sorted([d for d in checkpoints_dir.iterdir() if d.is_dir()], 
                            key=lambda x: int(x.name.split("_")[1]))
    
    if not checkpoint_dirs:
        print("No checkpoints found.")
        return
    
    print(f"Found {len(checkpoint_dirs)} checkpoints.")
    
    # Process each checkpoint
    for checkpoint_dir in checkpoint_dirs:
        iteration = int(checkpoint_dir.name.split("_")[1])
        print_separator(f"Checkpoint {iteration}")
        
        # Get the best program
        best_program_path = checkpoint_dir / "best_program.py"
        if best_program_path.exists():
            with open(best_program_path, "r") as f:
                program = f.read()
            
            print("Best Program:")
            print(program)
        else:
            print("No best program found.")
        
        # Get the metrics
        metrics_path = checkpoint_dir / "best_program_info.json"
        if metrics_path.exists():
            with open(metrics_path, "r") as f:
                metrics = json.load(f)
            
            print("\nMetrics:")
            if "metrics" in metrics:
                for name, value in metrics["metrics"].items():
                    print(f"  {name}: {value:.4f}")
            
            print("\nArtifacts:")
            if "artifacts" in metrics:
                for name, value in metrics["artifacts"].items():
                    if isinstance(value, (int, float)):
                        print(f"  {name}: {value:.6f if isinstance(value, float) else value}")
                    else:
                        print(f"  {name}: {value}")
        else:
            print("No metrics found.")

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="View the programs at each checkpoint")
    parser.add_argument("--output-dir", type=str, help="Output directory (default: ./output)")
    args = parser.parse_args()
    
    view_checkpoints(args.output_dir)

if __name__ == "__main__":
    main()