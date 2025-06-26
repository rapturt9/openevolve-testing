"""
Simple demonstration of OpenEvolve with OpenRouter.
This script runs a minimal example without the web interface.
"""

import os
import asyncio
from pathlib import Path

from openevolve import OpenEvolve

async def main():
    """Run a simple OpenEvolve demonstration"""
    # Set up OpenRouter API key
    if "OPENROUTER_API_KEY" in os.environ:
        os.environ["OPENAI_API_KEY"] = os.environ["OPENROUTER_API_KEY"]
        print(f"Using OpenRouter API key from environment variables")
    else:
        print("Warning: OPENROUTER_API_KEY environment variable not found.")
        print("The demo will continue, but API calls may fail without a valid key.")
    
    # Get current directory
    current_dir = Path(__file__).parent
    
    # Initialize OpenEvolve
    evolve = OpenEvolve(
        initial_program_path=str(current_dir / "initial_program.py"),
        evaluation_file=str(current_dir / "evaluator.py"),
        config_path=str(current_dir / "config.yaml"),
        output_dir=str(current_dir / "output")
    )
    
    print(f"Starting OpenEvolve demonstration...")
    print(f"Using OpenRouter API with base URL: {evolve.config.llm.api_base}")
    print(f"Using model: {evolve.config.llm.models[0].name}")
    
    # Run a short evolution (just 5 iterations for demonstration)
    best_program = await evolve.run(iterations=5)
    
    # Print results
    print("\nEvolution complete!")
    print(f"Best program metrics:")
    for name, value in best_program.metrics.items():
        print(f"  {name}: {value:.4f}")
    
    # Print artifacts if available
    if hasattr(best_program, "artifacts") and best_program.artifacts:
        print("\nBest program artifacts:")
        for name, value in best_program.artifacts.items():
            if isinstance(value, (int, float)):
                print(f"  {name}: {value:.6f}" if isinstance(value, float) else f"  {name}: {value}")
            else:
                print(f"  {name}: {value}")
    
    print(f"\nOutput directory: {evolve.output_dir}")
    print(f"Best program saved to: {evolve.output_dir}/best_program.py")

if __name__ == "__main__":
    asyncio.run(main())