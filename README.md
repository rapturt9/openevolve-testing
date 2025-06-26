# Using OpenEvolve with OpenRouter

This guide explains how to integrate [OpenEvolve](https://github.com/codelion/openevolve) with [OpenRouter](https://openrouter.ai/) to evolve code using various Large Language Models (LLMs).

## What is OpenEvolve?

OpenEvolve is an evolutionary coding agent that uses LLMs to optimize code through an iterative process. It was inspired by Google DeepMind's AlphaEvolve research and allows you to evolve code to improve performance, efficiency, or functionality.

## What is OpenRouter?

OpenRouter is a unified API that provides access to multiple LLM providers through a single interface. It allows you to use models from Anthropic, OpenAI, Google, Meta, and others with a single API key.

## Integration Benefits

- **Access to Multiple Models**: Use different LLMs for code evolution without changing your code
- **Cost Optimization**: Choose models based on performance/cost tradeoffs
- **Simplified Authentication**: Use a single API key for all models
- **Consistent Interface**: OpenRouter provides an OpenAI-compatible API

## Setup Instructions

### Prerequisites

- Python 3.8+
- Git
- OpenRouter API key (get one at [OpenRouter](https://openrouter.ai/))

### Installation

1. Install OpenEvolve:

   ```bash
   git clone https://github.com/codelion/openevolve.git
   cd openevolve
   pip install -e .
   cd ..
   ```

2. Install required dependencies:

   ```bash
   pip install openai>=1.0.0 gradio>=4.0.0 numpy>=1.20.0 matplotlib>=3.5.0 pyyaml>=6.0 requests>=2.25.0 tqdm>=4.60.0
   ```

3. Set your OpenRouter API key as an environment variable:
   ```bash
   export OPENROUTER_API_KEY=your_openrouter_api_key
   ```

## Configuration

Create a `config.yaml` file with OpenRouter settings:

```yaml
# OpenEvolve Configuration for OpenRouter
max_iterations: 50
checkpoint_interval: 10
log_level: "INFO"
random_seed: 42

# LLM configuration
llm:
  # Models for evolution
  models:
    - name: "anthropic/claude-3-haiku" # Or any other model available on OpenRouter
      weight: 1.0

  # API configuration
  api_base: "https://openrouter.ai/api/v1" # OpenRouter API base URL
  api_key: null # Will use OPENAI_API_KEY env variable

  # Generation parameters
  temperature: 0.7
  max_tokens: 4096
```

## Creating Your Project

### 1. Create Initial Program

Create an initial program with the code you want to evolve:

```python
# my_program.py
"""
My program to evolve.
"""

# EVOLVE-BLOCK-START
def my_function():
    # This code will be evolved
    return "Hello, world!"
# EVOLVE-BLOCK-END
```

The code between `# EVOLVE-BLOCK-START` and `# EVOLVE-BLOCK-END` will be evolved by OpenEvolve.

### 2. Create Evaluator

Create an evaluator to assess the performance of evolved programs:

```python
# evaluator.py
from openevolve.evaluation_result import EvaluationResult

def evaluate(program_path):
    # Import the program
    import importlib.util
    spec = importlib.util.spec_from_file_location("program", program_path)
    program = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(program)

    # Evaluate the program
    try:
        # Your evaluation logic here
        score = 0.5  # Example score

        return EvaluationResult(
            metrics={"score": score},
            artifacts={"notes": "Evaluation successful"}
        )
    except Exception as e:
        return EvaluationResult(
            metrics={"score": 0.0},
            artifacts={"error": str(e)}
        )
```

### 3. Create Run Script

Create a script to run the evolution:

```python
# run.py
import os
import asyncio
from openevolve import OpenEvolve

async def main():
    # Set up OpenRouter API key
    if "OPENROUTER_API_KEY" in os.environ:
        os.environ["OPENAI_API_KEY"] = os.environ["OPENROUTER_API_KEY"]

    # Initialize OpenEvolve
    evolve = OpenEvolve(
        initial_program_path="my_program.py",
        evaluation_file="evaluator.py",
        config_path="config.yaml"
    )

    # Run the evolution
    best_program = await evolve.run(iterations=50)

    # Print results
    print(f"Best program metrics:")
    for name, value in best_program.metrics.items():
        print(f"  {name}: {value:.4f}")

if __name__ == "__main__":
    asyncio.run(main())
```

### 4. Run Your Project

Set your OpenRouter API key and run the evolution:

```bash
export OPENROUTER_API_KEY=your_openrouter_api_key
python run.py
```

## Available Models

You can use any of the following models with OpenRouter:

- `anthropic/claude-3-haiku`
- `anthropic/claude-3-sonnet`
- `anthropic/claude-3-opus`
- `openai/gpt-4o`
- `google/gemini-1.5-pro`
- `meta-llama/llama-3-70b-instruct`

To use a different model, simply change the model name in your `config.yaml` file.

## Advanced Features

### Checkpoints

OpenEvolve saves checkpoints at regular intervals, allowing you to:

- Resume evolution from any checkpoint
- Compare solutions across different generations
- Analyze performance improvements over time

### Island Model

OpenEvolve supports an island model for evolution, which maintains separate populations that evolve independently:

```yaml
database:
  num_islands: 3 # Number of islands for island model
```

### Prompt Engineering

You can customize the prompts used for code generation:

```yaml
prompt:
  system_message: "You are an expert coder helping to improve programs through evolution."
  evaluator_system_message: "You are an expert code reviewer."
```

## Example: Function Minimization

Here's a complete example of using OpenEvolve with OpenRouter to minimize the Rosenbrock function:

### Initial Program

```python
# initial_program.py
import random
import math

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
    """
    # Initialize with random values
    best_x = random.uniform(-5, 5)
    best_y = random.uniform(-5, 5)
    best_value = rosenbrock(best_x, best_y)

    # Simple random search
    for _ in range(iterations):
        x = random.uniform(-5, 5)
        y = random.uniform(-5, 5)
        value = rosenbrock(x, y)
        if value < best_value:
            best_x, best_y, best_value = x, y, value

    return best_x, best_y, best_value
# EVOLVE-BLOCK-END
```

### Evaluator

```python
# evaluator.py
from openevolve.evaluation_result import EvaluationResult

def evaluate(program_path):
    """Evaluate the program's ability to minimize the Rosenbrock function"""
    # Import the program
    import importlib.util
    spec = importlib.util.spec_from_file_location("program", program_path)
    program = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(program)

    # Run the minimization
    iterations = 5000
    try:
        x, y, min_val = program.minimize_function(iterations=iterations)

        # Calculate distance from true minimum (1,1)
        distance = ((x-1)**2 + (y-1)**2)**0.5

        # Calculate scores (higher is better)
        value_score = max(0, 1 - min_val) if min_val >= 0 else 0
        distance_score = max(0, 1 - distance/2)  # Normalize by expected max distance
        combined_score = (value_score + distance_score) / 2

        return EvaluationResult(
            metrics={
                "value_score": value_score,
                "distance_score": distance_score,
                "combined_score": combined_score
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
        import traceback
        return EvaluationResult(
            metrics={"value_score": 0.0, "distance_score": 0.0, "combined_score": 0.0},
            artifacts={
                "error": str(e),
                "traceback": traceback.format_exc(),
                "failure_stage": "execution"
            }
        )
```

### Run Script

```python
# run_experiment.py
import os
import asyncio
from openevolve import OpenEvolve

async def main():
    # Set up OpenRouter API key
    if "OPENROUTER_API_KEY" in os.environ:
        os.environ["OPENAI_API_KEY"] = os.environ["OPENROUTER_API_KEY"]
        print("Using OpenRouter API key from environment variables")

    # Initialize OpenEvolve
    evolve = OpenEvolve(
        initial_program_path="initial_program.py",
        evaluation_file="evaluator.py",
        config_path="config.yaml",
        output_dir="output"
    )

    print(f"Starting OpenEvolve experiment...")
    print(f"Using OpenRouter API with base URL: {evolve.config.llm.api_base}")
    print(f"Using model: {evolve.config.llm.models[0].name}")

    # Run the evolution
    best_program = await evolve.run(iterations=50)

    # Print results
    print("\nEvolution complete!")
    print(f"Best program metrics:")
    for name, value in best_program.metrics.items():
        print(f"  {name}: {value:.4f}")

if __name__ == "__main__":
    asyncio.run(main())
```

## Troubleshooting

### API Key Issues

If you encounter authentication errors, make sure:

1. Your OpenRouter API key is valid
2. You've set the `OPENROUTER_API_KEY` environment variable
3. Your account has sufficient credits

### Model Availability

If a specific model is unavailable:

1. Check if the model is supported by OpenRouter
2. Try a different model by changing the model name in your config
3. Check the OpenRouter status page for any outages

### Rate Limiting

If you encounter rate limits:

1. Increase the retry settings in your config
2. Add delay between API calls
3. Consider upgrading your OpenRouter plan

## Resources

- [OpenEvolve GitHub Repository](https://github.com/codelion/openevolve)
- [OpenRouter Documentation](https://openrouter.ai/docs)
- [AlphaEvolve Paper](https://arxiv.org/abs/2404.09655) - The original research paper by Google DeepMind
