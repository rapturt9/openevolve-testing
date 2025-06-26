"""
A simple web interface to run and visualize OpenEvolve experiments.
"""

import os
import asyncio
import subprocess
import threading
import time
from pathlib import Path

import gradio as gr

# Get the current directory
CURRENT_DIR = Path(__file__).parent
OUTPUT_DIR = CURRENT_DIR / "output"

def run_experiment(iterations, model_name):
    """Run the OpenEvolve experiment"""
    # Update the config file with the selected model
    config_path = CURRENT_DIR / "config.yaml"
    with open(config_path, "r") as f:
        config_content = f.read()
    
    # Replace the model name
    config_content = config_content.replace("anthropic/claude-3-haiku", model_name)
    
    with open(config_path, "w") as f:
        f.write(config_content)
    
    # Use the OpenRouter API key from environment variables
    if "OPENROUTER_API_KEY" in os.environ:
        os.environ["OPENAI_API_KEY"] = os.environ["OPENROUTER_API_KEY"]
        yield "Using OpenRouter API key from environment variables\n"
    else:
        yield "Warning: OPENROUTER_API_KEY environment variable not found.\n"
        yield "API calls may fail without a valid key.\n"
    
    # Run the experiment in a separate process
    cmd = [
        "python", str(CURRENT_DIR / "run_experiment.py"),
        "--iterations", str(iterations)
    ]
    
    process = subprocess.Popen(
        cmd, 
        stdout=subprocess.PIPE, 
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1
    )
    
    # Stream the output
    output = []
    for line in iter(process.stdout.readline, ''):
        if not line:
            break
        output.append(line)
        yield "\n".join(output)
    
    process.stdout.close()
    process.wait()
    
    # Check if the experiment completed successfully
    if process.returncode != 0:
        output.append(f"Experiment failed with return code {process.returncode}")
        yield "\n".join(output)
        return
    
    # Add a message about visualization
    output.append("\nExperiment completed! You can now view the results in the Visualization tab.")
    yield "\n".join(output)

def get_best_program():
    """Get the best program from the output directory"""
    best_program_path = OUTPUT_DIR / "best_program.py"
    if not best_program_path.exists():
        return "No best program found. Run an experiment first."
    
    with open(best_program_path, "r") as f:
        return f.read()

def get_metrics():
    """Get the metrics from the output directory"""
    metrics_path = OUTPUT_DIR / "best_program_info.json"
    if not metrics_path.exists():
        return "No metrics found. Run an experiment first."
    
    import json
    with open(metrics_path, "r") as f:
        metrics = json.load(f)
    
    # Format the metrics
    result = "# Best Program Metrics\n\n"
    if "metrics" in metrics:
        result += "## Metrics\n"
        for name, value in metrics["metrics"].items():
            result += f"- **{name}**: {value:.4f}\n"
    
    if "artifacts" in metrics:
        result += "\n## Artifacts\n"
        for name, value in metrics["artifacts"].items():
            if isinstance(value, (int, float)):
                result += f"- **{name}**: {value:.6f if isinstance(value, float) else value}\n"
            else:
                result += f"- **{name}**: {value}\n"
    
    return result

def get_evolution_history():
    """Get the evolution history from the checkpoints"""
    checkpoints_dir = OUTPUT_DIR / "checkpoints"
    if not checkpoints_dir.exists():
        return "No checkpoints found. Run an experiment first."
    
    # Get all checkpoint directories
    checkpoint_dirs = sorted([d for d in checkpoints_dir.iterdir() if d.is_dir()], 
                            key=lambda x: int(x.name.split("_")[1]))
    
    if not checkpoint_dirs:
        return "No checkpoints found. Run an experiment first."
    
    # Get metrics from each checkpoint
    import json
    history = []
    for checkpoint_dir in checkpoint_dirs:
        metrics_path = checkpoint_dir / "best_program_info.json"
        if metrics_path.exists():
            with open(metrics_path, "r") as f:
                metrics = json.load(f)
            
            iteration = int(checkpoint_dir.name.split("_")[1])
            if "metrics" in metrics:
                entry = {"iteration": iteration}
                entry.update(metrics["metrics"])
                history.append(entry)
    
    if not history:
        return "No metrics found in checkpoints."
    
    # Format the history as a table
    result = "# Evolution History\n\n"
    result += "| Iteration | Value Score | Distance Score | Combined Score |\n"
    result += "|-----------|-------------|---------------|---------------|\n"
    
    for entry in history:
        result += f"| {entry['iteration']} | {entry.get('value_score', 'N/A'):.4f} | {entry.get('distance_score', 'N/A'):.4f} | {entry.get('combined_score', 'N/A'):.4f} |\n"
    
    return result

def create_interface():
    """Create the Gradio interface"""
    with gr.Blocks(title="OpenEvolve with OpenRouter") as interface:
        gr.Markdown("# OpenEvolve with OpenRouter")
        gr.Markdown("This interface allows you to run and visualize OpenEvolve experiments using OpenRouter API.")
        
        with gr.Tab("Run Experiment"):
            with gr.Row():
                iterations = gr.Slider(minimum=10, maximum=100, value=20, step=10, label="Iterations")
            
            model_name = gr.Dropdown(
                choices=[
                    "anthropic/claude-3-haiku",
                    "anthropic/claude-3-sonnet",
                    "anthropic/claude-3-opus",
                    "openai/gpt-4o",
                    "google/gemini-1.5-pro",
                    "meta-llama/llama-3-70b-instruct"
                ],
                value="anthropic/claude-3-haiku",
                label="Model"
            )
            
            run_button = gr.Button("Run Experiment")
            output = gr.Textbox(label="Output", lines=20)
            
            run_button.click(fn=run_experiment, inputs=[iterations, model_name], outputs=output)
        
        with gr.Tab("Best Program"):
            refresh_button = gr.Button("Refresh")
            program = gr.Code(language="python", label="Best Program")
            
            refresh_button.click(fn=get_best_program, inputs=[], outputs=program)
        
        with gr.Tab("Metrics"):
            metrics_refresh = gr.Button("Refresh Metrics")
            metrics_output = gr.Markdown()
            
            metrics_refresh.click(fn=get_metrics, inputs=[], outputs=metrics_output)
        
        with gr.Tab("Evolution History"):
            history_refresh = gr.Button("Refresh History")
            history_output = gr.Markdown()
            
            history_refresh.click(fn=get_evolution_history, inputs=[], outputs=history_output)
    
    return interface

if __name__ == "__main__":
    # Create the output directory if it doesn't exist
    OUTPUT_DIR.mkdir(exist_ok=True)
    
    # Create and launch the interface
    interface = create_interface()
    interface.launch(server_name="0.0.0.0", server_port=12000, share=True)