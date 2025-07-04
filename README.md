# Parallel Hyperparameter Search with @hparam_search
This document provides a guide for using the `@hparam_search` decorator, a powerful tool designed to simplify and accelerate hyperparameter optimization for machine learning models using [Optuna](https://optuna.readthedocs.io/en/stable).

# High-Level Overview
The `@hparam_search` decorator transforms a standard Python training function into a parallel hyperparameter search launcher.

Core Functionality:
- Define your entire **hyperparameter search space**, **fixed parameters**, and study settings in a single **YAML file**.
- It **automatically parallelizes** the search across multiple CPU cores (`n_jobs` entry of yaml file) to speed up the optimization process.
- Requires minimal changes to your existing training code. **You simply decorate your main training function**.
- The search progress is saved to a SQLite database (.db file), allowing you to **resume interrupted** searches and **analyze results later**.
- Can override any parameter from your YAML config directly from the command line.

At its core, it reads a configuration, spawns several worker processes, and lets Optuna's efficient algorithms (now implemented TPE, but with minimal changes you can implement others) find the best hyperparameters for your model.

# How to Use
Using the decorator involves three simple steps: 
1. writing your training logic (you should already have this), 
2. creating a configuration file (`.yaml`), and 
3. decorating your main function with `@hparam_search`.

This should be your code structure:

```plaintext
your_project/
├── your_script.py          # Your training script
├── config/                 # Directory for configuration files
│   └── config.yaml         # Your YAML configuration file
├── optuna_studies/         # Directory for Optuna study databases (will be created automatically)
└── hparam_search.py        # File in this repo
```

## Step 1: Write your training function
Your training function must accept a single dictionary argument (config) and return a single float value (the metric to be optimized, e.g., validation loss).

```python
# your_script.py
import time
from hparam_search import hparam_search

def train_model(config: dict) -> float:
    """
    A dummy training function.
    - Receives a configuration dictionary with all hyperparameters.
    - Returns the value to optimize (e.g., loss).
    """
    print(f"\n--- Starting Trial ---")
    print(f"  - Learning Rate: {config['learning_rate']}")
    print(f"  - Optimizer: {config['optimizer']}")
    print(f"  - Epochs: {config['num_epochs']}")
    print(f"  - Hidden Layers: {config['hidden_layers']}")
    
    # Simulate a training process
    time.sleep(2)
    
    # Example: calculate a dummy loss based on the parameters
    loss = 1.0 / config['learning_rate'] + (5 - config['hidden_layers'])**2
    
    print(f"  - Final Loss: {loss:.4f}")
    print(f"--- Finished Trial ---\n")
    
    return loss # This is the value Optuna will try to minimize.
```

## Step 2: Create a `config.yaml` file
Create a YAML file to define the search space and other settings. This file should be placed in a `config/` directory relative to your script.

```yaml
# config.yaml
# Configuration file for the hyperparameter search.

# --- Fixed Parameters ---
# These are passed to every trial without modification.
experiment_name: "ImageNet_exp"
tag: "first_sweep"
hidden_layers: 8

# --- Optuna Study Configuration ---
direction: "minimize"       # "minimize" a loss or "maximize" an accuracy
n_jobs: 8                   # Number of parallel jobs to run
n_trials: 100               # Total number of trials to run across all jobs
n_startup_trials: 20        # Number of initial random trials before TPE sampler takes over

# --- Hyperparameter Search Space (sweep_parameters) ---
# Defines the hyperparameters for Optuna to search over.
sweep_parameters:
  learning_rate: # Exaple of a floating-point parameter
    type: "float" 
    low: 5.0e-5
    high: 1.0e-2
    log: True # Use a logarithmic scale for sampling

  num_epochs: # Example of an integer parameter
    type: "int"
    low: 10
    high: 100
    step: 10

  optimizer: # Example of a categorical parameter
    type: "categorical"
    choices: ["Adam", "RMSprop", "SGD"]
```
You can have different configurations for different experiments by creating multiple YAML files, e.g., `config/imagenet_exp.yaml`, `config/cifar10_exp.yaml`, etc.


## Step 3: Decorate Your Main Function
Apply the `@hparam_search` decorator to your main function. The decorator needs the path to your config file.

```python
# your_script.py (continued)

@hparam_search(config_path="config.yaml", save_db_directory="./optuna_studies")
def train(config: dict) -> float:
    """
    Main function to run the training with hyperparameter search.
    This function will be called by the decorator.
    """
    return train_model(config)

if __name__ == "__main__":
    train()

```

To run the search execute your script from the terminal (i.e., `python your_script.py`).

A file named ImageNet_exp_first_sweep.db will be created in the optuna_studies/ directory.

# Customization and Advanced Features
## Command-Line Overrides

You can override any parameter from the YAML file using the `--set` flag. This is useful for quick, one-off experiments without editing the config file.

Override a top-level parameter:
```bash
python your_script.py --set n_trials=200
```
Override a nested hyperparameter:
```bash
python your_script.py --set model.first_layer.bias=False
```
Override multiple parameters:
```bash
python your_script.py --set n_jobs=16 tag=second_sweep
```

## Customizing the Search Space
The `sweep_parameters` section in your YAML is highly flexible. Optuna supports several parameter types:
- float: Samples floating-point numbers. Use log: True for parameters that span several orders of magnitude (like learning rate).
- int: Samples integers within a [low, high] range. Use step to define the interval.
- categorical: Samples from a predefined list of choices.
Since the decorator uses Optuna's TPE sampler, refer to the [Optuna documentation](https://optuna.readthedocs.io/en/stable) for more details on advanced sampling strategies.

## Analyzing Results
The search results are stored in a SQLite database. You can analyze them programmatically with Optuna or visually with the `optuna-dashboard`.
Install the dashboard:
```bash
pip install optuna-dashboard
```
Launch the dashboard:
```bash
optuna-dashboard sqlite:///optuna_studies/ImageNet_exp_first_sweep.db
```
This will start a web server where you can visualize the importance of hyperparameters, study optimization history, and explore individual trials.

## Error Handling
The wrapper includes basic error handling. If a trial fails due to an exception (e.g., CUDA out of memory), it will be marked as PRUNED, and the search will continue with the next trial. This prevents a single faulty trial from crashing the entire study. 
