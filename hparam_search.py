import argparse
import yaml
import multiprocessing
import optuna
from functools import wraps
from typing import Dict, Any, Callable
import ast
import os

def convert_string_to_type(string: str) -> Any:
    """
    Attempts to convert a string to the most appropriate Python type.
    """
    if not isinstance(string, str):
        return string  
    
    # Try None
    lower_s = string.strip().lower()
    if lower_s == 'none':
        return None
    # Try boolean
    if lower_s == 'true':
        return True
    elif lower_s == 'false':
        return False
    # Try integer
    try:
        return int(string)
    except ValueError:
        pass  
    # Try float
    try:
        return float(string)
    except ValueError:
        pass  
    # Try ast.literal_eval for more complex types (lists, dicts, tuples)
    try:
        return ast.literal_eval(string)
    except (ValueError, SyntaxError):
        pass 
    # Fallback: return the original string
    return string



def _parse_cli_overrides(config: Dict[str, Any], cli_args: list) -> Dict[str, Any]:
    """Applies command-line overrides to the configuration dictionary."""
    if not cli_args:
        return config

    print(f"Applying {len(cli_args)} command-line overrides...")
    for override in cli_args:
        key, value = override.split('=', 1)
        keys = key.split('.')
        d = config
        for k in keys[:-1]:
            d = d.get(k, {})
        
        try:
            converted_value = convert_string_to_type(value)
            d[keys[-1]] = converted_value
        except (KeyError, TypeError, ValueError):
            d[keys[-1]] = value
        print(f"  - Overrode '{key}' to '{d[keys[-1]]}'")
            
    return config

####################################################################################
def _run_worker_process(
    train_func: Callable,
    base_config: Dict[str, Any],
    study_name: str,
    storage_url: str,
    trials_per_worker: int
):
    """
    This function is the target for each worker process. It's completely
    self-contained and receives all necessary information via arguments.
    """
    
    # The objective function is defined inside the worker to have access
    # to the trial object and the config.
    def objective(trial: optuna.trial.Trial) -> float:
        try:
            # Create a copy of the config for this specific trial
            trial_cfg = base_config.copy()

            # Suggest hyperparameters based on the 'sweep_parameters' section of the config
            hparams_config = trial_cfg.get('sweep_parameters', {})
            for name, p in hparams_config.items():
                if p['type'] == 'categorical':
                    trial_cfg[name] = trial.suggest_categorical(name, p['choices'])
                elif p['type'] == 'int':
                    trial_cfg[name] = trial.suggest_int(name, p['low'], p['high'], log=p.get('log', False))
                elif p['type'] == 'float':
                    trial_cfg[name] = trial.suggest_float(name, p['low'], p['high'], log=p.get('log', False))
            
            # Call the original train function with the new config
            return train_func(trial_cfg)
        except Exception as e:
            print(f"Trial {trial.number} failed with exception: {e}. Pruning trial.")
            # Treat the failed trial as a pruned one (TPE adds this configuration to the "bad trials"))
            return optuna.TrialPruned()


    # Configure the sampler for this worker
    sampler_config = base_config.get('sampler', {})
    n_startup_trials = sampler_config.get('n_startup_trials', 10)
    sampler = optuna.samplers.TPESampler(n_startup_trials=n_startup_trials)
    
    print(f"Worker process (PID: {multiprocessing.current_process().pid}) started.")

    # Connect to the shared study
    study = optuna.load_study(
        study_name=study_name,
        storage=storage_url,
        sampler=sampler
        # direction=base_config.get('direction', 'minimize'),
        # load_if_exists=True
    )

    # Run the optimization loop
    # The worker will stop when its trial count is met or the study's total is reached.
    study.optimize(objective, n_trials=trials_per_worker)


####################################################################################
def hparam_search(config_path: str, save_db_directory: str = ".") :
    """
    A decorator that transforms a `train(cfg)` function into a parallel
    hyperparameter search launcher.
    """
    def decorator(train_func: Callable[[Dict[str, Any]], float]):
        @wraps(train_func)
        def wrapper():
            parser = argparse.ArgumentParser(description="Optuna Search Launcher")
            parser.add_argument('--set', nargs='*', help="Override config params, e.g., --set key=value")
            
            args = parser.parse_args()

            #Load and prepare config
            with open(os.path.join('config', config_path), 'r') as f:
                config = yaml.safe_load(f)
            config = _parse_cli_overrides(config, args.set or [])
            
            # Set up the storage
            experiment_name = config.get('experiment_name', 'base_experiment')
            tag = config.get('tag', 'default_tag')
            study_name = f"{experiment_name}_{tag}"
            if not os.path.exists(save_db_directory):
                os.makedirs(save_db_directory)
            storage_url = f"sqlite:///{save_db_directory}/{study_name}.db"

            print("\n" + "="*50)
            print(f"Starting Hyperparameter Search")
            print(f"Study Name: {study_name}")
            print(f"Storage URL: {storage_url}")
            print(f"Total Trials: {config['n_trials']}")
            print(f"Parallel Jobs: {config['n_jobs']}")
            print("="*50 + "\n")

            print(f"Initializing study '{study_name}' at {storage_url}...")
            # Safely initialize the study (so that if multiple workers try to create it, it won't fail)
            optuna.create_study(
                study_name=study_name,
                storage=storage_url,
                direction=config.get('direction', 'minimize'),
                load_if_exists=True  # Use True to avoid errors on re-runs
            )

            # Spawn worker processes
            # Distribute trials among workers, Optuna handles the rest.
            trials_per_worker = (config['n_trials'] + config['n_jobs'] - 1) // config['n_jobs']
            
            processes = []
            for _ in range(config['n_jobs']):
                # Create a new process for each worker
                p = multiprocessing.Process(
                    target=_run_worker_process,
                    args=(
                        train_func,
                        config,
                        study_name,
                        storage_url,
                        trials_per_worker
                    )
                )
                processes.append(p)
                # Start the worker process
                p.start()

            # Wait for all worker processes to complete
            for p in processes:
                p.join()

            print("\n" + "="*50)
            print("Hyperparameter search complete.")
            print(f"Results are stored in: {storage_url}")
            print("="*50)

        return wrapper
    return decorator