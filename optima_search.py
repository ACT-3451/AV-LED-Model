import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import differential_evolution, Bounds
from skopt import gp_minimize, forest_minimize
from skopt.space import Real, Integer
from skopt.plots import plot_convergence, plot_objective
from skopt.utils import use_named_args
import joblib
from tqdm import tqdm
import time
from main_model import initialize_simulation, run_simulation

def define_parameter_bounds(base_params, parameter_bounds=None):
    """
    Define bounds for parameters to be optimized.
    
    Args:
        base_params: Dictionary of base model parameters
        parameter_bounds: Dictionary with parameter names and bounds as (min, max) tuples.
                         If None, default bounds will be used.
                         
    Returns:
        tuple: (parameter_names, search_space, current_bounds)
    """
    # Default bounds based on reasonable ranges (±50% of default values except where noted)
    default_bounds = {
        'start_season':         (0  , 365),  # day of year
        'days':                 (50 , 120),  # growing season length
        'rue':                  (base_params['rue'] * 0.5, base_params['rue'] * 1.5),
        # 'k':                    (0.1, 0.9),  # light extinction coefficient
        # 'sla':                  (base_params['sla'] * 0.5, base_params['sla'] * 1.5),
        # 'alloc_leaf':           (0.1, 0.5),  # biomass allocation to leaves
        # 'alloc_yield':          (0.3, 0.7),  # biomass allocation to yield
        # 'max_lai':              (3.0, 8.0),  # maximum leaf area index
        'rue_shaded':           (base_params['rue_shaded'] * 0.5, base_params['rue_shaded'] * 1.5),
        # 'sla_shaded':           (base_params['sla_shaded'] * 0.5, base_params['sla_shaded'] * 1.5),
        'alloc_leaf_shaded':    (base_params['alloc_leaf_shaded'] * 0.5, base_params['alloc_leaf_shaded'] * 1.5),
        'alloc_yield_shaded':   (base_params['alloc_yield_shaded'] * 0.5, base_params['alloc_yield_shaded'] * 1.5),
        'dmc_shaded':           (base_params['dmc_shaded'] * 0.5, base_params['dmc_shaded'] * 1.5),
        'dmc':                  (base_params['dmc'] * 0.5, base_params['dmc'] * 1.5),
        'panel_coverage':       (0, 1),  # fraction of field covered
        # 'shaded_light_fraction':(0.2, 0.8),  # carefuell, no panelty -> no equelibrium
        'led_light_amount_kWh_d_target': (0, 10),  # amount of LED light
        # 'panel_efficiency':     (0.15, 0.30),  # solar panel efficiency
        # 'led_efficiency':       (0.3, 0.7)  # LED efficiency
    }
    
    # Use provided bounds or defaults
    current_bounds = parameter_bounds if parameter_bounds else default_bounds
    
    # Convert bounds to scikit-optimize search space
    search_space = []
    parameter_names = []
    
    for param_name, (low, high) in current_bounds.items():
        parameter_names.append(param_name)
        
        # Integer parameters
        if param_name in ['start_season', 'days']:
            search_space.append(Integer(low, high, name=param_name))
        else:
            search_space.append(Real(low, high, name=param_name))
            
    print(f"Defined optimization bounds for {len(parameter_names)} parameters")
    
    return parameter_names, search_space, current_bounds

def create_param_dict(base_params, parameter_names, param_values):
    """
    Create a parameter dictionary from parameter values.
    
    Args:
        base_params: Base parameters dictionary
        parameter_names: List of parameter names
        param_values: List of parameter values
        
    Returns:
        dict: Complete parameter dictionary
    """
    # Create a full parameter set by copying the base parameters
    params = base_params.copy()
    
    # Update with new parameter values
    for i, param_name in enumerate(parameter_names):
        params[param_name] = param_values[i]
        
    # Ensure integer parameters
    if 'start_season' in params:
        params['start_season'] = int(params['start_season'])
    if 'days' in params:
        params['days'] = int(params['days'])
        
    return params

def evaluate_parameters(param_values, base_params, parameter_names, target_metric, optimization_history):
    """
    Evaluate a set of parameters by running the model and calculating the target metric.
    
    Args:
        param_values: List of parameter values
        base_params: Base parameters dictionary
        parameter_names: List of parameter names
        target_metric: 'yield' or 'profit'
        optimization_history: List to store optimization history
        
    Returns:
        float: Negative of the target metric (for minimization)
    """
    # Create parameter set from values
    params = create_param_dict(base_params, parameter_names, param_values)
    
    try:
        # Run simulation with the parameter set
        df, results = run_simulation(params)
        
        # Calculate metrics
        dmc = results['dmc']
        dmc_shaded = results['dmc_shaded']
        yield_profit = results['yield_profit']
        energy_selling_price = results['energy_selling_price']
        
        # Calculate total yield (fresh weight)
        base_yield = (df.iloc[-1]['sun_accumulated_yield_biomass'] * (1/dmc)) + \
                    (df.iloc[-1]['shade_accumulated_yield_biomass_sun'] * (1/dmc_shaded))
        led_yield = df.iloc[-1]['shade_accumulated_yield_biomass_led'] * (1/dmc_shaded)
        total_yield = base_yield + led_yield
        
        # Calculate energy profit
        energy_profit = df.iloc[-1]['net_produced_energy'] * energy_selling_price - \
                       results['total_grid_energy_cost']
        
        # Calculate total profit
        total_profit = (total_yield * yield_profit) + energy_profit
        
        # Record this evaluation in history
        if optimization_history is not None:
            optimization_history.append({
                'parameters': {name: value for name, value in zip(parameter_names, param_values)},
                'total_yield': total_yield,
                'total_profit': total_profit,
                'energy_profit': energy_profit
            })
        
        # Return negative value because optimizers minimize by default
        if target_metric == 'yield':
            return -total_yield
        else:  # 'profit'
            return -total_profit
        
    except Exception as e:
        print(f"Error evaluating parameters: {e}")
        # Return a very high value to indicate this is a bad solution
        return 1e10

# Wrapper function for scikit-optimize
def create_skopt_evaluator(base_params, parameter_names, target_metric, optimization_history):
    # Simple evaluator that takes a list of parameter values directly
    def evaluate_parameters_skopt(x):
        return evaluate_parameters(
            x, 
            base_params, 
            parameter_names, 
            target_metric, 
            optimization_history
        )
    return evaluate_parameters_skopt

def optimize_parameters(
    base_crop_name='Model Default (Potato)', 
    target_metric='profit', 
    algorithm='bayesian',
    n_calls=50, 
    n_jobs=-1,
    parameter_bounds=None,
    parameters_to_optimize=None
):
    """
    Run optimization to find parameters that maximize yield or profit.
    
    Args:
        base_crop_name: Name of the base crop to optimize
        target_metric: 'yield' or 'profit'
        algorithm: 'bayesian', 'forest', or 'differential_evolution'
        n_calls: Number of optimization iterations
        n_jobs: Number of parallel jobs (-1 for all available cores)
        parameter_bounds: Dictionary with parameter names and their bounds
        parameters_to_optimize: List of parameters to optimize, or None for all parameters
        
    Returns:
        dict: {'best_params': dict, 'best_result': float, 'optimization_history': list}
    """
    start_time = time.time()
    optimization_history = []
    
    # Get base parameters
    base_params = initialize_simulation(base_crop_name)
    
    # Define parameter bounds
    parameter_names, search_space, current_bounds = define_parameter_bounds(base_params, parameter_bounds)
    
    print(f"Starting {algorithm} optimization for maximum {target_metric}...")
    
    # Filter parameters if needed
    if parameters_to_optimize:
        filtered_indices = [i for i, name in enumerate(parameter_names) 
                           if name in parameters_to_optimize]
        search_space = [search_space[i] for i in filtered_indices]
        parameter_names = [parameter_names[i] for i in filtered_indices]
    
    try:
        if algorithm == 'bayesian':
            # Bayesian optimization with Gaussian Processes
            skopt_evaluator = create_skopt_evaluator(
                base_params, parameter_names, target_metric, optimization_history
            )
            
            result = gp_minimize(
                skopt_evaluator,
                search_space,
                n_calls=n_calls,
                n_initial_points=min(10, n_calls // 3),
                n_jobs=n_jobs,
                verbose=True,
                random_state=42
            )
        elif algorithm == 'forest':
            # Random Forest based optimization
            skopt_evaluator = create_skopt_evaluator(
                base_params, parameter_names, target_metric, optimization_history
            )
            
            result = forest_minimize(
                skopt_evaluator,
                search_space,
                n_calls=n_calls,
                n_initial_points=min(10, n_calls // 3),
                n_jobs=n_jobs,
                verbose=True,
                random_state=42
            )
        elif algorithm == 'differential_evolution':
            # Differential Evolution (genetic algorithm variant)
            bounds = [(space.low, space.high) for space in search_space]
            
            # Create a wrapper that captures the current context
            def de_evaluator(x):
                return evaluate_parameters(
                    x, base_params, parameter_names, target_metric, optimization_history
                )
            
            result = differential_evolution(
                de_evaluator,
                bounds,
                maxiter=n_calls,
                popsize=15,
                workers=n_jobs if n_jobs > 0 else None,
                updating='deferred',
                polish=True
            )
        else:
            raise ValueError(f"Unknown algorithm: {algorithm}")
        
        # Convert result
        if algorithm in ['bayesian', 'forest']:
            best_param_values = result.x
            best_value = -result.fun  # Negate because we minimized negative value
        else:  # differential_evolution
            best_param_values = result.x
            best_value = -result.fun  # Negate to get actual maximum
        
        # Create dictionary of best parameters
        best_params = {name: value for name, value in zip(parameter_names, best_param_values)}
        
        # If we only optimized a subset of parameters, add the original values for the rest
        if parameters_to_optimize:
            full_best_params = base_params.copy()
            for name, value in best_params.items():
                full_best_params[name] = value
            best_params = full_best_params
        
        duration = time.time() - start_time
        print(f"\nOptimization completed in {duration:.1f} seconds")
        print(f"Best {target_metric}: {best_value:.4f}")
        print("\nBest parameters:")
        for name, value in best_params.items():
            if name in current_bounds:
                print(f"{name}: {value:.4f}")
        
        return {
            'best_params': best_params,
            'best_result': best_value,
            'optimization_history': optimization_history,
            'target_metric': target_metric,
            'bounds': current_bounds,
            'base_crop_name': base_crop_name
        }
        
    except Exception as e:
        print(f"Optimization error: {e}")
        return None

def plot_optimization_history(optimization_results):
    """
    Plot the optimization history to visualize convergence.
    
    Args:
        optimization_results: Results dictionary from optimize_parameters
    """
    optimization_history = optimization_results.get('optimization_history', [])
    
    if not optimization_history:
        print("No optimization history to plot")
        return
    
    history_df = pd.DataFrame(optimization_history)
    metrics = ['total_yield', 'total_profit', 'energy_profit']
    
    # Create figure with multiple subplots
    fig, axes = plt.subplots(len(metrics), 1, figsize=(12, 10), sharex=True)
    fig.suptitle('Optimization Progress', fontsize=16)
    
    for i, metric in enumerate(metrics):
        ax = axes[i]
        ax.plot(history_df.index, history_df[metric], 'o-', markersize=4)
        ax.set_ylabel(metric.replace('_', ' ').title())
        ax.grid(True, linestyle='--', alpha=0.7)
        
        # Highlight best point
        best_idx = history_df[metric].idxmax()
        best_value = history_df[metric].max()
        ax.plot(best_idx, best_value, 'r*', markersize=10)
        ax.text(best_idx, best_value, f' {best_value:.4f}', verticalalignment='bottom')
    
    axes[-1].set_xlabel('Evaluation')
    plt.tight_layout(rect=[0, 0.03, 1, 0.97])
    plt.show()

def plot_parameter_importance(optimization_results):
    """
    Plot the relative importance of each parameter based on optimization history.
    
    Args:
        optimization_results: Results dictionary from optimize_parameters
    """
    optimization_history = optimization_results.get('optimization_history', [])
    
    if not optimization_history or len(optimization_history) < 5:
        print("Not enough optimization history to analyze parameter importance")
        return
    
    # Convert history to DataFrame
    history_df = pd.DataFrame(optimization_history)
    
    # Extract parameter values
    param_df = pd.DataFrame([h['parameters'] for h in optimization_history])
    
    # Combine with results
    analysis_df = pd.concat([param_df, history_df[['total_yield', 'total_profit']]], axis=1)
    
    # Calculate correlations
    corr_yield = analysis_df.corr()['total_yield'].drop(['total_yield', 'total_profit'])
    corr_profit = analysis_df.corr()['total_profit'].drop(['total_yield', 'total_profit'])
    
    # Plot correlation bars
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 8))
    fig.suptitle('Parameter Importance (Based on Correlation)', fontsize=16)
    
    # Sort for better visualization
    corr_yield = corr_yield.sort_values()
    corr_profit = corr_profit.sort_values()
    
    # Plot yield correlations
    bars1 = ax1.barh(corr_yield.index, corr_yield, color='skyblue')
    ax1.set_xlabel('Correlation with Yield')
    ax1.set_title('Parameter Impact on Yield')
    ax1.axvline(x=0, color='k', linestyle='--', alpha=0.7)
    ax1.set_xlim(-1, 1)
    
    # Plot profit correlations
    bars2 = ax2.barh(corr_profit.index, corr_profit, color='salmon')
    ax2.set_xlabel('Correlation with Profit')
    ax2.set_title('Parameter Impact on Profit')
    ax2.axvline(x=0, color='k', linestyle='--', alpha=0.7)
    ax2.set_xlim(-1, 1)
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.97])
    plt.show()

def save_optimization_results(optimization_results, filename):
    """
    Save optimization results to a pandas DataFrame and then to a CSV file.
    
    Args:
        optimization_results: Results dictionary from optimize_parameters
        filename: Path to save the CSV file
    """
    # Convert optimization_results to DataFrame
    history_df = pd.DataFrame(optimization_results['optimization_history'])
    
    # Extract best parameters and add them as a row
    best_params = optimization_results['best_params']
    best_params_df = pd.DataFrame([best_params])
    
    # Save to CSV
    history_df.to_csv(filename, index=False)
    best_params_df.to_csv(filename.replace('.csv', '_best_params.csv'), index=False)
    
    print(f"Results saved to {filename} and {filename.replace('.csv', '_best_params.csv')}")

def load_optimization_results(filename):
    """
    Load optimization results from a file.
    
    Args:
        filename: Path to the saved results
        
    Returns:
        dict: Optimization results dictionary
    """
    optimization_results = joblib.load(filename)
    target_metric = optimization_results.get('target_metric', 'unknown')
    best_result = optimization_results.get('best_result', 0)
    
    print(f"Results loaded from {filename}")
    print(f"Best {target_metric}: {best_result:.4f}")
    
    return optimization_results

def compare_with_baseline(optimization_results):
    """
    Compare optimized parameters with baseline.

    Args:
        optimization_results: Results dictionary from optimize_parameters
    """
    best_params = optimization_results.get('best_params')
    base_crop_name = optimization_results.get('base_crop_name')

    if not best_params:
        print("No optimized parameters to compare")
        return

    # Get base parameters
    base_params = initialize_simulation(base_crop_name)

    # Merge optimized parameters with default parameters
    full_best_params = base_params.copy()
    full_best_params.update(best_params)

    # Run baseline simulation
    baseline_df, baseline_results = run_simulation(base_params)

    # Run optimized simulation
    optimized_df, optimized_results = run_simulation(full_best_params)

    # Calculate metrics
    dmc = baseline_results['dmc']
    dmc_shaded = baseline_results['dmc_shaded']
    yield_profit = baseline_results['yield_profit']
    energy_selling_price = baseline_results['energy_selling_price']

    # Baseline metrics
    baseline_base_yield = (baseline_df.iloc[-1]['sun_accumulated_yield_biomass'] * (1 / dmc)) + \
                          (baseline_df.iloc[-1]['shade_accumulated_yield_biomass_sun'] * (1 / dmc_shaded))
    baseline_led_yield = baseline_df.iloc[-1]['shade_accumulated_yield_biomass_led'] * (1 / dmc_shaded)
    baseline_yield = baseline_base_yield + baseline_led_yield

    baseline_energy_profit = baseline_df.iloc[-1]['net_produced_energy'] * energy_selling_price - \
                             baseline_results['total_grid_energy_cost']

    baseline_total_profit = (baseline_yield * yield_profit) + baseline_energy_profit

    # Optimized metrics
    optimized_base_yield = (optimized_df.iloc[-1]['sun_accumulated_yield_biomass'] * (1 / dmc)) + \
                           (optimized_df.iloc[-1]['shade_accumulated_yield_biomass_sun'] * (1 / dmc_shaded))
    optimized_led_yield = optimized_df.iloc[-1]['shade_accumulated_yield_biomass_led'] * (1 / dmc_shaded)
    optimized_yield = optimized_base_yield + optimized_led_yield

    optimized_energy_profit = optimized_df.iloc[-1]['net_produced_energy'] * energy_selling_price - \
                              optimized_results['total_grid_energy_cost']

    optimized_total_profit = (optimized_yield * yield_profit) + optimized_energy_profit

    # Calculate percent improvements
    yield_improvement = ((optimized_yield / baseline_yield) - 1) * 100
    profit_improvement = ((optimized_total_profit / baseline_total_profit) - 1) * 100
    energy_profit_improvement = ((optimized_energy_profit / baseline_energy_profit) - 1) * 100 \
        if baseline_energy_profit != 0 else float('inf')

    # Print comparison
    print("\nComparison with Baseline:")
    print(f"{'Metric':<20} {'Baseline':<15} {'Optimized':<15} {'Improvement':<15}")
    print(f"{'-' * 65}")
    print(f"{'Yield (t/ha)':<20} {baseline_yield:<15.4f} {optimized_yield:<15.4f} {yield_improvement:>+15.2f}%")
    print(f"{'Energy Profit (€/ha)':<20} {baseline_energy_profit:<15.2f} {optimized_energy_profit:<15.2f} {energy_profit_improvement:>+15.2f}%")
    print(f"{'Total Profit (€/ha)':<20} {baseline_total_profit:<15.2f} {optimized_total_profit:<15.2f} {profit_improvement:>+15.2f}%")

    # Compare parameters
    bounds = optimization_results.get('bounds', {})
    print("\nParameter Comparison:")
    print(f"{'Parameter':<25} {'Baseline':<15} {'Optimized':<15} {'% Change':<15}")
    print(f"{'-' * 70}")

    for param_name in sorted(best_params.keys()):
        if param_name in bounds:
            baseline_value = base_params[param_name]
            optimized_value = best_params[param_name]
            pct_change = ((optimized_value / baseline_value) - 1) * 100 if baseline_value != 0 else float('inf')

            print(f"{param_name:<25} {baseline_value:<15.4f} {optimized_value:<15.4f} {pct_change:>+15.2f}%")

def run_optimization_example():
    custom_bounds = None  # This will use the default_bounds defined in the code

    # First phase: broad exploration for profit
    profit_results = optimize_parameters(
        base_crop_name='Model Default (Potato)',
        target_metric='profit',
        algorithm='bayesian',
        n_calls=10,
        parameter_bounds=custom_bounds,  # Use default_bounds
        parameters_to_optimize=None  # Optimize all parameters
    )

    # First phase: broad exploration for yield
    yield_results = optimize_parameters(
        base_crop_name='Model Default (Potato)',
        target_metric='yield',
        algorithm='bayesian',
        n_calls=10,
        parameter_bounds=custom_bounds,  # Use default_bounds
        parameters_to_optimize=None  # Optimize all parameters
    )

    # Compare results for profit
    print("\n\n========================================\n========Comparison for Profit:==========\n========================================")
    compare_with_baseline(profit_results)

    # Compare results for yield
    print("\n\n========================================\n========Comparison for Yield:===========\n========================================")
    compare_with_baseline(yield_results)

    return profit_results, yield_results


if __name__ == "__main__":
    profit_results, yield_results = run_optimization_example()