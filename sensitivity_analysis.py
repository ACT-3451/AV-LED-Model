import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm  # For progress bars
from main_model import initialize_simulation, run_simulation
import os
from datetime import datetime


parameter_ranges = {
    # 'start_season': {'min': 1, 'max': 365, 'default': 100},
    # 'days': {'min': 50, 'max': 200, 'default': 120},
    'rue': {'min': 1, 'max': 30, 'default': 10.8},
    # 'k': {'min': 0.3, 'max': 1.0, 'default': 0.5},
    'sla': {'min': 0.003, 'max': 0.3, 'default': 0.03},
    'alloc_leaf': {'min': 0.1, 'max': 0.8, 'default': 0.3},
    'alloc_yield': {'min': 0.01, 'max': 0.8, 'default': 0.3},
    # 'max_lai': {'min': 2, 'max': 8, 'default': 4},
    'rue_shaded': {'min': 1, 'max': 30, 'default': 10.8},
    'sla_shaded': {'min': 0.003, 'max': 0.3, 'default': 0.03},
    'alloc_leaf_shaded': {'min': 0.1, 'max': 0.8, 'default': 0.3},
    'alloc_yield_shaded': {'min': 0.01, 'max': 0.8, 'default': 0.3},
    'dmc_shaded': {'min': 0.1, 'max': 0.9, 'default': 0.2},
    'dmc': {'min': 0.1, 'max': 0.9, 'default': 0.2},
    'panel_coverage': {'min': 0.0, 'max': 1.0, 'default': 0.32},
    'shaded_light_fraction': {'min': 0.0, 'max': 0.5, 'default': 0.25},
    'led_light_amount_kWh_d_target': {'min': 0, 'max': 20, 'default': 4},
    'panel_efficiency': {'min': 0.1, 'max': 0.3, 'default': 0.2},
    'led_efficiency': {'min': 0.1, 'max': 1, 'default': 0.5},
    # 'energy_selling_price': {'min': 0.0, 'max': 0.2, 'default': 0.07},
    # 'grid_energy_price': {'min': 0, 'max': 0.9, 'default': 0.09},
    # 'tau': {'min': 0.1, 'max': 1.0, 'default': 0.5},
    # 'max_dtr': {'min': 5.0, 'max': 15.0, 'default': 13.8889},
    # 'min_dtr': {'min': 2.0, 'max': 10.0, 'default': 5.55556},
    # 'f_par': {'min': 0.4, 'max': 0.6, 'default': 0.5}  # Added 'f_par'
}


def save_figure(fig, title):
    """
    Save the given figure with a descriptive name and timestamp in both PDF and PNG formats.
    
    Args:
        fig: The matplotlib figure to save.
        title: The title of the plot, used to generate the filename.
    """
    # Create the 'figures' folder if it doesn't exist
    figures_folder = "figures"
    os.makedirs(figures_folder, exist_ok=True)
    
    # Generate a descriptive filename with the current date and time
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_title = title.replace(" ", "_").replace(":", "").replace("/", "_")
    filename = f"{safe_title}_{timestamp}"
    
    # Save the figure as both PDF and PNG in high resolution
    fig.savefig(os.path.join(figures_folder, f"{filename}.pdf"), dpi=300, bbox_inches='tight')
    fig.savefig(os.path.join(figures_folder, f"{filename}.png"), dpi=300, bbox_inches='tight')
    print(f"Figure saved as '{filename}.pdf' and '{filename}.png' in the 'figures' folder.")


def run_spider_sensitivity_analysis(
    crop_name='Model Default (Potato)',
    param_variation_pct=50,
    increments=10,
    parameters_to_test=None
):
    """
    Run sensitivity analysis for each parameter with variations of ±param_variation_pct% 
    for spider plot visualization.
    """
    # Initialize base parameters
    base_params = initialize_simulation(crop_name)
    
    # Default parameters to test if not provided
    if parameters_to_test is None:
        parameters_to_test = [
            'panel_coverage',
            'shaded_light_fraction',
            'led_light_amount_kWh_d_target',
            'panel_efficiency',
            'led_efficiency'
        ]
    
    # Filter to only include numeric parameters
    numeric_params = {}
    for param in parameters_to_test:
        if param in base_params and isinstance(base_params[param], (int, float)) and param != 'tau':
            numeric_params[param] = base_params[param]
    
    # Generate baseline results
    print("Running baseline simulation...")
    baseline_df, baseline_results = run_simulation(base_params)
    
    # Get metrics directly from the DataFrame
    last_row = baseline_df.iloc[-1]
    baseline_total_yield = last_row['total_yield_freshweight']
    baseline_total_profit = last_row['total_profit']
    
    print(f"Baseline total yield: {baseline_total_yield:.4f} t/ha")
    print(f"Baseline total profit: {baseline_total_profit:.2f} €/ha")
    
    # Prepare results storage
    results = []
    
    # Calculate total simulations for overall progress bar
    total_params = len(numeric_params)
    total_simulations = total_params * (2 * increments + 1)
    
    # Create overall progress bar
    print(f"\nRunning {total_simulations} simulations across {total_params} parameters...")
    overall_progress = tqdm(total=total_simulations, desc="Overall Progress", position=0)
    
    # Calculate variation range for each parameter
    for param_name, base_value in numeric_params.items():
        min_value = base_value * (1 - param_variation_pct/100)
        max_value = base_value * (1 + param_variation_pct/100)
        
        # Handle special cases
        if param_name == 'panel_coverage' or param_name == 'shaded_light_fraction':
            min_value = max(min_value, 0)  # Avoid zero coverage
            max_value = min(max_value, 1)  # Avoid full coverage
        
        # Create evenly spaced increments
        param_values = np.linspace(min_value, max_value, 2*increments+1)
        
        print(f"\nTesting parameter: {param_name} - Base value: {base_value}")
        for new_value in tqdm(param_values, desc=f"{param_name}", position=1, leave=False):
            # Create parameter set with the varied parameter
            test_params = base_params.copy()
            test_params[param_name] = new_value
            
            # Ensure start_season and days are integers
            if 'start_season' in test_params:
                test_params['start_season'] = int(test_params['start_season'])
            if 'days' in test_params:
                test_params['days'] = int(test_params['days'])
            
            # Run simulation
            df, sim_results = run_simulation(test_params)
            
            # Get metrics directly from the DataFrame
            last_row = df.iloc[-1]
            total_yield = last_row['total_yield_freshweight']
            total_profit = last_row['total_profit']
            
            # Calculate percent change from baseline
            yield_pct_change = ((total_yield / baseline_total_yield) - 1) * 100
            profit_pct_change = ((total_profit / baseline_total_profit) - 1) * 100
            
            # Store results
            results.append({
                'parameter': param_name,
                'base_value': base_value,
                'test_value': new_value,
                'percent_change': ((new_value / base_value) - 1) * 100,
                'total_yield': total_yield,
                'total_profit': total_profit,
                'yield_pct_change': yield_pct_change,
                'profit_pct_change': profit_pct_change
            })
            
            # Update overall progress bar
            overall_progress.update(1)
    
    overall_progress.close()
    
    # Convert results to DataFrame
    results_df = pd.DataFrame(results)
    
    # Return results as DataFrame
    return results_df, baseline_total_yield, baseline_total_profit


def run_sensitivity_analysis_with_specific_ranges(parameter_ranges, increments=10):
    """
    Run sensitivity analysis for each parameter within specific ranges for spider plot visualization.
    
    Args:
        parameter_ranges: Dictionary defining the min, max, and default values for each parameter.
        increments: Number of steps between the min and max values for each parameter.
    """
    # Filter parameters to include only those with a defined 'default' value
    filtered_parameter_ranges = {
        param: ranges for param, ranges in parameter_ranges.items() if 'default' in ranges
    }
    
    # Initialize base parameters with default values
    base_params = {param: ranges['default'] for param, ranges in filtered_parameter_ranges.items()}
    
    # Generate baseline results
    print("Running baseline simulation...")
    baseline_df, baseline_results = run_simulation(base_params)
    
    # Get metrics directly from the DataFrame
    last_row = baseline_df.iloc[-1]
    baseline_total_yield = last_row['total_yield_freshweight']
    baseline_total_profit = last_row['total_profit']
    
    print(f"Baseline total yield: {baseline_total_yield:.4f} t/ha")
    print(f"Baseline total profit: {baseline_total_profit:.2f} €/ha")
    
    # Prepare results storage
    results = []
    
    # Calculate total simulations for overall progress bar
    total_params = len(filtered_parameter_ranges)
    total_simulations = total_params * (increments + 1)
    
    # Create overall progress bar
    print(f"\nRunning {total_simulations} simulations across {total_params} parameters...")
    overall_progress = tqdm(total=total_simulations, desc="Overall Progress", position=0)
    
    # Run sensitivity analysis for each parameter
    for param_name, ranges in filtered_parameter_ranges.items():
        min_value = ranges['min']
        max_value = ranges['max']
        base_value = ranges['default']
        
        # Create evenly spaced increments
        param_values = np.linspace(min_value, max_value, increments + 1)
        
        print(f"\nTesting parameter: {param_name} - Base value: {base_value}")
        for new_value in tqdm(param_values, desc=f"{param_name}", position=1, leave=False):
            # Create parameter set with the varied parameter
            test_params = base_params.copy()
            test_params[param_name] = new_value
            
            # Run simulation
            df, sim_results = run_simulation(test_params)
            
            # Get metrics directly from the DataFrame
            last_row = df.iloc[-1]
            total_yield = last_row['total_yield_freshweight']
            total_profit = last_row['total_profit']
            
            # Calculate percent change from baseline
            yield_pct_change = ((total_yield / baseline_total_yield) - 1) * 100
            profit_pct_change = ((total_profit / baseline_total_profit) - 1) * 100
            
            # Store results
            results.append({
                'parameter': param_name,
                'base_value': base_value,
                'test_value': new_value,
                'percent_change': ((new_value / base_value) - 1) * 100,
                'total_yield': total_yield,
                'total_profit': total_profit,
                'yield_pct_change': yield_pct_change,
                'profit_pct_change': profit_pct_change
            })
            
            # Update overall progress bar
            overall_progress.update(1)
    
    overall_progress.close()
    
    # Convert results to DataFrame
    results_df = pd.DataFrame(results)
    
    # Return results as DataFrame
    return results_df, baseline_total_yield, baseline_total_profit


def create_combined_sensitivity_plots(results_df, output_variables):
    """
    Create combined line plots showing sensitivity analysis results for all parameters.
    
    Args:
        results_df: DataFrame with sensitivity analysis results
        output_variables: List of dictionaries with output variable information
            [{'name': 'yield_pct_change', 'title': 'Yield Change (%)', 'color': 'b'}, ...]
    """
    # Define a list of marker styles to ensure uniqueness
    marker_styles = ['s', 'D', '^', 'v', '<', '>', 'p', '*', 'h', 'H', 'X', 'd']
    
    for output_var in output_variables:
        var_name = output_var['name']
        var_title = output_var['title']
        
        # Create a new figure
        plt.figure(figsize=(12, 8))
        
        # Plot each parameter's sensitivity
        parameters = results_df['parameter'].unique()
        for i, param in enumerate(parameters):
            param_data = results_df[results_df['parameter'] == param]
            marker = marker_styles[i % len(marker_styles)]  # Cycle through marker styles
            plt.plot(
                param_data['percent_change'], 
                param_data[var_name], 
                marker=marker, 
                label=param.replace('_', ' ').title()
            )
        
        # Add labels, title, and legend
        plt.xlabel('Parameter Change (%)')
        plt.ylabel(var_title)
        plt.title(f'Parameter Sensitivity: Impact on {var_title}')
        plt.axhline(y=0, color='k', linestyle='--', alpha=0.5)  # Add horizontal line at y=0
        plt.axvline(x=0, color='k', linestyle='--', alpha=0.5)  # Add vertical line at x=0
        
        # Place the legend horizontally below the plot
        plt.legend(
            title='Parameters', 
            loc='upper center', 
            bbox_to_anchor=(0.5, -0.15), 
            ncol=5,  # Number of columns in the legend
            frameon=False
        )
        
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.xlim(-105, 405)  
        plt.tight_layout()
        
        save_figure(plt.gcf(), f'Sensitivity_Analysis_{var_title.replace(" ", "_")}')        
        plt.show()


def run_sensitivity_analysis_with_specific_ranges(parameter_ranges, increments=10):
    """
    Run sensitivity analysis for each parameter within specific ranges for spider plot visualization.
    
    Args:
        parameter_ranges: Dictionary defining the min, max, and default values for each parameter.
        increments: Number of steps between the min and max values for each parameter.
    """
    # Filter parameters to include only those with a defined 'default' value
    filtered_parameter_ranges = {
        param: ranges for param, ranges in parameter_ranges.items() if 'default' in ranges
    }
    
    # Get model default parameters first
    default_model_params = initialize_simulation('Model Default (Potato)')
    
    # Initialize base parameters with model defaults first, then override with our defined values
    base_params = default_model_params.copy()
    
    # Override with our specified default values
    for param, ranges in filtered_parameter_ranges.items():
        base_params[param] = ranges['default']
        
    # Ensure integer parameters are stored as integers
    if 'start_season' in base_params:
        base_params['start_season'] = int(base_params['start_season'])
    if 'days' in base_params:
        base_params['days'] = int(base_params['days'])
    
    # Generate baseline results
    print("Running baseline simulation...")
    baseline_df, baseline_results = run_simulation(base_params)
    
    # Get metrics directly from the DataFrame
    last_row = baseline_df.iloc[-1]
    baseline_total_yield = last_row['total_yield_freshweight']
    baseline_total_profit = last_row['total_profit']
    
    print(f"Baseline total yield: {baseline_total_yield:.4f} t/ha")
    print(f"Baseline total profit: {baseline_total_profit:.2f} €/ha")
    
    # Prepare results storage
    results = []
    
    # Calculate total simulations for overall progress bar
    total_params = len(filtered_parameter_ranges)
    total_simulations = total_params * (increments + 1)
    
    # Create overall progress bar
    print(f"\nRunning {total_simulations} simulations across {total_params} parameters...")
    overall_progress = tqdm(total=total_simulations, desc="Overall Progress", position=0)
    
    # Run sensitivity analysis for each parameter
    for param_name, ranges in filtered_parameter_ranges.items():
        min_value = ranges['min']
        max_value = ranges['max']
        base_value = ranges['default']
        
        # Create evenly spaced increments
        param_values = np.linspace(min_value, max_value, increments + 1)
        
        print(f"\nTesting parameter: {param_name} - Base value: {base_value}")
        for new_value in tqdm(param_values, desc=f"{param_name}", position=1, leave=False):
            # Create parameter set with the varied parameter
            test_params = base_params.copy()
            test_params[param_name] = new_value
            
            # Ensure integer parameters are stored as integers when modified
            if param_name == 'start_season':
                test_params['start_season'] = int(new_value)
            elif param_name == 'days':
                test_params['days'] = int(new_value)
            
            # Run simulation
            df, sim_results = run_simulation(test_params)
            
            # Get metrics directly from the DataFrame
            last_row = df.iloc[-1]
            total_yield = last_row['total_yield_freshweight']
            total_profit = last_row['total_profit']
            
            # Calculate percent change from baseline
            yield_pct_change = ((total_yield / baseline_total_yield) - 1) * 100
            profit_pct_change = ((total_profit / baseline_total_profit) - 1) * 100
            
            # Store results
            results.append({
                'parameter': param_name,
                'base_value': base_value,
                'test_value': new_value,
                'percent_change': ((new_value / base_value) - 1) * 100,
                'total_yield': total_yield,
                'total_profit': total_profit,
                'yield_pct_change': yield_pct_change,
                'profit_pct_change': profit_pct_change
            })
            
            # Update overall progress bar
            overall_progress.update(1)
    
    overall_progress.close()
    
    # Convert results to DataFrame
    results_df = pd.DataFrame(results)
    
    # Return results as DataFrame
    return results_df, baseline_total_yield, baseline_total_profit


def run_and_return_sensitivity_results(config):
    """
    Run sensitivity analysis and return results as DataFrames.
    
    Args:
        config: Dictionary with sensitivity analysis configuration.
    
    Returns:
        results_df: DataFrame with sensitivity analysis results.
        baseline_yield: Baseline yield value.
        baseline_profit: Baseline profit value.
    """
    # Calculate increments from step size
    range_size = config['upper_bound_pct'] - config['lower_bound_pct']
    increments = int(range_size / config['step_size_pct'])

    # Run the analysis
    print("Running sensitivity analysis...")
    results_df, baseline_yield, baseline_profit = run_spider_sensitivity_analysis(
        param_variation_pct=max(abs(config['lower_bound_pct']), abs(config['upper_bound_pct'])),
        increments=increments,
        parameters_to_test=config['parameters_to_test']
    )

    # Return results as DataFrames
    return results_df, baseline_yield, baseline_profit


def run_sensitivity_analysis():
    """
    Run configurable sensitivity analysis with combined plots for each output variable.
    Only uses parameters with defined default values.
    """
    # Filter parameters to include only those with a defined 'default' value
    filtered_params = {
        param: ranges for param, ranges in parameter_ranges.items() 
        if 'default' in ranges
    }
    
    # Sensitivity configuration
    sensitivity_config = {
        # Use only parameters with defined default values
        'parameters_to_test': list(filtered_params.keys()),
        
        # Sensitivity range and step configuration
        'lower_bound_pct': -50,  # Lower bound for parameter variation (%)
        'upper_bound_pct': 50,   # Upper bound for parameter variation (%)
        'step_size_pct': 25,     # Step size for parameter variation (%)
        
        # Output variables to analyze (results to plot)
        'output_variables': [
            {'name': 'yield_pct_change', 'title': 'Yield Change (%)', 'color': 'b'},
            {'name': 'profit_pct_change', 'title': 'Profit Change (%)', 'color': 'r'}
        ]
    }

    print(f"Running sensitivity analysis for {len(filtered_params)} parameters with defined default values...")
    
    # Run sensitivity analysis and get results
    results_df, baseline_yield, baseline_profit = run_and_return_sensitivity_results(sensitivity_config)

    # Create combined plots for each output variable
    create_combined_sensitivity_plots(results_df, sensitivity_config['output_variables'])

    # Return results as DataFrames
    return results_df, baseline_yield, baseline_profit


if __name__ == "__main__":
    # Run sensitivity analysis with specific ranges
    results_df, baseline_yield, baseline_profit = run_sensitivity_analysis_with_specific_ranges(parameter_ranges, increments=30)
    
    # Create combined plots for the results
    create_combined_sensitivity_plots(results_df, [
        {'name': 'yield_pct_change', 'title': 'Yield Change (%)', 'color': 'b'},
        {'name': 'profit_pct_change', 'title': 'Profit Change (%)', 'color': 'r'}
    ])