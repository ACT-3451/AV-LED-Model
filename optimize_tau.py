import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import time
import os
from datetime import datetime
from main_model import initialize_simulation, run_simulation
from matplotlib.ticker import LogLocator, FuncFormatter

def plot_combined_tau_analysis(results_df):
    """Plot tau analysis results in a single A4 portrait figure."""
    # A4 dimensions in inches (width x height): 8.27 x 11.69
    fig = plt.figure(figsize=(8.27, 11.69), dpi=300)
    gs = gridspec.GridSpec(3, 1, height_ratios=[1, 1, 1], hspace=0.3)
    fig.suptitle('Time Step Analysis for Agrivoltaic Model', fontsize=14)
    
    # Sort dataframes for plotting (by steps_per_day ascending)
    df_sorted = results_df.sort_values('steps_per_day')
    
    # Calculate relative difference from the most detailed simulation
    most_detailed = df_sorted.iloc[-1]  # Highest steps_per_day is most detailed
    df_sorted['yield_relative_diff'] = 100 * (df_sorted['yield'] - most_detailed['yield']) / most_detailed['yield']
    df_sorted['profit_relative_diff'] = 100 * (df_sorted['profit'] - most_detailed['profit']) / most_detailed['profit']
    
    # Format function for steps per day to show as integers when possible
    def format_steps(x, pos):
        if x == int(x):
            return f"{int(x)}"
        return f"{x:.1f}"
    
    # 1. Yield convergence
    ax1 = fig.add_subplot(gs[0])
    ax1.plot(df_sorted['steps_per_day'], df_sorted['yield_relative_diff'], 'o-', color='#0173B2', linewidth=1.5, markersize=2)
    ax1.set_ylabel('Yield Difference (%)')
    ax1.grid(True, linestyle='--', alpha=0.7)
    ax1.set_title('Yield Convergence Analysis', fontsize=12)
    ax1.axhline(y=0, color='black', linestyle='-', alpha=0.5)
    # ax1.axhline(y=-1, color='grey', linestyle='--', alpha=0.7)
    # ax1.axhline(y=1, color='grey', linestyle='--', alpha=0.7)
    ax1.set_xscale('log')
    ax1.set_yscale('symlog', linthresh=0.1)  # Symmetric log scale with linear region near zero
    
    # 2. Profit convergence
    ax2 = fig.add_subplot(gs[1], sharex=ax1)
    ax2.plot(df_sorted['steps_per_day'], df_sorted['profit_relative_diff'], 'o-', color='#029E73', linewidth=1.5, markersize=2)
    ax2.set_ylabel('Profit Difference (%)')
    ax2.grid(True, linestyle='--', alpha=0.7)
    ax2.set_title('Profit Convergence Analysis', fontsize=12)
    ax2.axhline(y=0, color='black', linestyle='-', alpha=0.5)
    ax2.axhline(y=-1, color='grey', linestyle='--', alpha=0.7) 
    ax2.axhline(y=1, color='grey', linestyle='--', alpha=0.7)
    ax2.set_yscale('symlog', linthresh=0.1)  # Symmetric log scale with linear region near zero
    
    # 3. Execution time vs steps per day
    ax3 = fig.add_subplot(gs[2], sharex=ax1)
    ax3.plot(df_sorted['steps_per_day'], df_sorted['execution_time'], 'o-', color='#D55E00', linewidth=1.5, markersize=2)
    ax3.set_ylabel('Execution Time (s)')
    ax3.set_xlabel('Steps per Day')
    ax3.grid(True, linestyle='--', alpha=0.7)
    ax3.set_title('Execution Time vs Steps per Day', fontsize=12)
    
    # Apply common formatting to all plots
    for ax in [ax1, ax2, ax3]:
        # Use LogLocator for logarithmic scale with evenly spaced labels
        ax.xaxis.set_major_formatter(FuncFormatter(format_steps))
        ax.tick_params(axis='x')
        ax.xaxis.set_major_locator(LogLocator(numticks=5))  # Reduce number of ticks

    # Only show x-tick labels on the bottom plot
    for ax in [ax1, ax2]:
        ax.tick_params(labelbottom=False)
    
    plt.tight_layout(rect=[0.02, 0.02, 0.98, 0.97])
    plt.subplots_adjust(bottom=0.1)
    
    return fig

def find_optimal_tau(results_df, error_threshold=0.01):
    """
    Find the optimal tau value based on convergence within an error threshold.
    
    Args:
        results_df: DataFrame with tau analysis results
        error_threshold: Maximum acceptable relative error (default: 1%)
        
    Returns:
        optimal_tau: The recommended tau value
    """
    # Sort by tau (ascending)
    df_sorted = results_df.sort_values('tau')
    
    # Use the most detailed simulation as reference
    most_detailed = df_sorted.iloc[0]
    
    # Calculate relative differences
    for idx, row in df_sorted.iterrows():
        yield_diff = abs((row['yield'] - most_detailed['yield']) / most_detailed['yield'])
        profit_diff = abs((row['profit'] - most_detailed['profit']) / most_detailed['profit'])
        
        # If both yield and profit are within threshold, this is optimal
        if yield_diff <= error_threshold and profit_diff <= error_threshold:
            optimal_tau = row['tau']
            optimal_steps = 1 / optimal_tau
            
            print(f"\nOptimal tau found: {optimal_tau:.6f} ({optimal_steps:.1f} steps per day)")
            print(f"  Yield error: {yield_diff*100:.4f}%")
            print(f"  Profit error: {profit_diff*100:.4f}%")
            print(f"  Execution time: {row['execution_time']:.2f} seconds")
            print(f"  Reference execution time: {most_detailed['execution_time']:.2f} seconds")
            print(f"  Speedup: {most_detailed['execution_time']/row['execution_time']:.2f}x")
            
            return optimal_tau
    
    # If no tau meets the threshold, return the most detailed one
    print("\nNo tau value within error threshold. Using most detailed.")
    return most_detailed['tau']

def run_tau_analysis(crop_name='Model Default (Potato)', tau_values=None):
    """
    Run the simulation with different tau values and analyze the results.
    
    Args:
        crop_name: Name of the crop to simulate
        tau_values: List of tau values to test (or None to use default range)
        
    Returns:
        DataFrame with results for each tau value
    """
    # Default tau values to test if not provided
    if tau_values is None:
        tau_values = np.logspace(np.log10(1/9600), np.log10(1/1), num=100)
    
    # Initialize results storage
    results = []
    
    print(f"Running tau analysis for {crop_name}...")
    
    # Get base parameters
    base_params = initialize_simulation(crop_name)
    
    # Track execution time for each tau
    for tau in tau_values:
        print(f"\nTesting tau = {tau:.6f} ({1/tau:.1f} steps per day)")
        
        # Create parameter set with the current tau
        params = base_params.copy()
        params['tau'] = tau
        
        # Time the execution
        start_time = time.time()
        
        # Run simulation with the current tau
        df, sim_results = run_simulation(params)
        
        # Calculate execution time
        execution_time = time.time() - start_time
        
        # Get metrics from the last row of the DataFrame
        last_row = df.iloc[-1]
        
        # Extract key metrics
        total_yield = last_row['total_yield_freshweight']
        total_profit = last_row['total_profit']
        
        # Store results
        results.append({
            'tau': tau,
            'steps_per_day': 1/tau,
            'yield': total_yield,
            'profit': total_profit,
            'execution_time': execution_time,
            'num_timesteps': len(df)
        })
        
        print(f"  Yield: {total_yield:.4f} t/ha")
        print(f"  Profit: {total_profit:.2f} €/ha")
        print(f"  Execution time: {execution_time:.2f} seconds")
        print(f"  Number of timesteps: {len(df)}")
    
    # Convert results to DataFrame
    results_df = pd.DataFrame(results)
    
    # Plot combined results
    fig = plot_combined_tau_analysis(results_df)
    
    return results_df, fig


def main():
    """Main function to run the tau analysis."""
    # Create 'figures' directory if it doesn't exist
    figures_dir = 'figures'
    os.makedirs(figures_dir, exist_ok=True)
    
    # Run analysis with default tau values
    results_df, fig = run_tau_analysis()
    
    # Generate timestamp for filenames
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Save the figure with descriptive names and timestamp
    fig_name = f"tau_analysis_{timestamp}"
    fig.savefig(os.path.join(figures_dir, f"{fig_name}.pdf"), format='pdf', bbox_inches='tight')
    fig.savefig(os.path.join(figures_dir, f"{fig_name}.png"), format='png', bbox_inches='tight', dpi=300)
    
    print(f"Figures saved in '{figures_dir}' as '{fig_name}.pdf' and '{fig_name}.png'")
    
    # Find optimal tau
    optimal_tau = find_optimal_tau(results_df)
    
    # Return results for further analysis if needed
    return results_df, optimal_tau


if __name__ == "__main__":
    results_df, optimal_tau = main()