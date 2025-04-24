import numpy as np
import matplotlib.pyplot as plt
from main_model import artificial_light_curves

def plot_continuous_artificial_radiation_behavior():
    """
    Plot the behavior of the artificial_light_curves function continuously
    from day 1 to 366 with 96 steps per day. The y-axis shows relative values (0-1),
    normalized by the maximum value across the entire year.
    """
    # Parameters
    steps_per_day = 96
    days = np.arange(1, 366)  # Days of the year (1 to 366)
    time_steps = np.linspace(0, 1, steps_per_day)  # Fraction of the day (0-1)
    
    # Calculate the global maximum radiation value for the entire year
    global_max_radiation = 0
    for doy in days:
        radiation_values = [artificial_light_curves(doy, fod) for fod in time_steps]
        daily_max = max(radiation_values)
        if daily_max > global_max_radiation:
            global_max_radiation = daily_max
    
    # Prepare data for continuous plotting
    continuous_time = []
    continuous_radiation = []
    
    for doy in days:
        # Calculate radiation values for the current day
        radiation_values = [artificial_light_curves(doy, fod) for fod in time_steps]
        # Normalize using the global maximum radiation value
        relative_radiation = [value / global_max_radiation for value in radiation_values]
        
        # Append time and radiation values for the current day
        continuous_time.extend(time_steps + (doy - 1))  # Shift time by day index
        continuous_radiation.extend(relative_radiation)
    
    # Plot
    plt.figure(figsize=(14, 8))
    plt.plot(continuous_time, continuous_radiation, color='orange', alpha=0.7)
    plt.ylim(0, 1.05)
    plt.xlabel('Day of Year (Fractional)')
    plt.ylabel('Relative Radiation (0-1)')
    plt.title('Continuous Behavior of Artificial Radiation Over the Year (Normalized)')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.xlim(0, 365)  # Set x-axis limits to cover the entire year
    plt.savefig('artificial_radiation_behavior_3days.png', dpi=300, bbox_inches='tight')
    plt.savefig('artificial_radiation_behavior_3days.pdf', bbox_inches='tight')
    plt.show()


def plot_continuous_selected_days_artificial_radiation_behavior():
    """
    Plot the behavior of the artificial_light_curves function continuously
    for selected days (e.g., days 73, 74, 75, 76) with 96 steps per day.
    The y-axis shows relative values (0-1), normalized by the maximum value
    across the entire year.
    """
    # Parameters
    steps_per_day = 96
    selected_days = [73, 74, 75, 76]  # Days to plot
    time_steps = np.linspace(0, 1, steps_per_day)  # Fraction of the day (0-1)
    
    # Calculate the global maximum radiation value for the entire year
    global_max_radiation = 0
    for doy in range(1, 367):
        radiation_values = [artificial_light_curves(doy, fod) for fod in time_steps]
        daily_max = max(radiation_values)
        if daily_max > global_max_radiation:
            global_max_radiation = daily_max
    
    # Prepare data for continuous plotting
    continuous_time = []
    continuous_radiation = []
    
    for doy in selected_days:
        # Calculate radiation values for the current day
        radiation_values = [artificial_light_curves(doy, fod) for fod in time_steps]
        # Normalize using the global maximum radiation value
        relative_radiation = [value / global_max_radiation for value in radiation_values]
        
        # Append time and radiation values for the current day
        continuous_time.extend(time_steps + (doy - selected_days[0]))  # Shift time by day index
        continuous_radiation.extend(relative_radiation)
    
    # Plot
    plt.figure(figsize=(14, 8))
    plt.plot(continuous_time, continuous_radiation, color='orange', alpha=0.7)
    plt.ylim(0, 1.05)
    plt.xlabel('Day of Year (Fractional)')
    plt.ylabel('Relative Radiation (0-1)')
    plt.title('Continuous Behavior of Artificial Radiation for Selected Days (Normalized)')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.xlim(0, selected_days[-1] - selected_days[0])  # Set x-axis limits to cover the selected days
    plt.savefig('artificial_radiation_behavior_year.png', dpi=300, bbox_inches='tight')
    plt.savefig('artificial_radiation_behavior_year.pdf', bbox_inches='tight')
    plt.show()


if __name__ == "__main__":
    plot_continuous_selected_days_artificial_radiation_behavior()
    plot_continuous_artificial_radiation_behavior()