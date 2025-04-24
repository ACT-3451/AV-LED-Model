# xTODO: make a graph, that shows the contribution of the sunlit area, the shaded area suna and shaded area led, to the total biomass production
# TODO: get solar power for entrie year, not just season
# TODO: calculate relative yield loss
# TODO: RUE modeling shaded_rue~amount_of_light linear to sun_rue~amount_of_light
# TODO: reserach where the crops run into light saturation, and how to model this
# BUG: if leds are on, the LAI contribution of the LEDs still at 0.1, should be zerro
# TODO: Dynamic electricity pricing
# TODO: 

# library imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from datetime import datetime
from load_crop_parameters import get_param_df


def pot_growth(dtr, f_par, rue, k, lai):
    """
    Calculate daily biomass increment.
    
    Args:
        dtr: Daily Total Radiation (kWh/m²/tau)
        par: fraction of PAR in total radiation (default=1.0)
        rue: Radiation Use Efficiency (g/kWh)
        k: Light extinction coefficient (-)
        lai: Leaf Area Index (m²/m²)

    Returns:
        float: Daily biomass increment (g/m²/tau)
    """
    return dtr * f_par * rue * (1 - np.exp(-k * lai))


def g_m2_to_t_ha(g_m2): # Convert g/m² to t/ha.
    return g_m2 / 100.0 # t/ha


def artificial_light_curves(doy, fod, min_solar=2.77778, max_solar=5.55556, phase_shift=80, min_daylength=8, max_daylength=16):
    """
    Args:
        doy: Day of year (1-365)
        fod: Fraction of day (0-1)
        min_solar: Minimum solar radiation (kWh/m²/day) during winter
        max_solar: Maximum solar radiation (kWh/m²/day) during summer
        phase_shift: Day offset to align seasonal curve (default=80 for Northern hemisphere)
        min_daylength: Shortest day length in hours (winter)
        max_daylength: Longest day length in hours (summer)
    """
    seasonal_factor = np.sin(2.0 * np.pi * (doy - phase_shift) / 365.0) # Calculate seasonal factor (varies from -1 to 1)
    
    # Calculate maximum possible solar radiation for the day
    rad_amplitude = (max_solar - min_solar) / 2
    mean_rad = (max_solar + min_solar) / 2
    max_day_solar = mean_rad + rad_amplitude * seasonal_factor
    
    # Convert day lengths from hours to fraction of day
    min_day_fraction = min_daylength / 24
    max_day_fraction = max_daylength / 24
    
    day_fraction = min_day_fraction + (max_day_fraction - min_day_fraction) * (seasonal_factor + 1) / 2     # Calculate day length for this day (as fraction of day)
    half_day = day_fraction / 2                             # Calculate half the day length

    # Calculate sunrise and sunset times
    sunrise = 0.5 - half_day
    sunset = 0.5 + half_day
    
    if fod < sunrise or fod > sunset:                       # Check if we're within daylight hours
        return 0.0                                          # No radiation during night
    
    normalized_fod = (fod - sunrise) / (sunset - sunrise)   # Normalize time to 0-1 range between sunrise and sunset
    reduction = np.sin(np.pi * normalized_fod) ** 2         # Reduction factor based on sun position during the day
    return max_day_solar * reduction                        # Final radiation


def should_led_be_on(fraction_of_day, fixed_intervals):
    # Decide if LEDs should be on based on the chosen schedule option.
    for start, end in fixed_intervals:
        if start <= fraction_of_day < end:
            return True
    return False


def input_validation(led_start_1, led_end_1, led_start_2, led_end_2):
    if not (0 <= led_start_1 <= 1 and 0 <= led_end_1 <= 1 and led_start_1 <= led_end_1):
        raise ValueError("Error in LED schedule: led_start_1 and led_end_1 are out of bounds or invalid (start > end).")

    if not (0 <= led_start_2 <= 1 and 0 <= led_end_2 <= 1 and led_start_2 <= led_end_2):
        raise ValueError("Error in LED schedule: led_start_2 and led_end_2 are out of bounds or invalid (start > end).")

    if not (led_end_1 <= led_start_2 or led_end_2 <= led_start_1):  # Check for overlapping intervals
        raise ValueError("Error in LED schedule: LED intervals overlap.")


def initialize_simulation(crop_name):
    """Initialize simulation parameters based on the selected crop."""
    print(f"Loading parameters for {crop_name}...")
    param_df = get_param_df(crop_name)

    if param_df is None:
        print("Error loading crop parameters. Simulation stopped.")
        raise ValueError("Failed to load crop parameters")

    # Convert parameter DataFrame to a dictionary for easier access
    param_dict = param_df['Value'].to_dict()
    
    # Simulation parameters
    tau                     = 1/960  # time step per day (1/96 days = 15 min)
    
    # Season parameters from crop data
    start_season            = int(param_dict['start_season'])  # start of the season (day of year)
    days                    = int(param_dict['days'])  # number of days to simulate
    max_dtr                 = 13.8889  # Maximum daily total radiation (kWh/m²/day)
    min_dtr                 = 5.55556  # Minimum daily total radiation (kWh/m²/day)
    f_par                   = 0.5  # Fraction of PAR in total radiation
    init_lai                = 0.1  # Initial Leaf Area Index (m²/m²)

    # === Crop parameters, SUN ===
    rue                     = param_dict['rue']  # Radiation Use Efficiency (g/kWh)
    k                       = param_dict['k']  # Light extinction coefficient
    lai_sun                 = init_lai  # initial Leaf Area Index
    sla                     = param_dict['sla']  # Specific Leaf Area (m²/g)
    alloc_leaf              = param_dict['alloc_leaf']  # biomass allocation to leaves
    alloc_yield             = param_dict['alloc_yield']  # biomass allocation to yield
    max_lai                 = param_dict['max_lai']  # Maximum LAI (m²/m²)
    dmc                     = param_dict['dmc']  # Dry matter content

    
    # === Economic parameters ===
    yield_profit = 400  # Price of yield (euro/t)
    energy_selling_price = 0.07  # Price of energy (euro/kWh)

    # AV system parameters
    shaded_light_fraction   = 0.25  # Fraction of sun light that reaches the shaded area (fraction: 0-1) 
    panel_coverage          = 0.32  # Fraction of field covered by panels (fraction: 0-1)
    led_light_amount_kWh_d_target = 4  # Amount of light from LED (kWh/m²/day), ONLY PAR
    panel_efficiency        = 0.21  # Efficiency of the panels (fraction: 0-1)
    led_efficiency          = 0.5  # Efficiency of LED lights (fraction: 0-1)

    # === Crop parameters, SHADED ===
    lai_shade_sun = init_lai  # initial Leaf Area Index for shaded area
    lai_shade_led = init_lai  # initial Leaf Area Index for shaded area with sun
    rue_shaded = param_dict['rue_shaded']  # Radiation Use Efficiency for shaded area
    sla_shaded = param_dict['sla_shaded']  # Specific Leaf Area for shaded area
    alloc_leaf_shaded = param_dict['alloc_leaf_shaded']  # biomass allocation to leaves
    alloc_yield_shaded = param_dict['alloc_yield_shaded']  # biomass allocation to yield
    dmc_shaded = param_dict['dmc_shaded']  # Dry matter content for shaded area

    # AV system parameters
    buy_energy_from_grid = False  # Whether to allow buying energy from the grid
    grid_energy_price = 0.09  # Price of energy from the grid (€/kWh)

    # === LED schedule ===
    led_start_1 = 8/24  # Start time for LED lights (fraction of day)
    led_end_1 = 16/24  # End time for LED lights (fraction of day)
    led_start_2 = 0/24  # Start time for LED lights (fraction of day)
    led_end_2 = 0/24  # End time for LED lights (fraction of day)
    fixed_intervals = [(led_start_1, led_end_1), (led_start_2, led_end_2)]
    
    input_validation(led_start_1, led_end_1, led_start_2, led_end_2)  # Validate LED schedule inputs
    
    # Print summary of loaded parameters
    print(f"\nSimulation will run with the following parameters for {crop_name}:\n")
    print(f"Start season\t\t\tDay {start_season}, Duration: {days} days")
    print(f"\n\t\t\t\tsun\t\tshade")
    print(f"Radiation Use Efficiency\t{rue:.2f} g/kWh\t{rue_shaded:.2f} g/kWh")
    print(f"Light extinction coefficient\t{k:.2f}\t\tsee 'sun'")
    print(f"Specific Leaf Area\t\t{sla:.4f} m²/g\t{sla_shaded:.4f} m²/g")
    print(f"Maximum LAI\t\t\t{max_lai:.2f} m²/m²\tsee 'sun'")
    print(f"Dry matter content\t\t{dmc:.2f}\t\t{dmc_shaded:.2f}\n\n")
    
    # Return all initialized parameters as a dictionary
    return {
        'param_dict': param_dict,
        'tau': tau,
        'start_season': start_season,
        'days': days,
        'max_dtr': max_dtr,
        'min_dtr': min_dtr,
        'f_par': f_par,
        'init_lai': init_lai,
        'rue': rue,
        'k': k,
        'lai_sun': lai_sun,
        'sla': sla,
        'alloc_leaf': alloc_leaf,
        'alloc_yield': alloc_yield,
        'max_lai': max_lai,
        'dmc': dmc,
        'yield_profit': yield_profit,
        'energy_selling_price': energy_selling_price,
        'shaded_light_fraction': shaded_light_fraction,
        'panel_coverage': panel_coverage,
        'led_light_amount_kWh_d_target': led_light_amount_kWh_d_target,
        'panel_efficiency': panel_efficiency,
        'led_efficiency': led_efficiency,
        'lai_shade_sun': lai_shade_sun,
        'lai_shade_led': lai_shade_led,
        'rue_shaded': rue_shaded,
        'sla_shaded': sla_shaded,
        'alloc_leaf_shaded': alloc_leaf_shaded,
        'alloc_yield_shaded': alloc_yield_shaded,
        'dmc_shaded': dmc_shaded,
        'buy_energy_from_grid': buy_energy_from_grid,
        'grid_energy_price': grid_energy_price,
        'fixed_intervals': fixed_intervals
    }


def run_simulation(params):
    """Run the simulation with the provided parameters and return the results."""
    simulation_results = []  # Initialize an empty list to store simulation results
    
    # Initialize accumulation variables
    accumulated_total_biomass = 0.0  # Initial accumulated biomass (g/m²)
    accumulated_leaf_biomass = 0.0  # Initial accumulated leaf biomass (g/m²)
    accumulated_yield_biomass = 0.0  # Initial accumulated yield biomass (g/m²)
    
    accumulated_total_biomass_sun = accumulated_total_biomass  # Initial accumulated biomass (g/m²)
    accumulated_leaf_biomass_sun = accumulated_leaf_biomass  # Initial accumulated leaf biomass (g/m²)
    accumulated_yield_biomass_sun = accumulated_yield_biomass  # Initial accumulated yield biomass (g/m²)
    
    accumulated_total_biomass_shade_sun = accumulated_total_biomass  # Initial accumulated biomass (g/m²)
    accumulated_leaf_biomass_shade_sun = accumulated_leaf_biomass  # Initial accumulated leaf biomass (g/m²)
    accumulated_yield_biomass_shade_sun = accumulated_yield_biomass  # Initial accumulated yield biomass (g/m²)
    
    accumulated_total_biomass_shade_led = accumulated_total_biomass  # Initial accumulated biomass (g/m²)
    accumulated_leaf_biomass_shade_led = accumulated_leaf_biomass  # Initial accumulated leaf biomass (g/m²)
    accumulated_yield_biomass_shade_led = accumulated_yield_biomass  # Initial accumulated yield biomass (g/m²)
    
    produced_energy = 0.0  # Initial produced energy (kWh/m²/tau)
    pot_produced_energy = 0.0  # Initial potential produced energy (kWh/m²/tau)
    net_produced_energy = 0.0  # Initial net energy production (kWh/m²/tau)
    power_consumed = 0.0  # Initial power consumed by LEDs (kWh/m²/tau)
    
    energy_from_grid = 0.0  # Energy bought from the grid (kWh/ha)
    energy_consumed_from_grid = 0.0  # Energy consumed from the grid (kWh/ha)
    total_grid_energy_cost = 0.0  # Total cost of energy bought from grid (€)
    
    # Extract parameters
    tau = params['tau']
    start_season = params['start_season']
    days = params['days']
    max_dtr = params['max_dtr']
    min_dtr = params['min_dtr']
    f_par = params['f_par']
    lai_sun = params['lai_sun']
    sla = params['sla']
    alloc_leaf = params['alloc_leaf']
    alloc_yield = params['alloc_yield']
    max_lai = params['max_lai']
    dmc = params['dmc']
    panel_coverage = params['panel_coverage']
    shaded_light_fraction = params['shaded_light_fraction']
    led_light_amount_kWh_d_target = params['led_light_amount_kWh_d_target']
    panel_efficiency = params['panel_efficiency']
    led_efficiency = params['led_efficiency']
    lai_shade_sun = params['lai_shade_sun']
    lai_shade_led = params['lai_shade_led']
    rue = params['rue']
    rue_shaded = params['rue_shaded']
    sla_shaded = params['sla_shaded']
    alloc_leaf_shaded = params['alloc_leaf_shaded']
    alloc_yield_shaded = params['alloc_yield_shaded']
    dmc_shaded = params['dmc_shaded']
    buy_energy_from_grid = params['buy_energy_from_grid']
    grid_energy_price = params['grid_energy_price']
    fixed_intervals = params['fixed_intervals']
    k = params['k']
    yield_profit = params['yield_profit']
    energy_selling_price = params['energy_selling_price']
    
    # Run simulation
    for day in range(start_season, start_season + days):  # Loop over days of the season
        for step in np.arange(0, 1, tau):  # Loop over time steps in each day
            
            # === SUN-LIT PART OF THE AGRIVOLTAICS SYSTEM ===
            light_sun = artificial_light_curves(day, step, min_dtr, max_dtr)  # Calculate light and biomass production
            biomass_sun = pot_growth(light_sun, f_par, rue, k, lai_sun) * tau  # Daily biomass increment (g/m²/tau)
            lai_sun = lai_sun + (biomass_sun * sla * alloc_leaf) * (1 - lai_sun / max_lai)

            # Update accumulated biomass
            accumulated_total_biomass_sun += g_m2_to_t_ha(biomass_sun) * (1-panel_coverage)
            accumulated_leaf_biomass_sun += g_m2_to_t_ha(biomass_sun * alloc_leaf) * (1-panel_coverage)
            accumulated_yield_biomass_sun += g_m2_to_t_ha(biomass_sun * alloc_yield) * (1-panel_coverage)

            # === SHADED PART OF THE AGRIVOLTAICS SYSTEM, SUN ===
            light_shade_sun = artificial_light_curves(day, step, min_dtr, max_dtr) * shaded_light_fraction  # Calculate light and biomass production
            biomass_shade_sun = pot_growth(light_shade_sun, f_par, rue, k, lai_shade_sun) * tau  # Daily biomass increment (g/m²/tau)
            lai_shade_sun += (biomass_shade_sun * sla * alloc_leaf) * (1 - lai_shade_sun / max_lai)

            # Update accumulated biomass
            accumulated_total_biomass_shade_sun += g_m2_to_t_ha(biomass_shade_sun) * panel_coverage
            accumulated_leaf_biomass_shade_sun += g_m2_to_t_ha(biomass_shade_sun * alloc_leaf) * panel_coverage
            accumulated_yield_biomass_shade_sun += g_m2_to_t_ha(biomass_shade_sun * alloc_yield) * panel_coverage

            # === SHADED PART OF THE AGRIVOLTAICS SYSTEM, LED ===
            led_on = should_led_be_on(step, fixed_intervals)  # Check if LEDs should be on based on the schedule option
            
            # Calculate potential energy production from panels
            pot_energy_production = (light_sun * panel_efficiency) * panel_coverage * tau * 10000  # Convert kWh/m²/tau to kWh/ha/tau
            
            if led_on:
                # First, calculate how much LED light we need to reach the target intensity
                # considering the existing sunlight in the shaded area
                max_led_light_needed = max(0, led_light_amount_kWh_d_target - light_shade_sun)
                
                # Calculate the potential LED power consumption based on adjusted light needed
                led_light_amount = max_led_light_needed * tau  # Light amount adjusted for time step
                
                potential_led_power_consumption = (led_light_amount / led_efficiency) * 10000 * panel_coverage  # Convert to kWh/ha/tau, only for panel-covered area

                # Check if there is enough energy produced by the panels
                if pot_energy_production >= potential_led_power_consumption:
                    # Enough energy from panels to power LEDs
                    led_power_consumption = potential_led_power_consumption
                    energy_from_grid_step = 0.0
                    # Use the adjusted light amount
                    shade_amount_of_light_led = max_led_light_needed
                else:
                    # Not enough energy from panels
                    if buy_energy_from_grid:
                        # Buy the remaining energy from the grid
                        led_power_consumption = potential_led_power_consumption
                        energy_from_grid_step = potential_led_power_consumption - pot_energy_production  # kWh/ha/tau
                        energy_from_grid += energy_from_grid_step
                        total_grid_energy_cost += energy_from_grid_step * grid_energy_price
                        # Use the adjusted light amount
                        shade_amount_of_light_led = max_led_light_needed
                    else:
                        # Reduce LED power consumption to match available energy
                        led_power_consumption = pot_energy_production
                        energy_from_grid_step = 0.0
                        # Reduce light amount proportionally
                        reduction_factor = pot_energy_production / potential_led_power_consumption
                        shade_amount_of_light_led = max_led_light_needed * reduction_factor

                power_consumed += led_power_consumption
                # Recalculate biomass production based on adjusted light amount
                biomass_shade_led = pot_growth(shade_amount_of_light_led, 1, rue_shaded, k, lai_shade_led) * tau
                lai_shade_led += (biomass_shade_led * sla_shaded * alloc_leaf_shaded) * (1 - lai_shade_led / max_lai)

                accumulated_total_biomass_shade_led += g_m2_to_t_ha(biomass_shade_led) * panel_coverage
                accumulated_leaf_biomass_shade_led += g_m2_to_t_ha(biomass_shade_led * alloc_leaf_shaded) * panel_coverage
                accumulated_yield_biomass_shade_led += g_m2_to_t_ha(biomass_shade_led * alloc_yield_shaded) * panel_coverage
            
            else:  # LEDs are off, there is no contribution from LEDs
                biomass_shade_led = 0.0
                led_power_consumption = 0.0
                shade_amount_of_light_led = 0.0
                energy_from_grid_step = 0.0

            # Add shade SUN and LED biomass production
            lai_shade = lai_shade_sun + lai_shade_led  # Calculate combined LAI for the shaded area

            if lai_shade > max_lai:  # Scale down contributions if combined LAI exceeds max_lai
                scaling_factor = max_lai / lai_shade
                lai_shade_sun *= scaling_factor
                lai_shade_led *= scaling_factor

            accumulated_total_biomass_shade = accumulated_total_biomass_shade_sun + accumulated_total_biomass_shade_led
            accumulated_leaf_biomass_shade = accumulated_leaf_biomass_shade_sun + accumulated_leaf_biomass_shade_led
            accumulated_yield_biomass_shade = accumulated_yield_biomass_shade_sun + accumulated_yield_biomass_shade_led
            
            # === ELECTRICITY PRODUCTION ===
            # Calculate net energy production
            net_energy_production = pot_energy_production - led_power_consumption
            net_produced_energy += net_energy_production
            energy_consumed_from_grid += energy_from_grid_step  # Add energy from grid to the total

            # Calculate yield metrics for this time step
            base_yield_freshweight = (accumulated_yield_biomass_sun * (1/dmc)) + (accumulated_yield_biomass_shade_sun * (1/dmc_shaded))
            led_yield_freshweight = accumulated_yield_biomass_shade_led * (1/dmc_shaded)
            total_yield_freshweight = base_yield_freshweight + led_yield_freshweight
            
            # Calculate profit metrics for this time step
            base_yield_profit = base_yield_freshweight * yield_profit
            led_yield_profit = led_yield_freshweight * yield_profit
            yield_profit_value = base_yield_profit + led_yield_profit
            energy_profit_value = net_produced_energy * energy_selling_price - total_grid_energy_cost
            total_profit = yield_profit_value + energy_profit_value

            # Append results to the list
            simulation_results.append({
                "doy": day,
                "fod": step,
                "sun_amount_of_light": light_sun,
                "sun_biomass_production": biomass_sun,
                "sun_lai": lai_sun,
                "sun_accumulated_total_biomass": accumulated_total_biomass_sun,
                "sun_accumulated_leaf_biomass": accumulated_leaf_biomass_sun,
                "sun_accumulated_yield_biomass": accumulated_yield_biomass_sun,
            
                # Shaded area (sunlight contribution)
                "shade_amount_of_light_sun": light_shade_sun,
                "shade_biomass_production_sun": biomass_shade_sun,
                "shade_accumulated_total_biomass_sun": accumulated_total_biomass_shade_sun,
                "shade_accumulated_leaf_biomass_sun": accumulated_leaf_biomass_shade_sun,
                "shade_accumulated_yield_biomass_sun": accumulated_yield_biomass_shade_sun,
            
                # Shaded area (LED contribution)
                "shade_amount_of_light_led": shade_amount_of_light_led,
                "shade_biomass_production_led": biomass_shade_led,
                "shade_accumulated_total_biomass_led": accumulated_total_biomass_shade_led,
                "shade_accumulated_leaf_biomass_led": accumulated_leaf_biomass_shade_led,
                "shade_accumulated_yield_biomass_led": accumulated_yield_biomass_shade_led,
            
                # Shaded area (total contribution)
                "shade_total_amount_of_light": light_shade_sun + shade_amount_of_light_led,
                "shade_total_biomass_production": biomass_shade_sun + biomass_shade_led,
                "shade_accumulated_total_biomass": accumulated_total_biomass_shade,
                "shade_accumulated_leaf_biomass": accumulated_leaf_biomass_shade,
                "shade_accumulated_yield_biomass": accumulated_yield_biomass_shade,
            
                # Shaded area LAI
                "shade_lai_sun": lai_shade_sun,
                "shade_lai_led": lai_shade_led,
                "shade_lai": lai_shade,

                # Energy metrics
                "net_produced_energy": net_produced_energy,
                "led_power_consumption": led_power_consumption,
                "power_consumed": power_consumed,
                "energy_from_grid": energy_from_grid_step,
                "energy_consumed_from_grid": energy_consumed_from_grid,
                "pot_energy_production": pot_energy_production,
                "led_on": led_on,
                
                # Yield and profit metrics
                "base_yield_freshweight": base_yield_freshweight,
                "led_yield_freshweight": led_yield_freshweight,
                "total_yield_freshweight": total_yield_freshweight,
                "base_yield_profit": base_yield_profit,
                "led_yield_profit": led_yield_profit,
                "yield_profit_value": yield_profit_value,
                "energy_profit_value": energy_profit_value,
                "total_profit": total_profit
            })

    df = pd.DataFrame(simulation_results)  # Convert results to a DataFrame
    
    # Additional results to return
    additional_results = {
        'total_grid_energy_cost': total_grid_energy_cost,
        'dmc': dmc,
        'dmc_shaded': dmc_shaded,
        'yield_profit': yield_profit,
        'energy_selling_price': energy_selling_price,
        'grid_energy_price': grid_energy_price
    }
    
    return df, additional_results


def save_figure(fig, base_name):
    """Save a figure as both PDF and PNG in high resolution."""
    figures_dir = 'figures'
    os.makedirs(figures_dir, exist_ok=True)  # Ensure the directory exists
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    fig.savefig(os.path.join(figures_dir, f"{base_name}_{timestamp}.pdf"), format='pdf', bbox_inches='tight')
    fig.savefig(os.path.join(figures_dir, f"{base_name}_{timestamp}.png"), format='png', bbox_inches='tight', dpi=300)
    print(f"Figure saved as '{base_name}_{timestamp}.pdf' and '{base_name}_{timestamp}.png' in '{figures_dir}'.")


def visualize_economic_performance(df, results):
    """Create an economic performance visualization using pre-calculated metrics."""
    grid_energy_price = results['grid_energy_price']
    energy_selling_price = results['energy_selling_price']
    total_grid_energy_cost = results['total_grid_energy_cost']
    
    # Get values from the last row of the DataFrame
    last_row = df.iloc[-1]
    
    # Get pre-calculated values
    base_yield_freshweight = last_row['base_yield_freshweight']
    led_yield_freshweight = last_row['led_yield_freshweight']
    total_yield_freshweight = last_row['total_yield_freshweight']
    
    base_yield_profit = last_row['base_yield_profit']
    led_yield_profit = last_row['led_yield_profit']
    yield_profit_value = last_row['yield_profit_value']
    
    energy_profit_value = last_row['energy_profit_value']
    total_profit = last_row['total_profit']
    
    # Calculate energy values for the energy breakdown bar
    energy_to_grid = last_row['net_produced_energy']  # Energy exported to grid
    energy_consumed = last_row['power_consumed']  # Energy consumed by LEDs
    energy_from_grid_total = last_row['energy_consumed_from_grid']  # Total energy bought from grid
    
    # Calculate total energy for scaling the plot
    energy_produced = energy_consumed + energy_to_grid  # Total energy produced by panels
    total_energy = max(energy_consumed + energy_to_grid, energy_from_grid_total)

    # Create enhanced bar chart for profit visualization
    fig, ax1 = plt.subplots(figsize=(14, 8))

    # First axis - Profit values (€/ha)
    profit_categories = ['Components']

    bars_base_yield = ax1.bar(profit_categories, base_yield_profit, color='#256D1B', linewidth=0, width=0.7)
    bars_led_yield = ax1.bar(profit_categories, led_yield_profit, bottom=base_yield_profit, color='#B9FF8B', linewidth=0, width=0.7)

    if energy_profit_value >= 0:
        bars_energy = ax1.bar(profit_categories, energy_profit_value, bottom=yield_profit_value, color='#FAD13C', linewidth=0, width=0.7)
    else:
        bars_energy = ax1.bar(profit_categories, energy_profit_value, color='#A4243B', linewidth=0, width=0.7)

    base_yield_label_y = base_yield_profit / 2
    ax1.text(0, base_yield_label_y, f'Crop Yield\n{base_yield_profit:,.0f} €\n({base_yield_freshweight:,.3f} t/ha)',
             ha='center', va='center', fontweight='bold', color='white')

    led_yield_label_y = base_yield_profit + (led_yield_profit / 2)
    ax1.text(0, led_yield_label_y, f'LED Yield\n{led_yield_profit:,.0f} €\n({led_yield_freshweight:,.3f} t/ha)',
             ha='center', va='center', fontweight='bold', color='black')

    if energy_profit_value >= 0:
        energy_label_y = yield_profit_value + (energy_profit_value / 2)
        ax1.text(0, energy_label_y, f'Solar Energy sold\n+{energy_profit_value:,.0f} €', ha='center', va='center',
                 fontweight='bold', color='black')
    else:
        energy_label_y = energy_profit_value / 2
        ax1.text(0, energy_label_y, f'Energy\n{energy_profit_value:,.0f} €', ha='center', va='center',
                 fontweight='bold', color='white')

    ax1.text(0, yield_profit_value + energy_profit_value + 10, f'Total\n{total_profit:,.0f} €',
             ha='center', va='bottom', fontweight='bold', color='black')

    # Set up second y-axis for energy values (kWh/ha)
    ax2 = ax1.twinx()

    energy_category = ['Energy Breakdown']

    # Power consumed from the grid (negative bar)
    if energy_from_grid_total > 0:
        bars_from_grid = ax2.bar(energy_category, -energy_from_grid_total, color='#ff374e', linewidth=0, width=0.7)
        ax2.text(1, -energy_from_grid_total / 2,
                 f'GRID IMPORT\n{energy_from_grid_total:,.0f} kWh/ha\n'
                 f'Cost: {total_grid_energy_cost:,.0f} €\n'
                 f'(Buy price: {grid_energy_price:.2f} €/kWh)',
                 ha='center', va='center', fontweight='bold', color='white')

    # Power exported to the grid (placed below self-consumed in the bar)
    if energy_to_grid > 0:
        bars_to_grid = ax2.bar(energy_category, energy_to_grid, color='#2ECC71', linewidth=0, width=0.7)
        grid_revenue = energy_to_grid * energy_selling_price
        ax2.text(1, energy_to_grid / 2,
                 f'GRID EXPORT\n{energy_to_grid:,.0f} kWh/ha\nRevenue: +{grid_revenue:,.0f} €',
                 ha='center', va='center', fontweight='bold', color='black')

    # Power produced but consumed by LEDs (stacked on grid export)
    if energy_consumed > 0:
        bars_consumed = ax2.bar(energy_category, energy_consumed, bottom=energy_to_grid, color='none',
                                edgecolor='black', linewidth=1, linestyle="--", width=0.7)
        opportunity_cost = energy_consumed * energy_selling_price
        ax2.text(1, energy_to_grid + (energy_consumed / 2),
                 f'SELF-CONSUMED ENERGY\n{energy_consumed:,.0f} kWh/ha\n'
                 f'Opportunity cost: {opportunity_cost:,.0f} €\n'
                 f'(Potential revenue at {energy_selling_price:.2f} €/kWh)',
                 ha='center', va='center', fontweight='bold', color='black')


    net_energy_balance = energy_to_grid - energy_from_grid_total
    balance_color = 'green' if net_energy_balance >= 0 else 'red'
    balance_sign = '+' if net_energy_balance >= 0 else ''
    ax2.text(1, total_energy * 1.15, f'Net Grid Balance: {balance_sign}{net_energy_balance:,.0f} kWh/ha',
             ha='center', va='bottom', fontweight='bold')

    current_ymin, current_ymax = ax2.get_ylim()
    ax2.set_ylim(min(current_ymin, -energy_from_grid_total * 1.1),
                 max(current_ymax, (energy_consumed + energy_to_grid) * 1.2))

    ax1.set_title('Agrivoltaic System Economic Performance', fontsize=18, pad=20)
    ax1.set_ylabel('Profit (€/ha)', fontsize=12)
    ax1.yaxis.grid(True, linestyle='--', alpha=0.7)
    ax1.axhline(y=0, color='black', linestyle='-', linewidth=1.0)
    ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:,.0f} €'))

    ax2.set_ylabel('Energy (kWh/ha)', fontsize=12)
    ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:,.0f}'))

    plt.xticks([0, 1], ['Profit\nComponents', 'Energy\nBreakdown'])
    ax1.set_facecolor('#f8f9fa')
    fig.patch.set_facecolor('white')


    plt.tight_layout(rect=[0, 0.03, 1, 0.97])

    # Save the figure
    save_figure(fig, "economic_performance")

    plt.show()


def visualize_simulation_results(df):
    """Create plots showing the simulation results."""
    fig, axes = plt.subplots(nrows=4, ncols=1, figsize=(15, 20), sharex=True)

    # Combined: Amount of Light
    axes[0].plot(df['doy'] + df['fod'], df['sun_amount_of_light'], label='Sunlit Area - Amount of Light', color='#fec44f')  # Dark Yellow
    axes[0].plot(df['doy'] + df['fod'], df['shade_total_amount_of_light'], label='Shaded Area - Total Light', color='black', linestyle=':')  # Purple
    axes[0].plot(df['doy'] + df['fod'], df['shade_amount_of_light_sun'], label='Shaded Area - Sunlight', color='#ec7014')  # Orange
    axes[0].plot(df['doy'] + df['fod'], df['shade_amount_of_light_led'], label='Shaded Area - LED Light', color='#e41f02')  # Red
    axes[0].set_ylabel('Amount of Light (kWh/m²)')
    axes[0].legend(loc='upper left', framealpha=0.4)
    axes[0].grid()

    # Combined: Biomass Production
    axes[1].plot(df['doy'] + df['fod'], df['sun_biomass_production'], label='Sunlit Area - Biomass Production', color='#9ebcda')
    axes[1].plot(df['doy'] + df['fod'], df['shade_biomass_production_sun'], label='Shaded Area - Sun Biomass', color='#8c6bb1')
    axes[1].plot(df['doy'] + df['fod'], df['shade_biomass_production_led'], label='Shaded Area - LED Biomass', color='#88419d')
    axes[1].plot(df['doy'] + df['fod'], df['shade_total_biomass_production'], label='Shaded Area - Total Biomass', color='#6e016b', linestyle=':')
    axes[1].set_ylabel('Biomass Production (g/m²/tau)')
    axes[1].legend(loc='upper left', framealpha=0.4)
    axes[1].grid()

    # Combined: Accumulated Biomass, Leaf Weight, and Yield
    axes[2].plot(df['doy'] + df['fod'], df['sun_accumulated_total_biomass'], label='Sunlit Area - Accumulated Biomass', color='#fe9929', linestyle=':')
    axes[2].plot(df['doy'] + df['fod'], df['sun_accumulated_leaf_biomass'], label='Sunlit Area - Leaf Weight', color='#d95f0e')
    axes[2].plot(df['doy'] + df['fod'], df['sun_accumulated_yield_biomass'], label='Sunlit Area - Yield', color='#993404')
    axes[2].plot(df['doy'] + df['fod'], df['shade_accumulated_total_biomass'], label='Shaded Area - Accumulated Biomass', color='#9e9ac8', linestyle=':')
    axes[2].plot(df['doy'] + df['fod'], df['shade_accumulated_leaf_biomass'], label='Shaded Area - Leaf Weight', color='#756bb1')
    axes[2].plot(df['doy'] + df['fod'], df['shade_accumulated_yield_biomass'], label='Shaded Area - Yield', color='#54278f')
    axes[2].set_ylabel('Dry Matter (t/ha)')
    axes[2].legend(loc='upper left', framealpha=0.4)
    axes[2].grid()

    # Combined: Leaf Area Index (LAI)
    axes[3].plot(df['doy'] + df['fod'], df['sun_lai'], label='Sunlit Area - LAI', color='#ffc047')
    axes[3].plot(df['doy'] + df['fod'], df['shade_lai_sun'], label='Shaded Area - LAI (Sun)', color='#addd8e')
    axes[3].plot(df['doy'] + df['fod'], df['shade_lai_led'], label='Shaded Area - LAI (LED)', color='#41ab5d')
    axes[3].plot(df['doy'] + df['fod'], df['shade_lai'], label='Shaded Area - Total LAI', color='#005a32', linestyle=':')
    axes[3].set_ylabel('LAI (m²/m²)')
    axes[3].legend(loc='upper left', framealpha=0.4)
    axes[3].grid()

    # Final adjustments
    plt.xlabel('Day of Year + Fraction of Day')
    plt.tight_layout()

    # Save the figure
    save_figure(fig, "simulation_results")

    plt.show()


def main():
    """Main function to run the simulation and visualize results."""
    # Select crop and load parameters
    crop_name = 'Model Default (Potato)'  # Change this to any supported crop
    
    # Initialize simulation parameters
    params = initialize_simulation(crop_name)
    
    # Run simulation
    df, results = run_simulation(params)
    
    # Visualize results
    visualize_economic_performance(df, results)
    visualize_simulation_results(df)
    
    return df, results


if __name__ == "__main__":
    df, results = main()