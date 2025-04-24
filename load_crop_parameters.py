import pandas as pd
import glob


def load_crop_parameters(crop_name):
    # All crop cases must be listed here with the same name as in the CSV file
    crop_column_map = {
        'Model Default (Potato)': 'Model Default (Potato)',
        'Potato': 'Potato',
        'Wheat': 'Wheat',
        'Barley': 'Barley',
        'Pear': 'Pear',
        'Raspberry': 'Raspberry',
        'Blueberry (Open field)': 'Blueberry (Open field)',
        'Blueberry (Tunnel)': 'Blueberry (Tunnel)',
        'Sugar beet': 'Sugar beet'
    }
    
    if crop_name not in crop_column_map:
        print(f"Warning: Crop '{crop_name}' not found. Falling back to 'Model Default (Potato)'.")
        crop_name = 'Model Default (Potato)'
    
    # Read the CSV file with the correct encoding
    try:
        # Find all matching CSV files in the current directory
        matching_files = glob.glob("*cases*.csv")
        
        # Check if there are multiple matching files
        if len(matching_files) > 1:
            raise ValueError(f"Multiple files match the pattern 'cases*.csv': {matching_files}. Please ensure only one file matches.")
        elif len(matching_files) == 0:
            raise FileNotFoundError("No file matching the pattern 'cases*.csv' was found.")
        
        # Load the first (and only) matching file
        file_path = matching_files[0]
        df = pd.read_csv(file_path, encoding='ISO-8859-1')
    except UnicodeDecodeError:
        raise ValueError("Failed to decode the CSV file. Please check the file encoding.")

    # Create a dictionary to store parameter values
    params = {}
    
    # Lookup column for the selected crop
    crop_col = crop_column_map[crop_name]
    
    # Strip whitespace from parameter names
    df['Parameter'] = df['Parameter'].str.strip()
    
    # Extract parameters for the selected crop
    for _, row in df.iterrows():
        param_name = row['Parameter']
        if pd.isna(param_name) or param_name == '':
            continue
            
        # Get the value for this crop, or use default if not available
        value = row[crop_col]
        reason = None
        if pd.isna(value) or value == '':
            value = row['Model Default (Potato)']
            reason = "missing"
        try:
            # Convert to float if possible
            value = float(value)
        except (ValueError, TypeError):
            reason = "not a number"
        
        # Store the parameter and log warnings if necessary
        if reason:
            print(f"Warning: Using default value for '{param_name}' ({value}) because it was {reason}.")
        params[param_name] = value
    
    return params


def apply_crop_parameters(params):
    # All parameters must be listed here with the same name as in the CSV file
    param_map = {
        'start_season': 'start_season',
        'days': 'days',
        'rue': 'rue',
        'k': 'k',
        'sla': 'sla',
        'alloc_leaf': 'alloc_leaf',
        'alloc_yield': 'alloc_yield',
        'max_lai': 'max_lai',
        'rue_shaded': 'rue_shaded',
        'sla_shaded': 'sla_shaded',
        'alloc_leaf_shaded': 'alloc_leaf_shaded',
        'alloc_yield_shaded': 'alloc_yield_shaded',
        'dmc_shaded': 'dmc_shaded',
        'dmc': 'dmc',
        'Light saturation': 'light_saturation',
    }
    
    # Define hardcoded limits for each parameter
    limits = {
        'start_season': (1, 365),
        'days': (1, 365),
        'rue': (0.0, 15),
        'k': (0.0, 1.0),
        'sla': (0.0, 0.5),
        'alloc_leaf': (0.0, 1.0),
        'alloc_yield': (0.0, 1.0),
        'max_lai': (0.0, 10.0),
        'rue_shaded': (0.0, 15),
        'sla_shaded': (0.0, 0.5),
        'alloc_leaf_shaded': (0.0, 1.0),
        'alloc_yield_shaded': (0.0, 1.0),
        'dmc_shaded': (0.0, 1.0),
        'dmc': (0.0, 1.0),
        'light_saturation': (0.0, 10.0)
    }

    # Create a dictionary of variable assignments
    assignments = {}
    for param_name, var_name in param_map.items():
        if param_name in params:
            value = params[param_name]
            # Validate the value against its range
            if var_name in limits:
                min_val, max_val = limits[var_name]
                if not isinstance(value, (int, float)) or not (min_val <= value <= max_val):
                    print(f"Warning: Parameter '{param_name}' value ({value}) is out of range ({min_val}, {max_val}). Using default.")
                    continue
            assignments[var_name] = value

    return assignments

def get_param_df(crop_name):
    try:
        params = load_crop_parameters(crop_name)        # Load parameters for the selected crop
        assignments = apply_crop_parameters(params)        # Apply and validate parameters
        param_df = pd.DataFrame.from_dict(assignments, orient='index', columns=['Value'])        # Create a DataFrame to return the parameters

        print (f"\n\nParameters for {crop_name}:")
        print (param_df)
        
        return param_df
    
    except Exception as e:
        print(f"Error: {e}")
        return None


# Test environment
if __name__ == "__main__":
    crop_name = 'Model Default (Potato)'  # Change this to the desired crop name
    get_param_df(crop_name)