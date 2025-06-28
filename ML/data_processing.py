import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
import statsmodels.api as sm
import numpy as np
import os
import glob

def load_data():
    """
    Automatically load CSV files from the current directory.
    Expects files with patterns like: *svc*.csv, *collision*.csv, *school*.csv, *tmc*.csv
    Or loads files based on common naming patterns.
    """
    # Get all CSV files in current directory
    csv_files = glob.glob("*.csv")
    
    if not csv_files:
        raise FileNotFoundError("No CSV files found in current directory")
    
    print(f"Found CSV files: {csv_files}")
    
    # Try to identify files based on common patterns
    svc_file = None
    coll_file = None
    tmc_file = None
    
    # Look for files with specific patterns
    for file in csv_files:
        file_lower = file.lower()
        if 'svc' in file_lower or 'speed' in file_lower or 'volume' in file_lower:
            svc_file = file
        elif 'collision' in file_lower or 'crash' in file_lower or 'accident' in file_lower:
            coll_file = file
        elif 'tmc' in file_lower or 'traffic' in file_lower or 'intersection' in file_lower:
            tmc_file = file
    
    # If pattern matching fails, ask user to rename files or use first few files
    if not all([svc_file, coll_file, tmc_file]):
        print("Could not automatically identify all files. Attempting to match based on available files:")
        # More specific matching for your files
        for file in csv_files:
            file_lower = file.lower()
            if file_lower == 'svc.csv':
                svc_file = file
            elif file_lower == 'collisions.csv':
                coll_file = file
            elif file_lower == 'intersections.csv':
                tmc_file = file
        
        # Final fallback
        if not all([svc_file, coll_file, tmc_file]):
            if len(csv_files) >= 3:
                # Filter out schools.csv for TMC assignment
                non_school_files = [f for f in csv_files if 'school' not in f.lower()]
                if len(non_school_files) >= 3:
                    svc_file = svc_file or non_school_files[0]
                    coll_file = coll_file or non_school_files[1] 
                    tmc_file = tmc_file or non_school_files[2]
                else:
                    print("Not enough non-school CSV files found. Please ensure you have at least 3 CSV files.")
                    print("Expected: SVC counts, Collisions, and TMC/Intersection counts")
                    raise FileNotFoundError("Insufficient CSV files")
            else:
                print("Not enough CSV files found. Please ensure you have at least 3 CSV files.")
                print("Expected: SVC counts, Collisions, and TMC/Intersection counts")
                raise FileNotFoundError("Insufficient CSV files")
    
    print(f"Loading SVC data from: {svc_file}")
    print(f"Loading collision data from: {coll_file}")
    print(f"Loading TMC data from: {tmc_file}")
    
    # Load CSVs
    svc = pd.read_csv(svc_file)
    coll = pd.read_csv(coll_file, parse_dates=['OCC_DATE'] if 'OCC_DATE' in pd.read_csv(coll_file, nrows=1).columns else None)
    tmc = pd.read_csv(tmc_file)
    
    # Look for school data (could be CSV, GeoJSON, or shapefile)
    school_files = glob.glob("*school*.*") + glob.glob("*School*.*")
    schools = None
    
    if school_files:
        school_file = school_files[0]
        print(f"Loading school data from: {school_file}")
        try:
            schools = gpd.read_file(school_file)
        except:
            # If it's a CSV, try to load as regular pandas DataFrame
            schools = pd.read_csv(school_file)
    else:
        print("No school data file found. Creating empty GeoDataFrame.")
        schools = gpd.GeoDataFrame()
    
    return svc, coll, schools, tmc

def to_gdf(df, lon_col=None, lat_col=None, crs="EPSG:4326"):
    """
    Convert DataFrame to GeoDataFrame, automatically detecting longitude/latitude columns if not specified.
    """
    # If columns not specified, try to detect them
    if lon_col is None or lat_col is None:
        
        # Common longitude column patterns (more specific patterns first)
        lon_patterns = ['longitude', 'long_wgs84', 'lng', 'long', 'lon', 'x_coord', 'x']
        lat_patterns = ['latitude', 'lat_wgs84', 'lat', 'y_coord', 'y']
        
        detected_lon = None
        detected_lat = None
        
        # Look for exact matches first
        for col in df.columns:
            col_lower = col.lower()
            if col_lower in [p.lower() for p in lon_patterns]:
                detected_lon = col
                break
        
        for col in df.columns:
            col_lower = col.lower()
            if col_lower in [p.lower() for p in lat_patterns]:
                detected_lat = col
                break
        
        # If no exact match, look for partial matches
        if not detected_lon:
            for col in df.columns:
                col_lower = col.lower()
                for pattern in lon_patterns:
                    if pattern in col_lower and not any(exclude in col_lower for exclude in ['heavy', 'pct', 'percent']):
                        detected_lon = col
                        break
                if detected_lon:
                    break
        
        if not detected_lat:
            for col in df.columns:
                col_lower = col.lower()
                for pattern in lat_patterns:
                    if pattern in col_lower and not any(exclude in col_lower for exclude in ['heavy', 'pct', 'percent']):
                        detected_lat = col
                        break
                if detected_lat:
                    break
        
        if detected_lon and detected_lat:
            lon_col = detected_lon
            lat_col = detected_lat
            print(f"Auto-detected coordinates: {lon_col}, {lat_col}")
        else:
            # Show available columns to help debug
            print(f"Available columns in dataframe: {list(df.columns)}")
            
            # Check if there's already a geometry column
            if 'geometry' in df.columns:
                print("Found existing geometry column, converting to GeoDataFrame")
                return gpd.GeoDataFrame(df, crs=crs)
            
            raise ValueError(f"Could not detect longitude/latitude columns. Available columns: {list(df.columns)}")
    
    # Check if specified columns exist
    if lon_col not in df.columns or lat_col not in df.columns:
        print(f"Available columns: {list(df.columns)}")
        raise KeyError(f"Columns '{lon_col}' or '{lat_col}' not found in dataframe")
    
    # Handle cases where coordinate data might be in JSON format
    if df[lon_col].dtype == 'object':
        # Check if it's JSON geometry data
        sample_val = str(df[lon_col].iloc[0])
        if 'coordinates' in sample_val and 'type' in sample_val:
            print(f"Detected JSON geometry in {lon_col}, attempting to parse...")
            try:
                import json
                geometries = []
                for idx, row in df.iterrows():
                    try:
                        geom_data = json.loads(row[lon_col])
                        if geom_data['type'] == 'MultiPoint':
                            # Take the first point from MultiPoint
                            coords = geom_data['coordinates'][0]
                            geometries.append(Point(coords[0], coords[1]))
                        elif geom_data['type'] == 'Point':
                            coords = geom_data['coordinates']
                            geometries.append(Point(coords[0], coords[1]))
                        else:
                            geometries.append(None)
                    except:
                        geometries.append(None)
                
                gdf = gpd.GeoDataFrame(df, geometry=geometries, crs=crs)
                gdf = gdf.dropna(subset=['geometry'])
                return gdf
            except Exception as e:
                print(f"Failed to parse JSON geometry: {e}")
                raise
    
    # Standard coordinate processing
    df = df.dropna(subset=[lon_col, lat_col])
    
    # Convert to numeric if they're strings
    df[lon_col] = pd.to_numeric(df[lon_col], errors='coerce')
    df[lat_col] = pd.to_numeric(df[lat_col], errors='coerce')
    
    # Drop rows where conversion failed
    df = df.dropna(subset=[lon_col, lat_col])
    
    gdf = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df[lon_col], df[lat_col]), crs=crs)
    return gdf

def preprocess(svc, coll, schools, tmc, buffer_m=100, school_buf_m=200):
    print("Converting data to GeoDataFrames...")
    
    # Convert to GeoDataFrames with automatic column detection
    try:
        svc_gdf = to_gdf(svc).to_crs(epsg=3857)
        print("✓ SVC data converted to GeoDataFrame")
    except Exception as e:
        print(f"Error converting SVC data: {e}")
        raise
    
    try:
        tmc_gdf = to_gdf(tmc).to_crs(epsg=3857)
        print("✓ TMC/Intersection data converted to GeoDataFrame")
    except Exception as e:
        print(f"Error converting TMC data: {e}")
        raise
    
    try:
        coll_gdf = to_gdf(coll).to_crs(epsg=3857)
        print("✓ Collision data converted to GeoDataFrame")
    except Exception as e:
        print(f"Error converting collision data: {e}")
        raise
    
    # Handle schools data
    if not schools.empty:
        try:
            if 'geometry' not in schools.columns:
                schools_gdf = to_gdf(schools).to_crs(epsg=3857)
            else:
                schools_gdf = gpd.GeoDataFrame(schools).to_crs(epsg=3857)
            print("✓ School data converted to GeoDataFrame")
        except Exception as e:
            print(f"Warning: Could not process school data: {e}")
            schools_gdf = gpd.GeoDataFrame()
    else:
        schools_gdf = gpd.GeoDataFrame()
        print("✓ No school data provided")

    # Buffer sites
    svc_gdf['buffer'] = svc_gdf.geometry.buffer(buffer_m)
    sites = svc_gdf.set_geometry('buffer')

    # Collision counts per site
    joined = gpd.sjoin(coll_gdf, sites, how='inner', predicate='within')
    
    # Check if latest_count_id exists, if not use index or create one
    if 'latest_count_id' not in svc_gdf.columns:
        svc_gdf.reset_index(inplace=True)
        svc_gdf['latest_count_id'] = svc_gdf.index
        print("Warning: 'latest_count_id' not found, using index as ID")
    
    # Handle potential duplicate index issues
    if 'latest_count_id' in joined.columns:
        coll_counts = joined.groupby('latest_count_id').size().rename('collision_count')
        svc_gdf = svc_gdf.set_index('latest_count_id').join(coll_counts).fillna({'collision_count': 0})
        svc_gdf.reset_index(inplace=True)
    else:
        print("Warning: Could not join collision counts, setting all to 0")
        svc_gdf['collision_count'] = 0

    # Exposure veh-km
    # Check for volume column with flexible naming
    vol_cols = ['avg_daily_vol', 'daily_volume', 'volume', 'aadt', 'traffic_volume']
    vol_col = None
    for col in vol_cols:
        if col in svc_gdf.columns:
            vol_col = col
            break
    
    if vol_col is None:
        print(f"Warning: No volume column found. Available columns: {list(svc_gdf.columns)}")
        print("Using default volume of 1000 vehicles/day")
        svc_gdf['avg_daily_vol'] = 1000
        vol_col = 'avg_daily_vol'
    
    # Requires a segment length column in km; assume 'length_km' exists or default to 0.1 km
    svc_gdf['length_km'] = svc_gdf.get('length_km', 0.1)
    svc_gdf['veh_km'] = svc_gdf[vol_col] * 365 * svc_gdf['length_km']

    # School proximity flag
    if not schools_gdf.empty:
        schools_buf = schools_gdf.buffer(school_buf_m)
        schools_union = gpd.GeoSeries(schools_buf.unary_union)
        svc_gdf['near_school'] = svc_gdf.geometry.centroid.within(schools_union)
    else:
        svc_gdf['near_school'] = False

    # Merge TMC totals by spatial join within 50m
    tmc_gdf['tmc_geom_buff'] = tmc_gdf.geometry.buffer(50)
    tmc_sites = tmc_gdf.set_geometry('tmc_geom_buff')
    
    # Create default TMC columns if they don't exist
    tmc_defaults = {
        'total_vehicle': 0,
        'total_pedestrian': 0,
        'total_bike': 0
    }
    
    for col, default_val in tmc_defaults.items():
        if col not in tmc_gdf.columns:
            tmc_gdf[col] = default_val
            print(f"Warning: '{col}' not found in TMC data, using default value {default_val}")
    
    # Prepare columns for spatial join
    join_cols = ['total_vehicle', 'total_pedestrian', 'total_bike', 'tmc_geom_buff']
    available_cols = [col for col in join_cols if col in tmc_sites.columns]
    
    if available_cols:
        try:
            merged = gpd.sjoin(svc_gdf, tmc_sites[available_cols], how='left', predicate='intersects')
            
            # Handle duplicate matches by taking the first match
            if len(merged) > len(svc_gdf):
                print("Warning: Multiple TMC matches found, taking first match for each site")
                merged = merged.groupby(merged.index).first()
            
            svc_gdf['total_vehicle'] = merged['total_vehicle'].fillna(0)
            svc_gdf['total_pedestrian'] = merged['total_pedestrian'].fillna(0)
            svc_gdf['total_bike'] = merged['total_bike'].fillna(0)
        except Exception as e:
            print(f"Warning: Could not join TMC data: {e}")
            svc_gdf['total_vehicle'] = 0
            svc_gdf['total_pedestrian'] = 0
            svc_gdf['total_bike'] = 0
    else:
        print("Warning: No TMC columns available for joining")
        svc_gdf['total_vehicle'] = 0
        svc_gdf['total_pedestrian'] = 0
        svc_gdf['total_bike'] = 0

    # Prepare final DataFrame with flexible column selection
    required_cols = ['latest_count_id', 'collision_count', 'veh_km', 'near_school', 'total_vehicle', 'total_pedestrian', 'total_bike']
    
    # Add available traffic columns
    traffic_cols = ['avg_daily_vol', 'avg_speed', 'avg_85th_percentile_speed', 'avg_95th_percentile_speed', 'avg_heavy_pct']
    available_traffic_cols = [col for col in traffic_cols if col in svc_gdf.columns]
    
    if not available_traffic_cols:
        print("Warning: No traffic columns found, using volume column")
        available_traffic_cols = [vol_col]
    
    final_cols = required_cols + available_traffic_cols
    available_final_cols = [col for col in final_cols if col in svc_gdf.columns]
    
    print(f"Final feature columns: {available_final_cols}")
    features = svc_gdf[available_final_cols].copy()
    return features

def fit_poisson(features):
    # Prepare design matrix with available columns
    y = features['collision_count']
    
    # Define preferred features in order of preference
    preferred_features = ['avg_daily_vol', 'avg_speed', 'avg_85th_percentile_speed',
                         'avg_95th_percentile_speed', 'avg_heavy_pct',
                         'near_school', 'total_vehicle', 'total_pedestrian', 'total_bike']
    
    # Select available features
    available_features = [col for col in preferred_features if col in features.columns]
    
    if not available_features:
        print("Warning: No standard features found, using all numeric columns except ID and target")
        # Fall back to any numeric columns
        numeric_cols = features.select_dtypes(include=[np.number]).columns
        available_features = [col for col in numeric_cols 
                            if col not in ['latest_count_id', 'collision_count', 'veh_km']]
    
    print(f"Using features for modeling: {available_features}")
    
    if not available_features:
        raise ValueError("No suitable features found for modeling")
    
    X = features[available_features].copy()
    
    # Ensure all features are numeric
    for col in X.columns:
        if X[col].dtype == 'object':
            print(f"Converting {col} to numeric...")
            X[col] = pd.to_numeric(X[col], errors='coerce')
        
        # Handle boolean columns
        if X[col].dtype == 'bool':
            X[col] = X[col].astype(int)
    
    # Fill any NaN values that might have been created during conversion
    X = X.fillna(0)
    
    # Check for any remaining non-numeric data
    non_numeric_cols = X.select_dtypes(exclude=[np.number]).columns
    if len(non_numeric_cols) > 0:
        print(f"Warning: Dropping non-numeric columns: {list(non_numeric_cols)}")
        X = X.select_dtypes(include=[np.number])
    
    if X.empty:
        raise ValueError("No numeric features available for modeling")
    
    print(f"Final feature matrix shape: {X.shape}")
    print(f"Feature data types: {X.dtypes.to_dict()}")
    
    X = sm.add_constant(X)
    offset = np.log(features['veh_km'] + 1e-6)
    
    print("Fitting Poisson GLM...")
    try:
        model = sm.GLM(y, X, family=sm.families.Poisson(), offset=offset).fit()
        features['lambda_hat'] = model.predict(X, offset=offset)
        features['predicted_risk'] = features['lambda_hat'] / features['veh_km']
        
        print("Model fitted successfully!")
        print(f"Model summary:")
        print(f"AIC: {model.aic:.2f}")
        print(f"Deviance: {model.deviance:.2f}")
        
        return model, features
    except Exception as e:
        print(f"Error fitting model: {e}")
        print("Attempting simplified model with fewer features...")
        
        # Try with just the most important features
        simple_features = [col for col in ['avg_daily_vol', 'near_school'] if col in X.columns]
        if simple_features:
            X_simple = sm.add_constant(X[simple_features])
            model = sm.GLM(y, X_simple, family=sm.families.Poisson(), offset=offset).fit()
            features['lambda_hat'] = model.predict(X_simple, offset=offset)
            features['predicted_risk'] = features['lambda_hat'] / features['veh_km']
            print("Simplified model fitted successfully!")
            return model, features
        else:
            raise

def main():
    """
    Main function to process traffic and collision data for speed-camera siting.
    Automatically loads CSV files from the current directory.
    """
    print("SpeedShield Data Processing")
    print("=" * 30)
    
    try:
        # Load data automatically from current directory
        svc, coll, schools, tmc = load_data()
        
        print("\nProcessing data...")
        features = preprocess(svc, coll, schools, tmc)
        
        print("Fitting Poisson regression model...")
        model, results = fit_poisson(features)
        
        # Save results
        output_file = "sited_scores.csv"
        results.sort_values('predicted_risk', ascending=False).to_csv(output_file, index=False)
        
        print(f"\nResults saved to {output_file}")
        print(f"Processed {len(results)} sites")
        print(f"Top 5 highest risk sites:")
        print(results.nlargest(5, 'predicted_risk')[['latest_count_id', 'predicted_risk', 'collision_count']].to_string())
        
    except Exception as e:
        print(f"Error: {e}")
        print("\nPlease ensure your CSV files are in the current directory with appropriate names:")
        print("- SVC/Speed/Volume data CSV")
        print("- Collision/Crash/Accident data CSV") 
        print("- TMC/Traffic data CSV")
        print("- School data file (optional)")
        return 1
    
    return 0

if __name__ == "__main__":
    main()
