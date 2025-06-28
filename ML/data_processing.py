import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
import statsmodels.api as sm
import numpy as np
import os
import glob
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

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

    # Extract coordinates from geometry for output
    svc_gdf['longitude'] = svc_gdf.geometry.to_crs(epsg=4326).x
    svc_gdf['latitude'] = svc_gdf.geometry.to_crs(epsg=4326).y
    
    # Prepare final DataFrame with flexible column selection
    # Start with coordinates as first columns
    coordinate_cols = ['longitude', 'latitude']
    required_cols = ['latest_count_id', 'collision_count', 'veh_km', 'near_school', 'total_vehicle', 'total_pedestrian', 'total_bike']
    
    # Add available traffic columns
    traffic_cols = ['avg_daily_vol', 'avg_speed', 'avg_85th_percentile_speed', 'avg_95th_percentile_speed', 'avg_heavy_pct']
    available_traffic_cols = [col for col in traffic_cols if col in svc_gdf.columns]
    
    if not available_traffic_cols:
        print("Warning: No traffic columns found, using volume column")
        available_traffic_cols = [vol_col]
    
    # Combine all columns with coordinates first
    final_cols = coordinate_cols + required_cols + available_traffic_cols
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

def add_percentiles_and_visualize(results):
    """
    Add percentile scores to the results and create visualizations.
    """
    print("Calculating percentiles and creating visualizations...")
    
    # Calculate percentiles for predicted_risk
    results['risk_percentile'] = stats.rankdata(results['predicted_risk'], method='average') / len(results) * 100
    
    # Create risk categories based on percentiles
    results['risk_category'] = pd.cut(results['risk_percentile'], 
                                    bins=[0, 50, 75, 90, 95, 100],
                                    labels=['Low', 'Medium', 'High', 'Very High', 'Critical'],
                                    include_lowest=True)
    
    # Set up the plotting style
    plt.style.use('seaborn-v0_8')
    sns.set_palette("viridis")
    
    # Create a figure with multiple subplots
    fig = plt.figure(figsize=(20, 16))
    
    # 1. Risk Distribution Histogram
    plt.subplot(3, 3, 1)
    plt.hist(results['predicted_risk'], bins=50, alpha=0.7, color='skyblue', edgecolor='black')
    plt.xlabel('Predicted Risk Score')
    plt.ylabel('Frequency')
    plt.title('Distribution of Risk Scores')
    plt.grid(True, alpha=0.3)
    
    # 2. Log-scale Risk Distribution
    plt.subplot(3, 3, 2)
    plt.hist(np.log10(results['predicted_risk'] + 1e-10), bins=50, alpha=0.7, color='lightcoral', edgecolor='black')
    plt.xlabel('Log10(Predicted Risk Score)')
    plt.ylabel('Frequency')
    plt.title('Log-Scale Risk Distribution')
    plt.grid(True, alpha=0.3)
    
    # 3. Risk vs Collision Count Scatter
    plt.subplot(3, 3, 3)
    plt.scatter(results['collision_count'], results['predicted_risk'], alpha=0.6, s=30)
    plt.xlabel('Actual Collision Count')
    plt.ylabel('Predicted Risk Score')
    plt.title('Risk Score vs Actual Collisions')
    plt.grid(True, alpha=0.3)
    
    # 4. Risk Categories Bar Chart
    plt.subplot(3, 3, 4)
    risk_counts = results['risk_category'].value_counts()
    colors = ['green', 'yellow', 'orange', 'red', 'darkred']
    risk_counts.plot(kind='bar', color=colors[:len(risk_counts)], alpha=0.7)
    plt.xlabel('Risk Category')
    plt.ylabel('Number of Sites')
    plt.title('Sites by Risk Category')
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3)
    
    # 5. Percentile vs Risk Score
    plt.subplot(3, 3, 5)
    plt.scatter(results['risk_percentile'], results['predicted_risk'], alpha=0.6, s=30, c=results['collision_count'], cmap='Reds')
    plt.xlabel('Risk Percentile')
    plt.ylabel('Predicted Risk Score')
    plt.title('Risk Percentile vs Risk Score')
    plt.colorbar(label='Collision Count')
    plt.grid(True, alpha=0.3)
    
    # 6. Geographic Distribution (if coordinates available)
    plt.subplot(3, 3, 6)
    if 'longitude' in results.columns and 'latitude' in results.columns:
        scatter = plt.scatter(results['longitude'], results['latitude'], 
                            c=results['predicted_risk'], cmap='Reds', 
                            s=50, alpha=0.7)
        plt.xlabel('Longitude')
        plt.ylabel('Latitude')
        plt.title('Geographic Distribution of Risk')
        plt.colorbar(scatter, label='Risk Score')
    else:
        plt.text(0.5, 0.5, 'Geographic coordinates\nnot available', 
                ha='center', va='center', transform=plt.gca().transAxes)
        plt.title('Geographic Distribution of Risk')
    
    # 7. Top 10 Highest Risk Sites
    plt.subplot(3, 3, 7)
    top_10 = results.nlargest(10, 'predicted_risk')
    plt.barh(range(len(top_10)), top_10['predicted_risk'])
    plt.ylabel('Site Rank')
    plt.xlabel('Risk Score')
    plt.title('Top 10 Highest Risk Sites')
    plt.yticks(range(len(top_10)), [f"Site {i+1}" for i in range(len(top_10))])
    plt.grid(True, alpha=0.3)
    
    # 8. Risk vs Traffic Volume
    plt.subplot(3, 3, 8)
    if 'avg_daily_vol' in results.columns:
        plt.scatter(results['avg_daily_vol'], results['predicted_risk'], alpha=0.6, s=30)
        plt.xlabel('Average Daily Volume')
        plt.ylabel('Predicted Risk Score')
        plt.title('Risk vs Traffic Volume')
        plt.grid(True, alpha=0.3)
    else:
        plt.text(0.5, 0.5, 'Traffic volume data\nnot available', 
                ha='center', va='center', transform=plt.gca().transAxes)
        plt.title('Risk vs Traffic Volume')
    
    # 9. Risk Percentile Distribution
    plt.subplot(3, 3, 9)
    plt.hist(results['risk_percentile'], bins=20, alpha=0.7, color='purple', edgecolor='black')
    plt.xlabel('Risk Percentile')
    plt.ylabel('Frequency')
    plt.title('Risk Percentile Distribution')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save the comprehensive visualization
    viz_filename = 'risk_analysis_dashboard.png'
    plt.savefig(viz_filename, dpi=300, bbox_inches='tight')
    print(f"Comprehensive visualization saved as {viz_filename}")
    
    # Create a separate high-resolution map if coordinates are available
    if 'longitude' in results.columns and 'latitude' in results.columns:
        plt.figure(figsize=(12, 10))
        
        # Create risk-based color mapping
        norm = plt.Normalize(vmin=results['predicted_risk'].min(), vmax=results['predicted_risk'].max())
        scatter = plt.scatter(results['longitude'], results['latitude'], 
                            c=results['predicted_risk'], cmap='Reds', 
                            s=80, alpha=0.7, norm=norm, edgecolors='black', linewidth=0.5)
        
        plt.xlabel('Longitude', fontsize=12)
        plt.ylabel('Latitude', fontsize=12)
        plt.title('SpeedShield Risk Analysis - Geographic Distribution', fontsize=14, fontweight='bold')
        
        # Add colorbar
        cbar = plt.colorbar(scatter, label='Predicted Risk Score')
        cbar.ax.tick_params(labelsize=10)
        
        # Add grid
        plt.grid(True, alpha=0.3)
        
        # Highlight top 5% highest risk sites
        top_5_percent = results[results['risk_percentile'] >= 95]
        if not top_5_percent.empty:
            plt.scatter(top_5_percent['longitude'], top_5_percent['latitude'], 
                       s=150, facecolors='none', edgecolors='blue', linewidth=2, 
                       label=f'Top 5% Risk Sites (n={len(top_5_percent)})')
            plt.legend()
        
        plt.tight_layout()
        map_filename = 'risk_geographic_map.png'
        plt.savefig(map_filename, dpi=300, bbox_inches='tight')
        print(f"Geographic risk map saved as {map_filename}")
    
    # Create summary statistics
    print("\n" + "="*50)
    print("RISK ANALYSIS SUMMARY")
    print("="*50)
    print(f"Total sites analyzed: {len(results):,}")
    print(f"Mean risk score: {results['predicted_risk'].mean():.6f}")
    print(f"Median risk score: {results['predicted_risk'].median():.6f}")
    print(f"Standard deviation: {results['predicted_risk'].std():.6f}")
    print(f"Min risk score: {results['predicted_risk'].min():.6f}")
    print(f"Max risk score: {results['predicted_risk'].max():.6f}")
    
    print(f"\nRisk Category Distribution:")
    for category in ['Critical', 'Very High', 'High', 'Medium', 'Low']:
        count = (results['risk_category'] == category).sum()
        percentage = count / len(results) * 100
        print(f"  {category}: {count:,} sites ({percentage:.1f}%)")
    
    print(f"\nTop 10 Percentile Thresholds:")
    percentiles = [90, 95, 97.5, 99, 99.5, 99.9]
    for p in percentiles:
        threshold = np.percentile(results['predicted_risk'], p)
        count = (results['predicted_risk'] >= threshold).sum()
        print(f"  {p}th percentile: {threshold:.6f} ({count:,} sites)")
    
    # Don't show plots interactively in batch processing
    # plt.show()
    
    return results

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
        
        # Add percentiles and create visualizations
        results = add_percentiles_and_visualize(results)
        
        # Save results with coordinates as first columns
        output_file = "sited_scores.csv"
        
        # Ensure coordinates are first columns in output, followed by percentiles
        output_cols = ['longitude', 'latitude']
        priority_cols = ['latest_count_id', 'predicted_risk', 'risk_percentile', 'risk_category', 'collision_count']
        remaining_cols = [col for col in results.columns if col not in output_cols + priority_cols]
        final_output_cols = output_cols + priority_cols + remaining_cols
        
        # Reorder columns and save
        results_ordered = results[final_output_cols]
        results_ordered.sort_values('predicted_risk', ascending=False).to_csv(output_file, index=False)
        
        print(f"\nResults saved to {output_file}")
        print(f"Processed {len(results)} sites")
        print(f"Output columns: {final_output_cols[:8]}...")  # Show first 8 columns
        print(f"Top 5 highest risk sites:")
        display_cols = ['longitude', 'latitude', 'latest_count_id', 'predicted_risk', 'risk_percentile', 'collision_count']
        print(results.nlargest(5, 'predicted_risk')[display_cols].to_string(index=False))
        
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
