import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import os
import json
from scipy.spatial.distance import cdist

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold

def load_training_data(filepath="training_data.csv"):
    """
    Load the prepared training data from the data processing pipeline.
    
    Parameters:
    filepath (str): Path to the training data CSV file
    
    Returns:
    pd.DataFrame: Loaded training data
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Training data file not found: {filepath}")
    
    print(f"Loading training data from {filepath}...")
    data = pd.read_csv(filepath)
    
    print(f"Loaded {len(data)} records with {len(data.columns)} features")
    print(f"Columns: {list(data.columns)}")
    
    return data

def load_camera_coordinates(filepath="cameras.csv"):
    """
    Load camera coordinates from CSV file and parse GeoJSON geometry.
    
    Parameters:
    filepath (str): Path to the cameras CSV file
    
    Returns:
    np.array: Array of camera coordinates [longitude, latitude]
    """
    cameras = pd.read_csv(filepath)
    
    camera_coords = []
    for _, row in cameras.iterrows():
        geom = json.loads(row['geometry'])
        coords = geom['coordinates'][0]  # First point in MultiPoint
        camera_coords.append([coords[0], coords[1]])  # [lon, lat]
    
    return np.array(camera_coords)

def fit_model(features):
    """
    Use existing camera locations to train a logistic regression model
    """

    # CAMERA STUFF
    camera_coords = load_camera_coordinates()
    
    site_coords = features[['longitude', 'latitude']].values
    
    distance_threshold = 0.0009  # approximately 100 meters

    # Calculate distances between all sites and cameras
    distances = cdist(site_coords, camera_coords, metric='euclidean')
    
    # Mark sites as having cameras if within threshold distance
    features['has_camera'] = (distances.min(axis=1) < distance_threshold).astype(int)

    # ACTUAL MODEL
    # load given features and create engineered features
    X = pd.DataFrame(index=features.index)
    
    # Speed risk calculation with null handling
    if 'avg_85th_percentile_speed' in features and 'avg_95th_percentile_speed' in features:
        speed_85th = pd.to_numeric(features['avg_85th_percentile_speed'], errors='coerce')
        speed_95th = pd.to_numeric(features['avg_95th_percentile_speed'], errors='coerce')
        if speed_85th.notna().any() and speed_95th.notna().any():
            # Calculate speed risk using available data, fill NaN with median
            speed_risk_vals = 0.5 * speed_85th + 0.5 * speed_95th
            median_speed_risk = speed_risk_vals.median()
            X['speed_risk'] = speed_risk_vals.fillna(median_speed_risk)
            features['speed_risk'] = X['speed_risk']
    elif 'avg_speed' in features:
        avg_speed = pd.to_numeric(features['avg_speed'], errors='coerce')
        if avg_speed.notna().any():
            median_speed = avg_speed.median()
            X['speed_risk'] = avg_speed.fillna(median_speed)
            features['speed_risk'] = X['speed_risk']
    
    # Volume risk calculation
    if 'avg_daily_vol' in features:
        volume = pd.to_numeric(features['avg_daily_vol'], errors='coerce')
        if volume.notna().any():
            # Use log1p for volume risk, fill NaN with median volume first
            median_volume = volume.median()
            volume_filled = volume.fillna(median_volume)
            X['volume_risk'] = np.log1p(volume_filled)
            features['volume_risk'] = X['volume_risk']
    
    # Collision history (always present)
    X['collision_history'] = np.log1p(features['collision_count'])
    features['collision_history'] = X['collision_history']
    
    # Heavy vehicle percentage with null handling
    if 'avg_heavy_pct' in features:
        heavy_pct = pd.to_numeric(features['avg_heavy_pct'], errors='coerce')
        if heavy_pct.notna().any():
            # For heavy percentage, fill NaN with 0 (assuming no heavy vehicles if not reported)
            X['heavy_share'] = heavy_pct.fillna(0)
            features['heavy_share'] = X['heavy_share']
    
    # School proximity
    if 'near_school' in features:
        X['near_school'] = features['near_school'].astype(int)
    
    # School zone and combined school risk
    if 'in_school_zone' in features:
        X['in_school_zone'] = features['in_school_zone'].astype(int)
        # create a combined school risk factor that considers both proximity and designation
        school_risk = features['near_school'].astype(int) + features['in_school_zone'].astype(int)
        X['school_risk_factor'] = np.minimum(school_risk, 2)  # cap at 2 for sites that are both near schools AND in zones
        features['school_risk_factor'] = X['school_risk_factor']
    
    # Retirement home proximity and vulnerable population risk
    if 'near_retirement_home' in features:
        X['near_retirement_home'] = features['near_retirement_home'].astype(int)
        # create a combined vulnerable population risk factor (schools + retirement homes)
        vulnerable_pop_risk = 0
        if 'near_school' in features:
            vulnerable_pop_risk += features['near_school'].astype(int)
        if 'near_retirement_home' in features:
            vulnerable_pop_risk += features['near_retirement_home'].astype(int) * 0.8  # weight retirement homes slightly less than schools
        X['vulnerable_population_risk'] = vulnerable_pop_risk
        features['vulnerable_population_risk'] = X['vulnerable_population_risk']
    
    # Enforcement gap calculation
    if 'dist_to_nearest_camera' in features and 'cameras_within_500m' in features:
        X['enforcement_gap'] = features['dist_to_nearest_camera'] / (1 + features['cameras_within_500m'])
        features['enforcement_gap'] = X['enforcement_gap']

    # Remove NaN values and prepare for model training
    X = X.dropna(axis=1, how='all')
    X = X.fillna(0)

    y = features['has_camera']

    # model training and cross validation
    model = LogisticRegression(max_iter=1000)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    cv_scores = []
    for train_idx, test_idx in skf.split(X, y):
        model.fit(X.iloc[train_idx], y.iloc[train_idx])
        score = model.score(X.iloc[test_idx], y.iloc[test_idx])
        cv_scores.append(score)
    
    print(f"Cross-validation accuracy: {np.mean(cv_scores):.3f} (+/- {np.std(cv_scores) * 2:.3f})")
    model.fit(X, y)

    # predict camera likliness score
    features['camera_score'] = model.predict_proba(X)[:, 1]

    if 'collision_count' in features and 'veh_km' in features:
        latent_risk = features['collision_count'] / features['veh_km']
        # gamma is adjustable to change the balance between latent risk and camera likliness
        gamma = 0.5
        features['final_score'] = gamma * features['camera_score'] + (1 - gamma) * latent_risk
    else:
        features['final_score'] = features['camera_score']
    
    features['predicted_risk'] = features['final_score']

    print(f"Model trained on features: {list(X.columns)}")

    return model, features

def add_percentiles_and_visualize(results):
    """
    Add percentile scores to the results and create visualizations.
    """
    print("Calculating percentiles and creating visualizations...")
    
    # Calculate percentiles for predicted_risk
    results['risk_percentile'] = stats.rankdata(results['predicted_risk'], method='average') / len(results) * 100
    
    # Calculate percentiles for all risk metrics
    risk_metrics = ['speed_risk', 'volume_risk', 'collision_history', 'heavy_share']
    for metric in risk_metrics:
        if metric in results.columns:
            results[f'{metric}_percentile'] = stats.rankdata(results[metric], method='average') / len(results) * 100
    
    # Create risk categories based on percentiles
    results['risk_category'] = pd.cut(results['risk_percentile'], 
                                    bins=[0, 50, 75, 90, 95, 100],
                                    labels=['Low', 'Medium', 'High', 'Very High', 'Critical'],
                                    include_lowest=True)
    
    # Create categories for other risk metrics
    for metric in risk_metrics:
        if metric in results.columns:
            results[f'{metric}_category'] = pd.cut(results[f'{metric}_percentile'], 
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
    
    # 8. Enhanced Feature Analysis (Speed vs Volume with Risk)
    plt.subplot(3, 3, 8)
    if 'avg_speed' in results.columns and 'avg_daily_vol' in results.columns:
        scatter = plt.scatter(results['avg_speed'], results['avg_daily_vol'], 
                            c=results['predicted_risk'], cmap='Reds', 
                            s=60, alpha=0.7, edgecolors='black', linewidth=0.5)
        plt.xlabel('Average Speed')
        plt.ylabel('Average Daily Volume')
        plt.title('Speed vs Volume (Risk Color-coded)')
        plt.colorbar(scatter, label='Risk Score')
        plt.grid(True, alpha=0.3)
    elif 'avg_daily_vol' in results.columns:
        plt.scatter(results['avg_daily_vol'], results['predicted_risk'], alpha=0.6, s=30)
        plt.xlabel('Average Daily Volume')
        plt.ylabel('Predicted Risk Score')
        plt.title('Risk vs Traffic Volume')
        plt.grid(True, alpha=0.3)
    else:
        plt.text(0.5, 0.5, 'Enhanced feature data\nnot available', 
                ha='center', va='center', transform=plt.gca().transAxes)
        plt.title('Enhanced Feature Analysis')
    
    # 9. Key Factor Emphasis - School Proximity Impact
    plt.subplot(3, 3, 9)
    if 'near_school' in results.columns:
        school_risk = results.groupby('near_school')['predicted_risk'].agg(['mean', 'std'])
        categories = ['Away from Schools', 'Near Schools']
        means = [school_risk.loc[False, 'mean'] if False in school_risk.index else 0,
                school_risk.loc[True, 'mean'] if True in school_risk.index else 0]
        stds = [school_risk.loc[False, 'std'] if False in school_risk.index else 0,
               school_risk.loc[True, 'std'] if True in school_risk.index else 0]
        
        plt.bar(categories, means, yerr=stds, alpha=0.7, color=['lightblue', 'red'], 
               capsize=5, edgecolor='black')
        plt.ylabel('Average Risk Score')
        plt.title('Risk Near vs Away from Schools')
        plt.grid(True, alpha=0.3)
        
        # Add sample size annotations
        for i, (cat, mean) in enumerate(zip(categories, means)):
            count = len(results[results['near_school'] == (i == 1)])
            plt.text(i, mean + stds[i] + 0.1, f'n={count}', ha='center', va='bottom')
    else:
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
    
    # Create risk metrics analysis
    fig2 = plt.figure(figsize=(20, 12))
    
    # Risk metrics correlation heatmap
    plt.subplot(2, 4, 1)
    metrics_for_corr = ['predicted_risk'] + [m for m in risk_metrics if m in results.columns]
    if len(metrics_for_corr) > 1:
        corr_matrix = results[metrics_for_corr].corr()
        sns.heatmap(corr_matrix, annot=True, cmap='RdYlBu_r', center=0, 
                   square=True, fmt='.2f', cbar_kws={'shrink': 0.8})
        plt.title('Risk Metrics Correlation Matrix')
    else:
        plt.text(0.5, 0.5, 'Insufficient metrics for correlation', 
                ha='center', va='center', transform=plt.gca().transAxes)
        plt.title('Risk Metrics Correlation Matrix')
    
    # Speed risk distribution
    plt.subplot(2, 4, 2)
    if 'speed_risk' in results.columns:
        plt.hist(results['speed_risk'], bins=30, alpha=0.7, color='orange', edgecolor='black')
        plt.xlabel('Speed Risk Score')
        plt.ylabel('Frequency')
        plt.title('Speed Risk Distribution')
        plt.grid(True, alpha=0.3)
    else:
        plt.text(0.5, 0.5, 'Speed risk data\nnot available', 
                ha='center', va='center', transform=plt.gca().transAxes)
        plt.title('Speed Risk Distribution')
    
    # Volume risk distribution
    plt.subplot(2, 4, 3)
    if 'volume_risk' in results.columns:
        plt.hist(results['volume_risk'], bins=30, alpha=0.7, color='blue', edgecolor='black')
        plt.xlabel('Volume Risk Score')
        plt.ylabel('Frequency')
        plt.title('Volume Risk Distribution')
        plt.grid(True, alpha=0.3)
    else:
        plt.text(0.5, 0.5, 'Volume risk data\nnot available', 
                ha='center', va='center', transform=plt.gca().transAxes)
        plt.title('Volume Risk Distribution')
    
    # Collision history distribution
    plt.subplot(2, 4, 4)
    if 'collision_history' in results.columns:
        plt.hist(results['collision_history'], bins=30, alpha=0.7, color='red', edgecolor='black')
        plt.xlabel('Collision History Score')
        plt.ylabel('Frequency')
        plt.title('Collision History Distribution')
        plt.grid(True, alpha=0.3)
    else:
        plt.text(0.5, 0.5, 'Collision history data\nnot available', 
                ha='center', va='center', transform=plt.gca().transAxes)
        plt.title('Collision History Distribution')
    
    # Speed vs Volume risk scatter
    plt.subplot(2, 4, 5)
    if 'speed_risk' in results.columns and 'volume_risk' in results.columns:
        scatter = plt.scatter(results['speed_risk'], results['volume_risk'], 
                            c=results['predicted_risk'], cmap='Reds', 
                            s=30, alpha=0.6, edgecolors='black', linewidth=0.3)
        plt.xlabel('Speed Risk')
        plt.ylabel('Volume Risk')
        plt.title('Speed vs Volume Risk')
        plt.colorbar(scatter, label='Overall Risk')
        plt.grid(True, alpha=0.3)
    else:
        plt.text(0.5, 0.5, 'Speed/Volume risk data\nnot available', 
                ha='center', va='center', transform=plt.gca().transAxes)
        plt.title('Speed vs Volume Risk')
    
    # Risk metrics categories comparison
    plt.subplot(2, 4, 6)
    category_counts = {}
    colors_map = {'Low': 'green', 'Medium': 'yellow', 'High': 'orange', 'Very High': 'red', 'Critical': 'darkred'}
    
    metrics_to_plot = [m for m in ['speed_risk', 'volume_risk', 'collision_history'] if f'{m}_category' in results.columns]
    if metrics_to_plot:
        x_pos = 0
        width = 0.8 / len(metrics_to_plot)
        
        for i, metric in enumerate(metrics_to_plot):
            cat_col = f'{metric}_category'
            counts = results[cat_col].value_counts()
            
            bottom = 0
            for category in ['Low', 'Medium', 'High', 'Very High', 'Critical']:
                if category in counts.index:
                    count = counts[category]
                    plt.bar(x_pos + i * width, count, width, bottom=bottom, 
                           color=colors_map[category], alpha=0.7, edgecolor='black', linewidth=0.5)
                    bottom += count
        
        plt.xlabel('Risk Metrics')
        plt.ylabel('Number of Sites')
        plt.title('Risk Categories by Metric')
        plt.xticks([i * width for i in range(len(metrics_to_plot))], 
                  [m.replace('_', ' ').title() for m in metrics_to_plot], rotation=45)
        plt.grid(True, alpha=0.3)
    else:
        plt.text(0.5, 0.5, 'Risk category data\nnot available', 
                ha='center', va='center', transform=plt.gca().transAxes)
        plt.title('Risk Categories by Metric')
    
    # Percentile comparison across metrics
    plt.subplot(2, 4, 7)
    percentile_cols = [f'{m}_percentile' for m in risk_metrics if f'{m}_percentile' in results.columns]
    if percentile_cols:
        data_for_box = [results[col] for col in percentile_cols]
        labels = [col.replace('_percentile', '').replace('_', ' ').title() for col in percentile_cols]
        
        bp = plt.boxplot(data_for_box, labels=labels, patch_artist=True)
        colors = ['orange', 'blue', 'red', 'purple']
        for patch, color in zip(bp['boxes'], colors[:len(bp['boxes'])]):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        
        plt.ylabel('Percentile')
        plt.title('Risk Metrics Percentile Distribution')
        plt.xticks(rotation=45)
        plt.grid(True, alpha=0.3)
    else:
        plt.text(0.5, 0.5, 'Percentile data\nnot available', 
                ha='center', va='center', transform=plt.gca().transAxes)
        plt.title('Risk Metrics Percentile Distribution')
    
    # Combined risk factors relationship
    plt.subplot(2, 4, 8)
    if 'collision_history' in results.columns and 'speed_risk' in results.columns:
        scatter = plt.scatter(results['collision_history'], results['speed_risk'], 
                            c=results['predicted_risk'], cmap='Reds', 
                            s=30, alpha=0.6, edgecolors='black', linewidth=0.3)
        plt.xlabel('Collision History')
        plt.ylabel('Speed Risk')
        plt.title('Collision History vs Speed Risk')
        plt.colorbar(scatter, label='Overall Risk')
        plt.grid(True, alpha=0.3)
    else:
        plt.text(0.5, 0.5, 'Combined risk data\nnot available', 
                ha='center', va='center', transform=plt.gca().transAxes)
        plt.title('Collision History vs Speed Risk')
    
    plt.tight_layout()
    
    # Save the risk metrics analysis
    metrics_viz_filename = 'risk_metrics_analysis.png'
    plt.savefig(metrics_viz_filename, dpi=300, bbox_inches='tight')
    print(f"Risk metrics analysis saved as {metrics_viz_filename}")
    
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
    
    # Print statistics for other risk metrics
    for metric in risk_metrics:
        if metric in results.columns:
            print(f"\n{metric.replace('_', ' ').title()} Statistics:")
            print(f"  Mean: {results[metric].mean():.6f}")
            print(f"  Median: {results[metric].median():.6f}")
            print(f"  Standard deviation: {results[metric].std():.6f}")
            
            if f'{metric}_category' in results.columns:
                print(f"  Category Distribution:")
                for category in ['Critical', 'Very High', 'High', 'Medium', 'Low']:
                    count = (results[f'{metric}_category'] == category).sum()
                    percentage = count / len(results) * 100
                    print(f"    {category}: {count:,} sites ({percentage:.1f}%)")
    
    print(f"\nTop 10 Percentile Thresholds:")
    percentiles = [90, 95, 97.5, 99, 99.5, 99.9]
    for p in percentiles:
        threshold = np.percentile(results['predicted_risk'], p)
        count = (results['predicted_risk'] >= threshold).sum()
        print(f"  {p}th percentile: {threshold:.6f} ({count:,} sites)")
    
    return results

def create_enhanced_model_analysis(results, model=None):
    """
    Create additional visualizations specifically for the enhanced model's key factors.
    """
    print("Creating enhanced model analysis visualizations...")
    
    # Create a comprehensive analysis figure
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Enhanced Model Analysis - Emphasis on Speed, Volume & Collisions', fontsize=16, fontweight='bold')
    
    # 1. Speed Factor Analysis
    ax1 = axes[0, 0]
    speed_cols = ['avg_speed', 'avg_85th_percentile_speed', 'avg_95th_percentile_speed']
    available_speed_cols = [col for col in speed_cols if col in results.columns]
    
    if available_speed_cols:
        # Create multi-speed comparison
        speed_data = []
        speed_labels = []
        for col in available_speed_cols:
            speed_data.append(results[col])
            speed_labels.append(col.replace('avg_', '').replace('_', ' ').title())
        
        bp = ax1.boxplot(speed_data, labels=speed_labels, patch_artist=True)
        colors = ['lightblue', 'orange', 'red']
        for patch, color in zip(bp['boxes'], colors[:len(bp['boxes'])]):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        ax1.set_title('Speed Distribution Analysis')
        ax1.set_ylabel('Speed (units)')
        ax1.grid(True, alpha=0.3)
        plt.setp(ax1.get_xticklabels(), rotation=45, ha='right')
    else:
        ax1.text(0.5, 0.5, 'Speed data not available', ha='center', va='center', transform=ax1.transAxes)
        ax1.set_title('Speed Distribution Analysis')
    
    # 2. Volume Factor Analysis
    ax2 = axes[0, 1]
    volume_cols = ['avg_daily_vol']
    available_volume_cols = [col for col in volume_cols if col in results.columns]
    
    if available_volume_cols:
        # Create correlation heatmap for volume factors
        volume_data = results[available_volume_cols + ['predicted_risk']].corr()
        im = ax2.imshow(volume_data, cmap='RdYlBu_r', aspect='auto')
        ax2.set_xticks(range(len(volume_data.columns)))
        ax2.set_yticks(range(len(volume_data.columns)))
        ax2.set_xticklabels([col.replace('_', '\n') for col in volume_data.columns], rotation=45, ha='right')
        ax2.set_yticklabels([col.replace('_', '\n') for col in volume_data.columns])
        ax2.set_title('Volume Factors Correlation with Risk')
        
        # Add correlation values
        for i in range(len(volume_data.columns)):
            for j in range(len(volume_data.columns)):
                text = ax2.text(j, i, f'{volume_data.iloc[i, j]:.2f}', 
                              ha='center', va='center', color='white' if abs(volume_data.iloc[i, j]) > 0.5 else 'black')
        
        plt.colorbar(im, ax=ax2, shrink=0.8)
    else:
        ax2.text(0.5, 0.5, 'Volume data not available', ha='center', va='center', transform=ax2.transAxes)
        ax2.set_title('Volume Factors Correlation with Risk')
    
    # 3. Risk Distribution by Key Factors
    ax3 = axes[0, 2]
    if 'collision_count' in results.columns:
        # Create risk distribution by collision history
        collision_bins = pd.cut(results['collision_count'], bins=5, include_lowest=True)
        risk_by_collision = results.groupby(collision_bins)['predicted_risk'].mean()
        
        bars = ax3.bar(range(len(risk_by_collision)), risk_by_collision.values, 
                      color='darkred', alpha=0.7, edgecolor='black')
        ax3.set_xlabel('Historical Collision Count (Binned)')
        ax3.set_ylabel('Average Predicted Risk')
        ax3.set_title('Risk by Historical Collision Count')
        ax3.set_xticks(range(len(risk_by_collision)))
        ax3.set_xticklabels([f'{interval.left:.0f}-{interval.right:.0f}' for interval in risk_by_collision.index], 
                           rotation=45, ha='right')
        ax3.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for bar, value in zip(bars, risk_by_collision.values):
            ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                    f'{value:.2f}', ha='center', va='bottom')
    else:
        ax3.text(0.5, 0.5, 'Collision data not available', ha='center', va='center', transform=ax3.transAxes)
        ax3.set_title('Risk by Historical Collision Count')
    
    # 4. Combined Risk Factors (Speed + Volume + School Proximity)
    ax4 = axes[1, 0]
    if all(col in results.columns for col in ['avg_speed', 'avg_daily_vol', 'near_school']):
        # Create 3D-like visualization using color and size
        school_sites = results[results['near_school'] == True]
        non_school_sites = results[results['near_school'] == False]
        
        if len(non_school_sites) > 0:
            scatter1 = ax4.scatter(non_school_sites['avg_speed'], non_school_sites['avg_daily_vol'], 
                                 c=non_school_sites['predicted_risk'], cmap='Blues', 
                                 s=50, alpha=0.6, label='Away from Schools', edgecolors='black', linewidth=0.5)
        if len(school_sites) > 0:
            scatter2 = ax4.scatter(school_sites['avg_speed'], school_sites['avg_daily_vol'], 
                                 c=school_sites['predicted_risk'], cmap='Reds', 
                                 s=80, alpha=0.8, label='Near Schools', marker='^', edgecolors='black', linewidth=0.5)
        
        ax4.set_xlabel('Average Speed')
        ax4.set_ylabel('Average Daily Volume')
        ax4.set_title('Combined Risk Factors Analysis')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        # Add colorbar
        if len(non_school_sites) > 0:
            plt.colorbar(scatter1, ax=ax4, label='Risk Score')
    else:
        ax4.text(0.5, 0.5, 'Combined factor data not available', ha='center', va='center', transform=ax4.transAxes)
        ax4.set_title('Combined Risk Factors Analysis')
    
    # 5. Model Performance Metrics
    ax5 = axes[1, 1]
    if 'collision_count' in results.columns:
        # Calculate model performance metrics
        actual = results['collision_count']
        predicted = results['lambda_hat'] if 'lambda_hat' in results.columns else results['predicted_risk']
        
        # Create residual plot
        residuals = actual - predicted
        ax5.scatter(predicted, residuals, alpha=0.6, s=30)
        ax5.axhline(y=0, color='red', linestyle='--', alpha=0.8)
        ax5.set_xlabel('Predicted Values')
        ax5.set_ylabel('Residuals')
        ax5.set_title('Model Residuals Analysis')
        ax5.grid(True, alpha=0.3)
        
        # Add performance metrics as text
        mae = np.mean(np.abs(residuals))
        rmse = np.sqrt(np.mean(residuals**2))
        ax5.text(0.05, 0.95, f'MAE: {mae:.3f}\nRMSE: {rmse:.3f}', 
                transform=ax5.transAxes, verticalalignment='top', 
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    else:
        ax5.text(0.5, 0.5, 'Performance metrics not available', ha='center', va='center', transform=ax5.transAxes)
        ax5.set_title('Model Performance Metrics')
    
    # 6. Feature Importance (if model coefficients available)
    ax6 = axes[1, 2]
    if model and hasattr(model, 'params'):
        # Extract and visualize key coefficients
        params = model.params.drop('const', errors='ignore')  # Remove intercept
        
        # Focus on enhanced features
        enhanced_features = {}
        for param_name, coef in params.items():
            if any(key in param_name.lower() for key in ['volume', 'speed', 'school', 'interaction', 'emphasis', 'risk']):
                enhanced_features[param_name] = coef
        
        if enhanced_features:
            feature_names = list(enhanced_features.keys())
            coefficients = list(enhanced_features.values())
            
            colors = ['red' if coef > 0 else 'blue' for coef in coefficients]
            bars = ax6.barh(range(len(feature_names)), coefficients, color=colors, alpha=0.7, edgecolor='black')
            ax6.set_yticks(range(len(feature_names)))
            ax6.set_yticklabels([name.replace('_', '\n') for name in feature_names])
            ax6.set_xlabel('Coefficient Value')
            ax6.set_title('Enhanced Feature Coefficients')
            ax6.grid(True, alpha=0.3)
            ax6.axvline(x=0, color='black', linestyle='-', alpha=0.8)
            
            # Add value labels
            for bar, value in zip(bars, coefficients):
                ax6.text(value + (0.01 if value > 0 else -0.01), bar.get_y() + bar.get_height()/2, 
                        f'{value:.3f}', ha='left' if value > 0 else 'right', va='center')
        else:
            ax6.text(0.5, 0.5, 'Enhanced feature coefficients\nnot available', ha='center', va='center', transform=ax6.transAxes)
            ax6.set_title('Enhanced Feature Coefficients')
    else:
        ax6.text(0.5, 0.5, 'Model coefficients not available', ha='center', va='center', transform=ax6.transAxes)
        ax6.set_title('Enhanced Feature Coefficients')
    
    plt.tight_layout()
    
    # Save the enhanced model analysis
    enhanced_viz_filename = 'enhanced_model_analysis.png'
    plt.savefig(enhanced_viz_filename, dpi=300, bbox_inches='tight')
    print(f"Enhanced model analysis saved as {enhanced_viz_filename}")
    
    return enhanced_viz_filename

def save_results(results, output_file="sited_scores.csv"):
    """
    Save the model results with proper column ordering.
    
    Parameters:
    results (pd.DataFrame): Results dataframe with predictions
    output_file (str): Output filename
    """
    # Ensure coordinates are first columns in output, followed by key metrics
    output_cols = ['longitude', 'latitude']
    priority_cols = ['latest_count_id', 'predicted_risk', 'risk_percentile', 'risk_category', 'collision_count']
    risk_metrics = ['speed_risk', 'volume_risk', 'collision_history', 'heavy_share', 'near_school', 'enforcement_gap']
    
    # Add percentile and category columns for risk metrics
    percentile_cols = [f'{m}_percentile' for m in ['speed_risk', 'volume_risk', 'collision_history', 'heavy_share'] if f'{m}_percentile' in results.columns]
    category_cols = [f'{m}_category' for m in ['speed_risk', 'volume_risk', 'collision_history', 'heavy_share'] if f'{m}_category' in results.columns]
    
    remaining_cols = [col for col in results.columns if col not in output_cols + priority_cols + risk_metrics + percentile_cols + category_cols]
    final_output_cols = output_cols + priority_cols + risk_metrics + percentile_cols + category_cols + remaining_cols
    
    # Filter to only existing columns
    final_output_cols = [col for col in final_output_cols if col in results.columns]
    
    # Reorder columns and save
    results_ordered = results[final_output_cols]
    results_ordered.sort_values('predicted_risk', ascending=False).to_csv(output_file, index=False)
    
    print(f"\nResults saved to {output_file}")
    print(f"Processed {len(results)} sites")
    print(f"Output columns: {final_output_cols[:8]}...")  # Show first 8 columns
    print(f"Top 5 highest risk sites:")
    display_cols = ['longitude', 'latitude', 'latest_count_id', 'predicted_risk', 'risk_percentile', 'collision_count']
    display_cols = [col for col in display_cols if col in results.columns]
    print(results.nlargest(5, 'predicted_risk')[display_cols].to_string(index=False))

def main():
    """
    Main function to run the modeling pipeline.
    """
    print("SpeedShield Model Training and Evaluation")
    print("=" * 40)
    
    try:
        # Load training data
        training_data = load_training_data()
        
        print("Fitting model...")
        model, results = fit_model(training_data)
        
        # Add percentiles and create visualizations
        results = add_percentiles_and_visualize(results)
        
        # Create enhanced model analysis
        create_enhanced_model_analysis(results, model)
        
        # Save results
        save_results(results)
        
        print("\nModel training and evaluation completed successfully!")
        
    except Exception as e:
        print(f"Error: {e}")
        print("\nPlease ensure the training data file exists and is properly formatted.")
        return 1
    
    return 0

main()
