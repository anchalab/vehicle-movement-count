import base64
import io
import pandas as pd
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull, distance
import cv2
import numpy as np
import matplotlib.cm as cm
from sklearn.metrics import pairwise_distances_argmin_min
import matplotlib.lines as mlines
import string
import matplotlib.patches as patches
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

def analyze(csv_file, image_path, cluster_count):
    column_names = ['frame','vehicle_id','vehicle_type','a','b','c','X1','Y1','X2','Y2',]
    df = pd.read_csv(csv_file, sep=' ', header=None, usecols = [0,1,2,3,4,5,6,7,8,9], names=column_names)
    
    df['Center_X'] = (df['X1']+df['X2']) / 2
    df['Center_Y'] = (df['Y1']+df['Y2']) / 2

    df = df[['frame','vehicle_id','vehicle_type','Center_X','Center_Y']]

    df_sorted = df.sort_values(by=['vehicle_id','frame']).reset_index(drop=True)

    df_sorted = df_sorted.groupby('vehicle_id').filter(
        lambda x: np.sqrt((x['Center_X'].iloc[-1] - x['Center_X'].iloc[0])**2 +
                        (x['Center_Y'].iloc[-1] - x['Center_Y'].iloc[0])**2) > 100
    )    

    df_first_frame = df_sorted.groupby('vehicle_id').head(1).reset_index(drop=True)
    df_last_frame = df_sorted.groupby('vehicle_id').tail(1).reset_index(drop=True)
    df_combined = pd.concat([df_first_frame, df_last_frame]).sort_values(by=['vehicle_id','frame']).reset_index(drop=True)

    # Assume df_first contains the first occurrence of each vehicle with 'x_center' and 'y_center'
    # normalize X, Y coordinates
    df_combined['Center_X_norm'] = (df_combined['Center_X'] - df_combined['Center_X'].mean()) / df_combined['Center_X'].std()
    df_combined['Center_Y_norm'] = (df_combined['Center_Y'] - df_combined['Center_Y'].mean()) / df_combined['Center_Y'].std()

    # Step 1: Extract the coordinates
    X_combined = df_combined[['Center_X_norm', 'Center_Y_norm']].values

    # Step 2: Apply K-Means clustering
    k = cluster_count  # Define the number of clusters
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)

    # Step 3: Fit the K-Means model
    df_combined['Cluster'] = kmeans.fit_predict(X_combined)

    # Compute centroid of each cluster
    centroids = pd.DataFrame(kmeans.cluster_centers_, columns=['Center_X_norm', 'Center_Y_norm'])

    unique_clusters = df_combined['Cluster'].unique()

    # Create a dynamic mapping from cluster ID to gate name
    cluster_to_gate = {cluster_id: f'Gate {i+1}' for i, cluster_id in enumerate(unique_clusters)}

    # Assign the corresponding gate names to the 'Gate' column
    df_combined['Gate'] = df_combined['Cluster'].map(cluster_to_gate)

    # Step 1: Group by 'Gate' (cluster) and apply Convex Hull for each cluster
    fig, ax = plt.subplots()

    for gate, group in df_combined.groupby('Gate'):
    # for Cluster, group in df_first_clustered_group:
        X_group = group[['Center_X','Center_Y']].values
        if len(X_group) < 3:
            # If less than 3 points, just plot the points without Convex Hull
            # for x, y in X_group:
            #     plt.text(x, y, f'{gate}', fontsize=10, color='blue', fontweight='bold',
            #              bbox=dict(facecolor='white', alpha=0.7, edgecolor='blue'))
            continue  # Convex Hull requires at least 3 points
        hull = ConvexHull(X_group)
        ax.scatter(group['Center_X'],group['Center_Y'], label=gate, s=10)
        
        for simplex in hull.simplices:
            ax.plot(X_group[simplex, 0], X_group[simplex, 1], 'k-')
            
        # Compute an appropriate label position (e.g., centroid of the convex hull)
        hull_points = X_group[hull.vertices]
        centroid_x, centroid_y = hull_points.mean(axis=0)

        # Place text label near the convex hull boundary
        plt.text(centroid_x, centroid_y, f'{gate}', 
                fontsize=10, color='red', fontweight='bold', 
                bbox=dict(facecolor='white', alpha=0.7, edgecolor='red'))
    ax.imshow(cv2.imread(image_path))

    # Add labels and legend
    ax.set_title("Convex Hulls for Each Cluster")
    ax.set_xlabel("Center_X")
    ax.set_ylabel("Center_Y")
    ax.legend()
    # plt.show()

    # Save the plot to a base64-encoded image
    buf = io.BytesIO()
    fig.savefig(buf, format='jpg', bbox_inches='tight')
    buf.seek(0)
    image_base64 = base64.b64encode(buf.read()).decode('utf-8')
    buf.close()
    plt.close(fig)  # Prevent memory leaks if called repeatedly

    
    # Pivot to wide format by assigning first and last values
    df_wide = df_combined.groupby('vehicle_id').agg(
        **{"Start Frame":('frame', 'first'),
        "End Frame":('frame', 'last'),
        "Class":('vehicle_type', 'first'),
        "InGate":('Gate', 'first'),
        "OutGate":('Gate', 'last')}
    ).reset_index()
    df_wide.rename(columns={'vehicle_id':'Vehicle ID'}, inplace=True)

    df_wide['Counted'] = np.where(df_wide['InGate'] == df_wide['OutGate'], '', 'Y')
    
    new_column_order = ['Vehicle ID','Class','Start Frame','End Frame','Counted','InGate','OutGate']
    output = df_wide[new_column_order]

    df_count = df_wide[['InGate','OutGate']].value_counts().reset_index(name='Vehicle Count').reset_index(drop=True)

    # output.to_csv("../output.csv",index=False)

    return image_base64, df_count, output

