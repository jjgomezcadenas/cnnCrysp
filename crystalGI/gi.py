import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def event_size(df):
    """
    Return the size of the df
    
    """
    return len(np.unique(df.event))


def event_mult(df):
    """
    Input: Data frame (df) ordered by event number. 
    Return: the multiplicity of the event
    """
    grdf = df.groupby('event')
    ekin = grdf['event'].count()
    return ekin
    

def fiducial_select(df, d, z):
    """
    Define a fiducial selection: |df.x, df.y| < d/2, |df.z| < z/2
    
    """
    df2 = df[np.abs(df['x'])<d/2]
    df3 = df2[np.abs(df2['y'])<d/2]
    df4 = df3[np.abs(df3['z'])<z/2]
    return df4


def fiducial_df(df, d=48.2, z=37.2):
    """
    Define a fiducial data frame which includes total energy and number o tracks
    in event
    
    """
    gdf = fiducial_select(df, d, z)
    grdf = gdf.groupby('event')
    gdfa = grdf.agg(
                etot = ('edep', 'sum'),
                ntrk = ('edep', 'count')).reset_index()
    
    return pd.merge(gdf, gdfa, on='event')


def twocluster_df(df):
    """
    Creates a df with the position and energy of the two clusters of maximum energy
    as well as their baricenter
    """
    grouped = df.groupby('event')

    # Initialize lists to store results
    results = []
    
    # Iterate over each event group
    for event_id, group in grouped:
        # Sort particles by energy in descending order
        sorted_group = group.sort_values(by='time', ascending=True).reset_index(drop=True)
        
        # Extract etot, ntrk from the first particle (they are the same for all particles in the event)
        event = sorted_group.loc[0, 'event']
        etot = sorted_group.loc[0, 'etot']
        ntrk = sorted_group.loc[0, 'ntrk']
        
        # Get particle with early time
        t1, x1, y1, z1, e1 = sorted_group.loc[0, ['time','x', 'y', 'z', 'edep']]
        
        # Get particle with second early time
        if len(sorted_group) > 1:
            t2, x2, y2, z2, e2 = sorted_group.loc[1, ['time','x', 'y', 'z', 'edep']]
        else:
            t2, x2, y2, z2, e2 = t1, x1, y1, z1, e1  # If no second particle, set to first particle
        
        
        
        # Append the results
        results.append({'event':event, 'etot':etot, 'ntrk': ntrk,
            't1': t1, 'x1': x1, 'y1': y1, 'z1': z1, 'e1': e1,
            't2': t2, 'x2': x2, 'y2': y2, 'z2': z2, 'e2': e2,
        })
    
    # Create the final DataFrame
    return (pd.DataFrame(results).reset_index())


def d12(df):
    """
        Calculate d12: Distance between (x1, y1, z1) and (x2, y2, z2)
        
    """
    return np.sqrt((df['x2'] - df['x1'])**2 + (df['y2'] - df['y1'])**2 + (df['z2'] - df['z1'])**2)
    
def dtz(df):
    """
        Calculate d12: Distance between (x1, y1, z1) -> (minimum t) and
        (xz, yz, zz) -> (minimum z). This distance quantifies the error when one assigns
        the first cluster to the one with minimum z.
        
    """
    x1 = df['x1'].values 
    y1 = df['y1'].values 
    z1 = df['z1'].values

    x2 = df['x2'].values 
    y2 = df['y2'].values 
    z2 = df['z2'].values

    xz = np.where(z1 < z2, x1, x2)
    yz = np.where(z1 < z2, y1, y2)
    zz = np.where(z1 < z2, z1, z2)
        
    return np.sqrt((x1 - xz)**2 + (y1 - yz)**2 + (z1 - zz)**2)


def plot_amplitude(df, num_bins = 20, xmin=2e+4, xmax=4e+4, figsize=(6, 4), title=""):
    energies = df.groupby("event").sum().amplitude.values
    
    fig, ax0 = plt.subplots(1, 1, figsize=figsize)
    h = ax0.hist(energies, num_bins, (xmin, xmax))
    ax0.set_xlabel("Event energy")
    ax0.set_ylabel('Events/bin')
    ax0.set_title(title)
    
    fig.tight_layout()
    return h[0],h[1]


def select_image_from_df(df,evtsel, n = 8):
    """
    For an event evtsel, extracts the image matrix from
    the DF with sensor information
    
    """
    df2=df[df.event==evtsel]
    
    charge_matrix = np.zeros((n, n))
    sensor_id = df2['sensor_id'].values
    charge    = df2['amplitude'].values

    for id in sensor_id:
        charge_matrix[sensor_id[id] // n, sensor_id[id] % n] = charge[id]
    return charge_matrix


def get_gamma_position_in_pixels(df, x_spatial, y_spatial, debug=False):
    """
    Return the position of gammas in the pixel coordinate system
    df is a series containing the event, x1, y1, z1, e1, x2... 

    - xspatial, yspatial: coordinates of sensors
    
    """

    def transform_coordinates(x, y, x_spatial, y_spatial, debug,
                              x_min2=0,  x_max2=8,  y_min2=0,  y_max2 = 8):
        """
        Transform coordinates from space RS to pixel RS
        """
                        
        x_min1=x_spatial[0] 
        x_max1=x_spatial[-1] 
        y_min1=y_spatial[0]
        y_max1 = y_spatial[-1]
    
        if x < x_min1:
            x = x_min1
    
        if x > x_max1:
            x = x_max1
    
        if y < y_min1:
            y = y_min1
    
        if y > y_max1:
            y = y_max1
     
        # Apply the transformation for x and y
        x_new = ((x - x_min1) / (x_max1 - x_min1)) * (x_max2 - x_min2) + x_min2
        y_new = ((y - y_min1) / (y_max1 - y_min1)) * (y_max2 - y_min2) + y_min2
    
        return x_new, y_new
    
    xt1, yt1 = transform_coordinates(df.x1, df.y1, 
                                     x_spatial, y_spatial, debug)

    xt2, yt2 = transform_coordinates(df.x2, df.y2, 
                                     x_spatial, y_spatial, debug)
    

    if debug:
        print(f"xg1 = {df.x1:.2f}, yg1 ={df.y1:.2f}")
        print(f"xt1 = {xt1:.2f}, yt1 ={yt1:.2f}")
        print(f"xg2 = {df.x2:.2f}, yg2 ={df.y2:.2f}")
        print(f"xt2 = {xt2:.2f}, yt2 ={yt2:.2f}")

    return xt1, yt1,xt2, yt2
