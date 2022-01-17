import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
from scipy.signal import convolve2d
from tqdm import tqdm
import dask as da
from dask.diagnostics import ProgressBar

def cart2pol(x, y):
    r  = np.sqrt(x**2 + y**2)
    th = np.arctan2(y, x)
    return r, th

def pol2cart(r, th):
    x = r * np.cos(th)
    y = r * np.sin(th)
    return x, y

def get_cat_as_nb(ds):
    """For a given ds, returns the category as a number (0 for storm or dep, 1, ..., 5 for other categories)"""
    cat = np.array(ds['cyclone_category'])
    if cat == 'storm' or cat == 'dep':
        cat = 0
    else: # then it's 'cat-0', 1, ..., or 5
        cat = int(str(cat)[-1])
    return cat

def plot_wind_field(ds_all, time_idx):
    '''Given an xarray.Dataset object containing all the files (e.g 341) and a time index, plots the corresponding TC wind field using plt.pcolormesh().'''
    # Open figure
    fig, _ = plt.subplots()
    
    # Title
    name = ds_all.isel(time=time_idx)['storm_name'].values
    tcId = ds_all.isel(time=time_idx)['storm_id'].values
    cat  = int(ds_all.isel(time=time_idx)['cat'])
    plt.title('SAR wind field\n%s, %s, Cat. %i'%(name, tcId, cat), weight='bold')
    
    # Plot
    x, y = np.meshgrid(ds_all.isel(time=time_idx)['x'], ds_all.isel(time=time_idx)['y']) # Defining 2D coordinates
    plt.pcolormesh(x, y, ds_all.isel(time=time_idx)['wind_speed'])                       # Plot the wind field
    plt.xlabel('x (m)');plt.ylabel('y (m)');plt.grid()                                 # Add axis legend
    plt.subplots_adjust(right=0.8)                                                       # Colorbar - line 1
    cbar = plt.colorbar()                                                                # Colorbar - line 2
    cbar.set_label('Wind speed (m/s)', rotation=270);cbar.ax.get_yaxis().labelpad=25     # Colorbar - line 3
    
def find_time_idx(ds_all, storm_name):
    '''Given an xarray.Dataset object containing all the files (e.g 341) and a storm name (string), 
    returns a 1D array containing all the time indexes corresponding to the storm name in the global Dataset.'''
    return np.where(ds_all['storm_name']==storm_name)[0]

def compute_SAR_2D_Rmax_Vmax(ds):
    # Put zeros where NaNs and ones elsewhere
    ds_ws                    = np.asarray(ds['wind_speed'])
    ds_ones                  = np.ones(ds_ws.shape)
    ds_ones[np.isnan(ds_ws)] = 0.

    # Convolve this with a 10 x 10 kernel of ones to count the number of NaNs
    kernel                   = np.ones((10, 10))
    valid_counter            = convolve2d(ds_ones, kernel, mode='same')
    valid_counter           /= kernel.shape[0] * kernel.shape[1]

    # Set to NaN where threshold is exceeded
    thresh                        = 0.99
    ds_ws[valid_counter < thresh] = np.nan # if there is less than 99% (thresh = 0.99) of valid values, we set to nan 

    # Compute Rmax_SAR_2D and Vmax_SAR_2D
    idx_max                  = np.nanargmax(ds_ws)
    Vmax_SAR_2D              = np.reshape(np.asarray(ds_ws), -1)[idx_max]
    Rmax_SAR_2D              = np.reshape(np.asarray(ds['r']), -1)[idx_max]
    
    return Rmax_SAR_2D, Vmax_SAR_2D

def compute_Rmax_Vmax(ds, r):
    # Put zeros where NaNs and ones elsewhere
    ds_ws                    = np.asarray(ds['wind_speed'])
    ds_ones                  = np.ones(ds_ws.shape)
    ds_ones[np.isnan(ds_ws)] = 0.

    # Convolve this with a 10 x 10 kernel of ones to count the number of NaNs
    kernel                   = np.ones((10, 10))
    valid_counter            = convolve2d(ds_ones, kernel, mode='same')
    valid_counter           /= kernel.shape[0] * kernel.shape[1]

    # Set to NaN where threshold is exceeded
    thresh                        = 0.99
    ds_ws[valid_counter < thresh] = np.nan # if there is less than 99% (thresh = 0.99) of valid values, we set to nan 

    # Compute Rmax_SAR_2D and Vmax_SAR_2D
    idx_max                  = np.nanargmax(ds_ws)
    Vmax                     = np.reshape(np.asarray(ds_ws), -1)[idx_max]
    Rmax                     = np.reshape(r, -1)[idx_max]
    
    return Rmax, Vmax

def create_reference_axis(size, interval):
    '''Given a number of points and a step value, returns a an axis where values go from 0 to size with points every interval.'''
    pts_per_step = 1 / interval
    return np.linspace(0, size, int(size * pts_per_step) + 1)

def get_polar_grid(ds_all):
    '''Given the global xarray.Dataset ds_all, 
    returns a polar grid (ds_r, ds_th) of shape (1000, 1000) in the right orientation, and corresponding to (ds_all['x'], ds_all['y'])'''
    ds          = ds_all.isel(time=0)
    ds_r, ds_th = cart2pol(ds['x'], ds['y'])
    ds_th       = np.rad2deg(np.pi / 2 - ds_th)
    ds_th       = np.mod(ds_th, 360)
    # plt.pcolormesh(ds_th);plt.colorbar() # to check TC orientation/angle definition
    return ds_r, ds_th

def get_polar_coords_r_star(ds, ds_r, ds_th):
    # Assign polar coordinates
    ds            = ds.assign_coords({'r': ds_r, 'th': ds_th})
    # Divide r_values by Rmax
    Rmax, Vmax    = compute_SAR_2D_Rmax_Vmax(ds)
    ds_r         /= Rmax
    
    return ds_r, ds_th, Rmax, Vmax

def interpolate_to_reference_polar_grid_r_star(ds, ds_r, ds_th, r_star_reg_grid, theta_ref_grid):
    ds_ws         = np.array(ds['wind_speed']) # convert to array
    ds_r          = np.array(ds_r)
    ds_th         = np.array(ds_th)
    return griddata((ds_r.flatten(), ds_th.flatten()), ds_ws.flatten(), (r_star_reg_grid, theta_ref_grid), method='nearest')

# @da.delayed
def get_ds_in_polar_r_star_coords(r_star_ref_ax, theta_ref_ax, ds_all, time_idx):
    # Create reference polar grid
    r_star_reg_grid, theta_ref_grid = np.meshgrid(r_star_ref_ax, theta_ref_ax)
    
    # Polar coordinates of ds in r* referential
    ds                      = ds_all.isel(time=time_idx)
    ds_r, ds_th             = get_polar_grid(ds_all)
    ds_r, ds_th, Rmax, Vmax = get_polar_coords_r_star(ds, ds_r, ds_th)

    # Interpolate wind_speed on this reference grid
    ds_ws_polar_r_star      = interpolate_to_reference_polar_grid_r_star(ds, ds_r, ds_th, r_star_reg_grid, theta_ref_grid)
    ds_ws_polar_r_star      = np.transpose(ds_ws_polar_r_star) # Convert (th, r*) to (r*, th)
    ds_ws_polar_r_star      = np.expand_dims(ds_ws_polar_r_star, axis=0)

    # Create new xarray.Dataset with polar r* coordinates
    time     = ds_all['time'].values[time_idx] 
    ds_polar = xr.Dataset({'cat':        xr.DataArray(int(ds['cat']),               coords={'time': [time]}, dims=['time']),
                           'storm_name': xr.DataArray(str(ds['storm_name'].values), coords={'time': [time]}, dims=['time']),
                           'storm_id':   xr.DataArray(str(ds['storm_id'].values),   coords={'time': [time]}, dims=['time']),
                           'storm_id':   xr.DataArray(str(ds['storm_id'].values),   coords={'time': [time]}, dims=['time']),
                           'rmax':       xr.DataArray(Rmax,                         coords={'time': [time]}, dims=['time']),
                           'vmax':       xr.DataArray(Vmax,                         coords={'time': [time]}, dims=['time']),
                           # Wind speed
                           'wind_speed': xr.DataArray(ds_ws_polar_r_star, coords={'time': [time], 'r*': r_star_ref_ax, 'th': theta_ref_ax}, dims=['time', 'r*', 'th'])
    })
    
    return ds_polar

# @da.delayed
def get_ds_in_polar_r_star_coords_v02(ds_all, time_idx, res_ref=10, r_ref_divider=100000):
    '''Given the global xarray.Dataset ds_all, a time_idx and a resolution res_ref, 
    returns ds_polar, an xarray.Dataset containg ds['wind_speed'] interpolated on a (r*, th) polar grid
    Increase r_ref to degrade resolution;
    Decrease r_ref_divider to have a wider range of r* (e.g 0 < r* < 8 instead of 0 < r* < 4)'''
    # Meshgrid (x, y)
    ds   = ds_all.isel(time=time_idx)
    x, y = np.meshgrid(ds['x'], ds['y'])

    # Convert to polar (r, th)
    r, th = cart2pol(x, y)
    th    = np.pi / 2 - th

    # Compute Rmax and put on (r*, th)
    ds_r, ds_th      = get_polar_grid(ds_all)
    _, _, Rmax, Vmax = get_polar_coords_r_star(ds, ds_r, ds_th)
    ds               = ds.assign_coords({'r': ds_r, 'th': ds_th})
    Rmax, Vmax       = compute_Rmax_Vmax(ds, r)
    r               /= Rmax
    # Polar plot to check
    # plt.clf()
    # ax = plt.subplot(projection = "polar")
    # plt.pcolormesh(th, r, ds['wind_speed']);plt.colorbar()
    # ax.set_theta_zero_location("N")  # theta=0 at the top
    # ax.set_theta_direction(-1)  # theta increasing clockwise

    # Create reference grid
    x_ref         = x[::res_ref, ::res_ref]
    y_ref         = y[::res_ref, ::res_ref]
    r_ref, th_ref = cart2pol(x_ref, y_ref)
    r_ref        /= r_ref_divider              # Scale to have r_star
    th_ref        = np.pi / 2 - th_ref

    # Interpolate ds['wind_speed'] to this reference grid
    ds_ws         = np.array(ds['wind_speed'])
    ws_interp     = griddata((r.flatten(), th.flatten()), ds_ws.flatten(), (r_ref, th_ref), method='nearest')
    
    time     = ds_all['time'].values[time_idx] 
    ws_interp = np.expand_dims(ws_interp, axis=0)
    ds_polar = xr.Dataset({'cat':        xr.DataArray(int(ds['cat']),               coords={'time': [time]}, dims=['time']),
                           'storm_name': xr.DataArray(str(ds['storm_name'].values), coords={'time': [time]}, dims=['time']),
                           'storm_id':   xr.DataArray(str(ds['storm_id'].values),   coords={'time': [time]}, dims=['time']),
                           'storm_id':   xr.DataArray(str(ds['storm_id'].values),   coords={'time': [time]}, dims=['time']),
                           'rmax':       xr.DataArray(Rmax,                         coords={'time': [time]}, dims=['time']),
                           'vmax':       xr.DataArray(Vmax,                         coords={'time': [time]}, dims=['time']),
                           # Wind speed
                           'wind_speed': xr.DataArray(ws_interp, coords={'time': [time], 'r*_grid': (('r*', 'th'), r_ref), 'th_grid': (('r*', 'th'), th_ref)}, dims=['time', 'r*', 'th'])
    })
    return ds_polar

def get_ds_in_polar_r_star_coords_v03(ds_all, time_idx, r_ref_ax, th_ref_ax):
    # Create reference grid
    r_ref, th_ref = np.meshgrid(r_ref_ax, th_ref_ax)

    # Select ds and convert to polar (r, th)
    ds    = ds_all.isel(time=time_idx)
    x, y  = np.meshgrid(ds['x'], ds['y'])
    r, th = cart2pol(x, y)
    th    = np.pi / 2 - th

    # Compute Rmax and put on (r*, th)
    ds_r, ds_th      = get_polar_grid(ds_all)
    _, _, Rmax, Vmax = get_polar_coords_r_star(ds, ds_r, ds_th)
    ds               = ds.assign_coords({'r': ds_r, 'th': ds_th})
    Rmax, Vmax       = compute_Rmax_Vmax(ds, r)
    r               /= Rmax
    # # Polar plot to check
    # plt.clf()
    # ax = plt.subplot(projection = "polar")
    # plt.pcolormesh(th, r, ds['wind_speed']);plt.colorbar()
    # ax.set_theta_zero_location("N")  # theta=0 at the top
    # ax.set_theta_direction(-1)  # theta increasing clockwise

    # Interpolate ds['wind_speed'] to this reference grid
    ds_ws         = np.array(ds['wind_speed'])
    ds_r          = np.array(ds_r)
    ds_th         = np.array(ds_th)
    ws_interp     = griddata((ds_r.flatten(), ds_th.flatten()), ds_ws.flatten(), (r_ref, th_ref), method='nearest')
    # # Polar plot to check
    # plt.clf()
    # ax = plt.subplot(projection = "polar")
    # plt.pcolormesh(np.deg2rad(th_ref), r_ref, ws_interp);plt.colorbar() # Put theta in radians!! 
    # ax.set_theta_zero_location("N")  # theta=0 at the top
    # ax.set_theta_direction(-1)  # theta increasing clockwise

    time      = ds_all['time'].values[time_idx] 
    ws_interp = np.expand_dims(ws_interp, axis=0)
    ds_polar  = xr.Dataset({'cat':       xr.DataArray(int(ds['cat']),               coords={'time': [time]}, dims=['time']),
                           'storm_name': xr.DataArray(str(ds['storm_name'].values), coords={'time': [time]}, dims=['time']),
                           'storm_id':   xr.DataArray(str(ds['storm_id'].values),   coords={'time': [time]}, dims=['time']),
                           'storm_id':   xr.DataArray(str(ds['storm_id'].values),   coords={'time': [time]}, dims=['time']),
                           'rmax':       xr.DataArray(Rmax,                         coords={'time': [time]}, dims=['time']),
                           'vmax':       xr.DataArray(Vmax,                         coords={'time': [time]}, dims=['time']),
                           'lat':        xr.DataArray(float(ds['lat'][499, 499]),   coords={'time': [time]}, dims=['time']),
                           'lon':        xr.DataArray(float(ds['lon'][499, 499]),   coords={'time': [time]}, dims=['time']),
                           # Wind speed
                           'wind_speed': xr.DataArray(ws_interp, coords={'time': [time], 'th': ('th', th_ref_ax), 'r*': ('r*', r_ref_ax)}, dims=['time', 'th', 'r*']),
    })
    return ds_polar

def covariance_matrix(ds_polar_all, order_by=None, print_shapes=False):
    '''Given an xarray.Dataset in polar (r, th) coordinates, returns the covariance matrix of that dataset.
    Possible values for order_by: 'r*_grid', 'th_grid'
    '''
    ds_polar_stack = ds_polar_all.stack(dimensions={'pixel': ('r*', 'th')})
    if order_by:
        ds_polar_stack = ds_polar_stack.sortby(order_by) # reorder by r* for instance
        
    x_Ex            = ds_polar_stack['wind_speed'] - ds_polar_stack.mean(dim='pixel', skipna=True)['wind_speed']  # X - mean(X) for each pixel
    # Expand dimension to prepare matrix multiplication
    x_Ex_expanded   = x_Ex.assign_coords(y='vector_dim')
    x_Ex_expanded   = x_Ex_expanded.expand_dims('vector_dim', axis=2)
    if print_shapes: print('x_Ex_expanded shape: ', x_Ex_expanded.shape)
    x_Ex_transposed = x_Ex_expanded.transpose('time', 'vector_dim', 'pixel')
    if print_shapes: print('x_Ex_transposed shape: ', x_Ex_transposed.shape)
    # Matrix multiplication
    a       = np.array(x_Ex_expanded)
    b       = np.array(x_Ex_transposed)
    product = np.einsum('ijk, ikl -> ijl', a, b) # input: (20, 10 000, 1) and (20, 1, 10 000) -> output: (20, 10 000, 10 000)
    if print_shapes: print('product shape: ', product.shape)
    # Mean on time to get covariance
    cov_mat = np.nanmean(product, axis=0)
    if print_shapes: print('cov_mat shape: ', cov_mat.shape)
    
    return cov_mat, ds_polar_stack

def covariance_matrix_1D(ds_1D, dim='r*', print_shapes=False):
    '''Given an xarray.Dataset in 1D r* coordinates, returns the covariance matrix of that dataset.
    dim is the dimension used for covariance computation, either r* or th.
    '''
    x_Ex            = ds_1D['wind_speed'] - ds_1D.mean(dim=dim, skipna=True)['wind_speed']  # X - mean(X) for each pixel
    # Expand dimension to prepare matrix multiplication
    x_Ex_expanded   = x_Ex.assign_coords(y='vector_dim')
    x_Ex_expanded   = x_Ex_expanded.expand_dims('vector_dim', axis=2)
    if print_shapes: print('x_Ex_expanded shape: ', x_Ex_expanded.shape)
    x_Ex_transposed = x_Ex_expanded.transpose('time', 'vector_dim', dim)
    if print_shapes: print('x_Ex_transposed shape: ', x_Ex_transposed.shape)
    # Matrix multiplication
    a       = np.array(x_Ex_expanded)
    b       = np.array(x_Ex_transposed)
    product = np.einsum('ijk, ikl -> ijl', a, b) # input: (20, 10 000, 1) and (20, 1, 10 000) -> output: (20, 10 000, 10 000)
    if print_shapes: print('product shape: ', product.shape)
    # Mean on time to get covariance
    cov_mat = np.nanmean(product, axis=0)
    if print_shapes: print('cov_mat shape: ', cov_mat.shape)
    return cov_mat, product

def filter_out_wind_speed(ds_all, time_idx):
    ds_ws                    = np.asarray(ds_all.isel(time=time_idx)['wind_speed'])
    ds_ones                  = np.ones(ds_ws.shape)
    ds_ones[np.isnan(ds_ws)] = 0.

    # Convolve this with a 10 x 10 kernel of ones to count the number of NaNs
    kernel                   = np.ones((10, 10))
    valid_counter            = convolve2d(ds_ones, kernel, mode='same')
    valid_counter           /= kernel.shape[0] * kernel.shape[1]

    # Set to NaN where threshold is exceeded
    thresh                   = 0.99
    thresh_wind_speed        = 50   # We filter only if the pixel has a value superior than 50 m/s
    mask                     = valid_counter < thresh
    mask[ds_ws < 50]         = False
    ds_ws[mask]              = np.nan # if there is less than 99% (thresh = 0.99) of valid values, we set to nan 

    # Create new dataArray with right wind speed
    ds_ws = np.expand_dims(ds_ws, axis=0)
    new_ws = xr.DataArray(data=ds_ws,
                          dims=['time', 'x', 'y'],
                          coords={'time': (['time'], [ds_all.isel(time=time_idx)['time'].values]),
                                  'x': (['x'], np.array(ds_all.isel(time=time_idx)['x'])),
                                  'y': (['y'], np.array(ds_all.isel(time=time_idx)['y']))})
    ds = xr.Dataset({'wind_speed': new_ws})
    return ds

def filter_out_whole_dataset(ds_all):
    listOfDatasets = []
    print('Filtering wind speed pixel outliers...')
    for time_idx in tqdm(range(len(ds_all['time']))):
        ds = filter_out_wind_speed(ds_all, time_idx)
        listOfDatasets.append(ds)
    
    new_values = xr.concat(listOfDatasets, dim='time')
    ds_all     = ds_all.update(new_values)
    
    return ds_all

# @da.delayed
def save_wind_field(ds_all, time_idx, save_dir):
    # Open file
    ds          = ds_all.isel(time=time_idx)
    
    # Get additional information
    TC_cat      = int(ds['cat'].values)
    TC_name     = str(ds['storm_name'].values)
    TC_id       = str(ds['storm_id'].values)
    
    # Plot
    fig, _ = plt.subplots()
    plt.title('SAR wind field\n%s, %s'%(TC_name, TC_id) + ', Cat. %i'%TC_cat, weight='bold')
    
    x, y   = np.meshgrid(ds['x'], ds['y'])
    plt.pcolormesh(x, y, ds['wind_speed'][:, :])
    
    plt.xlabel('x (m)');plt.ylabel('y (m)');plt.grid()
    plt.subplots_adjust(right=0.8)
    cbar = plt.colorbar()
    cbar.set_label('Wind speed (m/s)', rotation=270);cbar.ax.get_yaxis().labelpad=25
    
    # Save
    fig.set_size_inches(25, 18)
    plt.savefig(save_dir + '{0:03}'.format(time_idx) + '.jpg', dpi=100)
    return None

def save_all_wind_fields(ds_all, save_dir):
    for time_idx in range(len(ds_all['time'])):
        save_wind_field(ds_all, time_idx, save_dir) 