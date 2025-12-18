import os
import struct
import xarray as xr
import numpy as np
from app.core.config import DATA_DIR

def _extract_cloud_arrays(filename, bounds, threshold):
    """
    CORE LOGIC: Opens file, crops to bounds, and returns raw numpy arrays 
    for points where rain > threshold.
    """
    file_path = os.path.join(DATA_DIR, filename)
    if not os.path.exists(file_path):
        raise FileNotFoundError("GPM File not found")

    # 1. Open
    try:
        ds = xr.open_dataset(file_path, group='Grid', engine='h5netcdf', decode_times=False)
    except:
        ds = xr.open_dataset(file_path, engine='h5netcdf', decode_times=False)

    # 2. Identify Vars
    candidates = ['precipitationCal', 'precipitation', 'precip']
    var_name = next((v for v in candidates if v in ds), None)
    lat_name = next((k for k in ds.coords if 'lat' in k.lower()), 'lat')
    lon_name = next((k for k in ds.coords if 'lon' in k.lower()), 'lon')

    if not var_name: raise ValueError("Variable not found")

    # 3. Crop
    lat_slice = slice(min(bounds['bottom'], bounds['top']), max(bounds['bottom'], bounds['top']))
    lon_slice = slice(min(bounds['left'], bounds['right']), max(bounds['left'], bounds['right']))
    
    try:
        ds_cropped = ds.sel({lat_name: lat_slice, lon_name: lon_slice})
        if ds_cropped[var_name].size == 0: ds_cropped = ds
    except:
        ds_cropped = ds

    # 4. Extract
    data = ds_cropped[var_name].squeeze().values
    lats = ds_cropped[lat_name].values
    lons = ds_cropped[lon_name].values

    if data.shape == (len(lons), len(lats)):
        data = data.T 

    # 5. Filter Sparse Data (Rain > Threshold)
    # Get indices
    y_idxs, x_idxs = np.where(data > threshold)
    
    # Extract values and cast to Float32 (Standard for WebGL/Binary)
    valid_rain = data[y_idxs, x_idxs].astype(np.float32)
    valid_lats = lats[y_idxs].astype(np.float32)
    valid_lons = lons[x_idxs].astype(np.float32)
    
    max_val = float(np.max(data)) if len(valid_rain) > 0 else 0.0
    
    ds.close()
    
    return valid_lats, valid_lons, valid_rain, max_val

def process_local_file(filename, bounds):
    """
    Opens HDF5, crops to bounds, returns (lats, lons, data).
    """
    file_path = os.path.join(DATA_DIR, filename)
    if not os.path.exists(file_path):
        raise FileNotFoundError("GPM File not found")

    # 1. Open
    try:
        ds = xr.open_dataset(file_path, group='Grid', engine='h5netcdf', decode_times=False)
    except:
        ds = xr.open_dataset(file_path, engine='h5netcdf', decode_times=False)

    # 2. Identify Variables
    candidates = ['precipitationCal', 'precipitation', 'precip']
    var_name = next((v for v in candidates if v in ds), None)
    if not var_name: raise ValueError("Variable not found in GPM file")

    lat_name = next((k for k in ds.coords if 'lat' in k.lower()), 'lat')
    lon_name = next((k for k in ds.coords if 'lon' in k.lower()), 'lon')

    # 3. Crop (Optimization)
    lat_slice = slice(min(bounds['bottom'], bounds['top']), max(bounds['bottom'], bounds['top']))
    lon_slice = slice(min(bounds['left'], bounds['right']), max(bounds['left'], bounds['right']))
    
    try:
        ds_cropped = ds.sel({lat_name: lat_slice, lon_name: lon_slice})
        if ds_cropped[var_name].size == 0: ds_cropped = ds # Fallback
    except:
        ds_cropped = ds

    # 4. Extract Arrays
    data = ds_cropped[var_name].squeeze().values
    lats = ds_cropped[lat_name].values
    lons = ds_cropped[lon_name].values

    # Transpose if (lon, lat)
    if data.shape == (len(lons), len(lats)):
        data = data.T 

    ds.close()
    return lats, lons, data

def list_available_files():
    if not os.path.exists(DATA_DIR): return []
    return [f for f in os.listdir(DATA_DIR) if f.endswith(('.HDF5', '.nc', '.nc4'))]

def get_sparse_cloud_data(filename, bounds, threshold=0.1):
    """
    Extracts precipitation data and converts it into a sparse JSON-friendly format.
    Only returns points where rain > threshold.
    """
    file_path = os.path.join(DATA_DIR, filename)
    if not os.path.exists(file_path):
        raise FileNotFoundError("GPM File not found")

    # 1. Open & Auto-Detect (Same logic as before)
    try:
        ds = xr.open_dataset(file_path, group='Grid', engine='h5netcdf', decode_times=False)
    except:
        ds = xr.open_dataset(file_path, engine='h5netcdf', decode_times=False)

    candidates = ['precipitationCal', 'precipitation', 'precip']
    var_name = next((v for v in candidates if v in ds), None)
    if not var_name: raise ValueError("Variable not found")

    lat_name = next((k for k in ds.coords if 'lat' in k.lower()), 'lat')
    lon_name = next((k for k in ds.coords if 'lon' in k.lower()), 'lon')

    # 2. Crop to Bounds
    # Ensure we don't load the whole world if we only need Java
    lat_slice = slice(min(bounds['bottom'], bounds['top']), max(bounds['bottom'], bounds['top']))
    lon_slice = slice(min(bounds['left'], bounds['right']), max(bounds['left'], bounds['right']))
    
    try:
        ds_cropped = ds.sel({lat_name: lat_slice, lon_name: lon_slice})
    except:
        ds_cropped = ds

    # 3. Extract Arrays
    data = ds_cropped[var_name].squeeze().values
    lats = ds_cropped[lat_name].values
    lons = ds_cropped[lon_name].values

    # Transpose if needed (we want shape: [lat, lon])
    if data.shape == (len(lons), len(lats)):
        data = data.T 

    # 4. Create Sparse Data (The Magic Step)
    # Find indices where it is actually raining
    # > threshold (0.1 mm/hr) filters out clear sky
    y_idxs, x_idxs = np.where(data > threshold)
    
    # Extract values at those indices
    valid_rain = data[y_idxs, x_idxs]
    valid_lats = lats[y_idxs]
    valid_lons = lons[x_idxs]

    # 5. Structure for Javascript
    # We return a list of objects or a "Columnar" format (more efficient for JS parsing)
    # Columnar is smaller/faster: { lats: [...], lons: [...], vals: [...] }
    
    # Convert numpy arrays to python lists for JSON serialization
    # Rounding float values significantly reduces JSON size
    response_data = {
        "lats": np.round(valid_lats, 3).tolist(),
        "lons": np.round(valid_lons, 3).tolist(),
        "vals": np.round(valid_rain, 2).tolist(),
        "stats": {
            "max": float(np.max(data)),
            "count": len(valid_rain)
        }
    }

    ds.close()
    return response_data