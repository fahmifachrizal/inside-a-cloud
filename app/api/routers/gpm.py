from fastapi import APIRouter, HTTPException, Query, Response
from fastapi.responses import JSONResponse
import xarray as xr
import numpy as np
import matplotlib
# Use Agg backend immediately to prevent server GUI errors
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.collections import PatchCollection
from geojson import Feature, FeatureCollection, MultiPolygon
from scipy.ndimage import gaussian_filter
import os
import io

# --- PROJECT IMPORTS ---
from app.services import gpm_service 

router = APIRouter()

# Thresholds for rain intensity (mm/hr)
LEVELS = [0.1, 0.5, 5.0, 10.0, 20.0]

# ==========================================
# 1. HELPER: CENTRALIZED DATA PROCESSING
# ==========================================
def _load_and_process_gpm(filename: str, bounds: dict):
    """
    Handles loading, slicing, transposing, flipping, and smoothing.
    Returns: (dataset_handle, lats, lons, raw_data, smooth_data)
    """
    file_path = f"app/data/{filename}"
    if not os.path.exists(file_path):
        raise FileNotFoundError("File not found")

    # A. Open Dataset
    try:
        ds = xr.open_dataset(file_path, engine='h5netcdf', group='Grid', decode_times=False)
    except OSError:
        ds = xr.open_dataset(file_path, engine='h5netcdf', decode_times=False)

    # B. Identify Variable
    if 'precipitation' in ds:
         data = ds['precipitation'][0]
    elif 'precipitationCal' in ds:
         data = ds['precipitationCal'][0]
    else:
         ds.close()
         raise ValueError("No precipitation variable found")

    # C. Slice Data
    lat_name = next((k for k in ds.coords if 'lat' in k.lower()), 'lat')
    lon_name = next((k for k in ds.coords if 'lon' in k.lower()), 'lon')

    subset = data.sel({
        lat_name: slice(bounds['bottom'], bounds['top']),
        lon_name: slice(bounds['left'], bounds['right'])
    })
    
    # Fallback for inverted latitude order (North->South)
    if subset.size == 0:
        subset = data.sel({
            lat_name: slice(bounds['top'], bounds['bottom']),
            lon_name: slice(bounds['left'], bounds['right'])
        })

    # D. Prepare Numpy Arrays
    precip_vals = np.nan_to_num(subset.values)
    lats = subset.coords[lat_name].values
    lons = subset.coords[lon_name].values

    # E. Fix Shape (Transpose if (Lon, Lat) -> (Lat, Lon))
    # Target shape: (Rows=Lat, Cols=Lon)
    if precip_vals.shape == (len(lons), len(lats)):
        precip_vals = precip_vals.T
    elif precip_vals.shape != (len(lats), len(lons)):
         # Handle singleton dims like (1, lat, lon)
         precip_vals = np.squeeze(precip_vals)
         if precip_vals.shape == (len(lons), len(lats)): 
             precip_vals = precip_vals.T

    # F. Force Monotonicity (Fix Inverted Axes for Matplotlib)
    if lats[0] > lats[-1]:
        lats = np.flip(lats)
        precip_vals = np.flipud(precip_vals)
    
    if lons[0] > lons[-1]:
        lons = np.flip(lons)
        precip_vals = np.fliplr(precip_vals)

    # G. Generate Smoothed Data (For Vectorizing)
    # sigma=1 connects scattered pixels into blobs suitable for contouring
    precip_smooth = gaussian_filter(precip_vals, sigma=1.0)

    return ds, lats, lons, precip_vals, precip_smooth

# ==========================================
# 2. MAIN ENDPOINT
# ==========================================
@router.get("/")
async def get_gpm_data(
    filename: str = Query(...),
    toplat: float = Query(...),
    bottomlat: float = Query(...),
    leftlon: float = Query(...),
    rightlon: float = Query(...),
    draw: str = Query("vector", enum=["vector", "plot"], description="Output mode")
):
    """
    Unified Endpoint for GPM Data.
    - draw='vector': Returns 3D GeoJSON Polygons (smoothed)
    - draw='plot': Returns a transparent PNG overlay (scatter + vectors)
    """
    bounds = {'top': toplat, 'bottom': bottomlat, 'left': leftlon, 'right': rightlon}

    try:
        # 1. Process Data
        ds, lats, lons, raw_data, smooth_data = _load_and_process_gpm(filename, bounds)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="File not found")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    try:
        # ==========================
        # MODE A: VECTOR (GeoJSON)
        # ==========================
        if draw == "vector":
            features = []
            
            # Use matplotlib to calculate contours purely mathematically (no visible plot)
            fig, ax = plt.subplots()
            
            for level in LEVELS:
                if np.max(smooth_data) < level: continue

                # Use contour (Lines) + allsegs for robust extraction
                cs = ax.contour(lons, lats, smooth_data, levels=[level])
                
                if len(cs.allsegs) > 0:
                    for vertices in cs.allsegs[0]:
                        if len(vertices) < 3: continue
                        
                        poly_coords = vertices.tolist()
                        
                        # --- FIX START ---
                        # Check if the polygon is closed. If not, snap the last point to the first.
                        # This eliminates the "Giant Wall" artifact at the edges of the map.
                        if poly_coords[0] != poly_coords[-1]:
                            poly_coords.append(poly_coords[0])
                        # --- FIX END ---
                        
                        polygon_structure = [poly_coords] 
                        
                        features.append(Feature(
                            geometry=MultiPolygon([polygon_structure]),
                            properties={"level": level}
                        ))
            
            plt.close(fig)
            ds.close()
            return JSONResponse(content=FeatureCollection(features))

        # ==========================
        # MODE B: PLOT (Image)
        # ==========================
        elif draw == "plot":
            fig, ax = plt.subplots(figsize=(10, 8), dpi=100)
            
            # 1. Raw Data Scatter (Blue Dots)
            xx, yy = np.meshgrid(lons, lats)
            mask = raw_data > 0.1
            ax.scatter(
                xx[mask], yy[mask], 
                s=raw_data[mask] * 10, # Size relative to intensity
                c='cyan', alpha=0.6, label="Raw Data"
            )

            # 2. Vector Overlay (Red Lines)
            # Use the exact same smoothing logic as Vector mode to ensure visual match
            debug_polys = []
            for level in LEVELS:
                if np.max(smooth_data) < level: continue
                
                cs = ax.contour(lons, lats, smooth_data, levels=[level], colors=['red'], linewidths=1.5, alpha=0.8)
                # Keep lines visible on the plot

            # 3. Formatting
            ax.set_xlim(leftlon, rightlon)
            ax.set_ylim(bottomlat, toplat)
            ax.axis('off') # Transparent background, no axis
            
            # Save to Buffer
            buf = io.BytesIO()
            plt.savefig(buf, format='png', transparent=True, bbox_inches='tight', pad_inches=0)
            buf.seek(0)
            plt.close(fig)
            ds.close()
            
            return Response(content=buf.getvalue(), media_type="image/png")

    except Exception as e:
        ds.close()
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Processing Error: {str(e)}")


# ==========================================
# 3. UTILITY ENDPOINTS
# ==========================================
@router.get("/files")
async def list_files():
    """List available HDF5 files."""
    return gpm_service.list_available_files()