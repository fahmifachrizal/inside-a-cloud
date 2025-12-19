import io
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.collections import PatchCollection
import cartopy.crs as ccrs
import cartopy.io.img_tiles as cimgt

def generate_debug_heatmap(lats, lons, data, bounds, polygons=None):
    """
    Debug Plot:
    - Layer 1: Google Satellite Tiles
    - Layer 2: Scatter Plot (Blue dots, size = rain intensity)
    - Layer 3: Vector Polygons (Red outlines)
    """
    # 1. Mask & Flatten Data for Scatter
    # We flatten the arrays because scatter() expects 1D lists, not 2D grids
    flat_lats = np.repeat(lats, len(lons)) if lats.ndim == 1 else lats.flatten()
    # Create meshgrid if lats/lons are 1D arrays
    if lats.ndim == 1 and lons.ndim == 1:
        xx, yy = np.meshgrid(lons, lats)
        flat_lons = xx.flatten()
        flat_lats = yy.flatten()
        flat_data = data.flatten()
    else:
        flat_lons = lons.flatten()
        flat_lats = lats.flatten()
        flat_data = data.flatten()

    # Filter: Only show rain > 0.1 mm/hr to keep plot clean
    mask = flat_data > 0.1
    x_masked = flat_lons[mask]
    y_masked = flat_lats[mask]
    z_masked = flat_data[mask]

    # 2. Setup Figure with Map Projection
    fig = plt.figure(figsize=(12, 10))
    tiler = cimgt.GoogleTiles(style='satellite')
    
    # Use Mercator for the map, but PlateCarree for plotting data
    ax = plt.axes(projection=tiler.crs)
    ax.set_extent([bounds['left'], bounds['right'], bounds['bottom'], bounds['top']], crs=ccrs.PlateCarree())

    # 3. Add Map Tiles
    try:
        ax.add_image(tiler, 9) # Zoom level 9
    except Exception:
        print("Warning: Could not fetch map tiles")
        ax.coastlines(color='white')

    # 4. PLOT SCATTER (The Raw Data)
    # s = size (scaled by rain value * 5 for visibility)
    scatter = ax.scatter(
        x_masked, y_masked, 
        s=z_masked * 5, 
        c='cyan', alpha=0.6, edgecolors='none',
        transform=ccrs.PlateCarree(),
        label='Raw Rain Data'
    )

    # 5. PLOT VECTORS (The Polygons)
    if polygons:
        patches_list = []
        for poly_coords in polygons:
            # poly_coords is list of [x, y]
            # Create a Patch
            polygon = patches.Polygon(
                poly_coords, 
                closed=True, 
                fill=False, 
                edgecolor='red', 
                linewidth=2,
                transform=ccrs.PlateCarree() # CRITICAL: Aligns vectors to map
            )
            ax.add_patch(polygon)
            
        # Add a dummy patch for legend
        ax.plot([], [], color='red', linewidth=2, label='Generated Vectors')

    # 6. Decorators
    # Colorbar
    cbar = plt.colorbar(scatter, ax=ax, orientation='horizontal', pad=0.05, fraction=0.04)
    cbar.set_label('Precipitation (mm/hr)', size=10, color='white')
    cbar.ax.xaxis.set_tick_params(color='white', labelcolor='white')

    # Gridlines
    gl = ax.gridlines(draw_labels=True, linestyle=':', alpha=0.5, color='yellow')
    gl.top_labels = False
    gl.right_labels = False

    # Title
    ax.set_title(
        f"DEBUG VIEW\nBlue = Raw Data | Red = Vector Output", 
        fontsize=14, fontweight='bold', pad=12, color='black', backgroundcolor='white'
    )
    ax.legend(loc='upper right')

    # 7. Save
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', dpi=100)
    buf.seek(0)
    plt.close(fig)
    return buf.getvalue()