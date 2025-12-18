import struct
from fastapi import APIRouter, Query, Response, HTTPException
from app.services import gpm_service
from app.utils import plotting, formatting

router = APIRouter()

@router.get("/files")
async def list_files():
    return gpm_service.list_available_files()

@router.get("/plot")
async def plot_gpm_file(
    filename: str = Query(...),
    toplat: float = Query(...),
    bottomlat: float = Query(...),
    leftlon: float = Query(...),
    rightlon: float = Query(...),
):
    bounds = {'top': toplat, 'bottom': bottomlat, 'left': leftlon, 'right': rightlon}

    try:
        # Service: Get Data
        lats, lons, data = gpm_service.process_local_file(filename, bounds)
        
        # Utils: Plot Data
        date_clean = formatting.parse_gpm_filename(filename)
        img_bytes = plotting.generate_heatmap(
            lats, lons, data, bounds, 
            "GPM IMERG", date_clean
        )
        return Response(content=img_bytes, media_type="image/png")

    except Exception as e:
        import traceback
        traceback.print_exc()
        return Response(status_code=500, content=str(e), media_type="text/plain")
    
@router.get("/data")
async def get_gpm_data(
    filename: str = Query(...),
    toplat: float = Query(...),
    bottomlat: float = Query(...),
    leftlon: float = Query(...),
    rightlon: float = Query(...),
    threshold: float = Query(0.1, description="Minimum mm/hr to include"),
    format: str = Query("json", enum=["json", "bin"], description="Response format")
):
    """
    Get 3D Cloud Data.
    - format='json': Returns easy-to-read JSON { lats: [], lons: [], vals: [] }
    - format='bin': Returns compact binary [Header 8B][Lats][Lons][Vals]
    """
    bounds = {'top': toplat, 'bottom': bottomlat, 'left': leftlon, 'right': rightlon}

    try:
        # 1. Get Raw Arrays from Service
        lats, lons, vals, max_val = gpm_service._extract_cloud_arrays(filename, bounds, threshold)
        count = len(vals)

        # --- OPTION A: BINARY ---
        if format == "bin":
            # Header: Count (Uint32, 4 bytes) + MaxVal (Float32, 4 bytes)
            header = struct.pack('<If', count, max_val)
            # Body: Concatenated Float32 Arrays
            body = lats.tobytes() + lons.tobytes() + vals.tobytes()
            
            return Response(content=header + body, media_type="application/octet-stream")

        # --- OPTION B: JSON ---
        else:
            import numpy as np
            return {
                "meta": {
                    "count": count,
                    "max_val": round(max_val, 2),
                    "bounds": bounds
                },
                # Rounding to 3 decimals saves huge bandwidth in JSON
                "lats": np.round(lats, 3).tolist(),
                "lons": np.round(lons, 3).tolist(),
                "vals": np.round(vals, 2).tolist()
            }

    except Exception as e:
        import traceback
        traceback.print_exc()
        return Response(status_code=500, content=str(e), media_type="text/plain")