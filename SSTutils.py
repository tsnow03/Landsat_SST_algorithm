#!/usr/bin/env python
# coding: utf-8

# # Landsat SST utility functions
# Author: Tasha Snow

# Note: After making changes to the `.ipynb` version of SSTutils, got to Terminal, `jupyter nbconvert --to script SSTutils.ipynb`,  make executable in terminal with `chmod +x SSTutils.py`, and rerun `imports` cell.

# In[ ]:


import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from mpl_toolkits.mplot3d import Axes3D
from skimage import exposure
from skimage.io import imsave, imread
from osgeo import ogr
import pystac_client
from pyproj import Transformer
from datetime import date, timedelta, datetime
from dateutil.relativedelta import relativedelta
import geopandas as gpd
import pandas as pd
import geoviews as gv
import hvplot.pandas
import intake
import xarray as xr
import numpy as np
from numpy.random import default_rng
import intake
from pyproj import Proj, transform
from osgeo import gdal
from sklearn.neighbors import BallTree
import earthaccess
import gzip

# for progress bar
from ipywidgets import IntProgress
from IPython.display import display
from ipywidgets import interact, Dropdown
import time
from tqdm.notebook import trange, tqdm

import boto3
import rasterio as rio
from rasterio.features import rasterize
from rasterio.session import AWSSession
import dask
import os
import rioxarray
from rasterio.enums import Resampling
from rasterio.warp import reproject
from rasterio.warp import Resampling as resample
import cartopy.crs as ccrs
import cartopy
from pykrige.ok import OrdinaryKriging
from sklearn.linear_model import LinearRegression, RANSACRegressor
from scipy.odr import Model, RealData, ODR
import scipy.odr as odr
import scipy
import statsmodels.formula.api as smf
from shapely.geometry.polygon import Polygon, Point
import pygmt
import gc
import pytz
import pyproj
import math
from pathlib import Path
from matplotlib.patches import Polygon as Pgon
from tqdm import tqdm
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import StandardScaler


# In[26]:


# Functions to search and open Lansat scenes
'''
Functions to search, open, and analyze Landsat scenes.
Search_stac finds the Landsat scene based on user parameters, 
plot_search plots the locations of the landsat scenes from the search,
landsat_to_xarray takes one of those scenes and puts all bands into an xarray,
and create_masks produces cloud/ice/water masks for the scene. Subset_img 
subsets a landsat scene with coordinates that have been reprojected from lat/lon
and may be flipped in which is larger in the pair. Lsat_reproj can be used to reproject
while ensuring x and y pairs don't get flipped (common converting between espg 3031 and wgs84.
'''

def landsat_to_xarray(sceneid, catalog, bandNames=None):
    """
    Loads selected Landsat bands (and QA layers for later cloud masking) from an 
    AWS S3 bucket (via the STAC item's alternate href) into an xarray Dataset.

    Parameters
    ----------
    sceneid : intake.STAC item
        A single STAC item pointing to Landsat assets.
    catalog : ?????
    bandNames : list of str, optional
        Names of bands to load (e.g., ['red', 'swir16']). If None, all non-thermal
        bands are included by default.

    Returns
    -------
    xr.DataArray
        A 3D xarray DataArray (dimensions: band, y, x) with a scalar coordinate
        for the observation time. The bands will include 'qa_pixel', 'qa_radsat',
        and 'VZA' in addition to any requested reflectance/thermal bands.
    
    Notes
    -------
    If multiple scenes are merged later on, xarray will fill non-overlapping areas with NaNs.
    
    """

    # Retrieve the STAC item from the catalog by its ID
    item = catalog[sceneid.id]

    bands = []
    band_names = []

    if bandNames is None:
        # Get band names
        for k in item.keys():
            M = getattr(item, k).metadata
            if 'eo:bands' in M:
                resol = M['eo:bands'][0]['gsd']
                if resol >= 30: # thermal bands are up sampled from 100 to 30
                    band_names.append(k)
    else:
        band_names = bandNames

    # Add QA bands for creating cloud mask later
    if 'qa_pixel' not in band_names:
        band_names.append('qa_pixel')
    
    band_names.append('VZA')
    band_names.append('qa_radsat')

    # Construct xarray for scene by concatenating all desired bands (including QA)
    for band_name in band_names:
        asset = sceneid.assets[band_name]
        href = asset.extra_fields['alternate']['s3']['href']
        band = xr.open_dataset(href, engine='rasterio', chunks=dict(band=1, x=512, y=512))
        band['band'] = [band_name]
        bands.append(band)
    ls_scene = xr.concat(bands, dim='band')
    ls_scene.coords['id'] = sceneid.id
    ls_scene.coords['time'] = item.metadata['datetime'].strftime('%Y-%m-%dT%H:%M:%S')
    ls_scene = ls_scene['band_data']

    return ls_scene

##########################

def create_masks(ls_scene, cloud_mask=True, ice_mask=False, ocean_mask=False):
    """
    Creates cloud, ice, and ocean masks from a Landsat scene QA band. By default, 
    clouds are labeled as 1, ice as 2, ocean as 3, and all other pixels are NaN.

    Parameters
    ----------
    ls_scene : xarray.DataArray
        A Landsat scene loaded with a 'qa_pixel' band (as created by `landsat_to_xarray`).
    cloud_mask : bool, optional
        Whether to generate the cloud mask. Default is True.
    ice_mask : bool, optional
        Whether to generate the ice mask. Default is False.
    ocean_mask : bool, optional
        Whether to generate the ocean mask. Default is False.

    Returns
    -------
    xarray.DataArray
        The same input xarray object, but with an added `"mask"` coordinate. 
        In that mask, cloud pixels are assigned 1, ice pixels 2, ocean pixels 3, 
        and everything else is set to NaN.
    """
    
    cloud = []
    ocean = []
    ice = []

    qa = ls_scene.sel(band='qa_pixel').astype('uint16')

    n,c = np.unique(qa, return_counts=True)

    for j in range(len(n)):
        longform = f'{n[j]:016b}'
        if (longform[-7]=='0')|(longform[-3]=='1'): #bit 2 and 6 are for cirrus and clear sky
            cloud.append(n[j])
        if longform[-8:]=='11000000': #bit 6 and 7 give clear sky and water, lower bits need to be 0 
            ocean.append(n[j])
        if longform[-7:]=='1100000': #bit 5 and 6 give ice and clear sky 
            ice.append(n[j])

    if 0 in cloud:
        cloud.remove(0)
    if 1 in cloud:
        cloud.remove(1)

    # mask cloud, ice, and ocean
    if cloud_mask==True:
        # cloud is 2
        mask_c = xr.where(qa.isin(cloud), 1, np.nan)

    if ice_mask==True:
        mask_c = xr.where(qa.isin(ice), 2, mask_c)

    if ocean_mask==True:
        mask_c = xr.where(qa.isin(ocean), 3, mask_c)

    ls_scene.coords['mask'] = (('y', 'x'), mask_c.data)
        
    return ls_scene

##########################

def normalize(array):
    '''
    normalize a dask array so all value are between 0 and 1
    '''
    array_min = array.min(skipna=True)
    array_max = array.max(skipna=True)
    return (array - array_min) / (array_max - array_min)

##########################

def search_stac(url, collection, gjson_outfile=None, bbox=None, timeRange=None, filename=None):
    """
    Search a STAC API for Landsat images based on either:
    - Bounding box and time range, or
    - Specific filename (STAC 'id').

    Parameters:
    -----------
    url : str
        URL to the STAC API.
    collection : str
        Collection name (e.g., "landsat-c2-l2").
    gjson_outfile : str or None
        Output file to save the search result as GeoJSON (optional).
    bbox : list or None
        Bounding box [west, south, east, north] (optional).
    timeRange : str or None
        Time range in ISO format, e.g., '2021-09-01/2023-03-31' (optional).
    filename : str or None
        Exact filename (product ID) to search for (optional).

    Returns:
    --------
    item_collection : pystac.ItemCollection
        Collection of matching STAC items.
    """
    
    api = pystac_client.Client.open(url)

    if filename:
        # Search by filename (ID)
        search = api.search(
            collections=[collection],
            ids=[filename],
        )
        # print(f"Searching for filename: {filename}")
    
    elif bbox and timeRange:
        # Search by bbox and timeRange
        search = api.search(
            bbox=bbox,
            datetime=timeRange,
            collections=[collection],
        )
        # print(f"Searching for items in bbox {bbox} and timeRange {timeRange}")
    
    else:
        raise ValueError("Must provide either a filename, or both bbox and timeRange.")

    items = search.item_collection()

    # print(f"Found {len(items)} item(s)")

    if gjson_outfile:
        items.save_object(gjson_outfile)
    
    return items

###############

def get_lst_mask(lstfile):
    """
    Generates an open ocean mask from a Landsat scene based on the QA band information.

    This function searches for a Landsat scene using a provided filename, loads the 
    'qa_pixel' band, applies cloud, ice, and ocean masking, and then extracts only 
    the open ocean pixels. The output is a mask where open ocean pixels are 1, and 
    all other pixels are NaN.

    Parameters
    ----------
    lstfile : str
        Path or name of the Landsat file used to derive the corresponding STAC search ID.

    Returns
    -------
    numpy.ndarray
        A 2D mask array where open ocean pixels are 1, and all other pixels are NaN.
    """
    filename = lstfile[:-11]
    items = search_stac(url,collection,filename=filename)
    
    # Open stac catalog for some needed info
    catalog = intake.open_stac_item_collection(items)
    sceneid = items[0]
    print(sceneid.id)
    
    scene = catalog[sceneid.id]
    
    # Open all desired bands for one scene
    ls_scene0 = landsat_to_xarray(sceneid,catalog,bandNames=['qa_pixel'])
    ls_scene0 = ls_scene0.rio.write_crs("epsg:3031", inplace=True)
    
    # Create a classification mask, applying cloud, ice, and ocean masks
    ls_scene0 = create_masks(ls_scene0, cloud_mask=True, ice_mask=True, ocean_mask=True)
    
    # Initialize a mask array and set all pixels not classified as open ocean (mask != 3) to NaN
    mask = np.ones(ls_scene0.shape[1:])
    mask[ls_scene0.mask!=3] = np.nan

    try:
        del ls_scene0
    except:
        pass
    
    gc.collect()

    return mask

##########################

def plot_search(gf,satellite,colnm):
    # Plot search AOI and frames on a map using Holoviz Libraries (more on these later)
    cols = gf.loc[:,('id',colnm[0],colnm[1],'geometry')]
    alpha = 1/gf.shape[0]**0.5 # transparency scales w number of images

    footprints = cols.hvplot(geo=True, line_color='k', hover_cols=[colnm[0],colnm[1]], alpha=alpha, title=satellite,tiles='ESRI')
    tiles = gv.tile_sources.CartoEco.options(width=700, height=500) 
    labels = gv.tile_sources.StamenLabels.options(level='annotation')
    tiles * footprints * labels
    
    return footprints

##########################

def subset_img(da,polarx,polary):
    '''
    ***Only works for square grid cropping along the orientation of the grid (not when cropping along lat/lon in a 3031 grid
    
    Subset image in xarray to desired coordinates. Because Landsat polar stereo projection can be oriented
    in many different directions, when coordinates to subset an image are reprojected from lat/lon they may get 
    flipped for which is larger in the pair. This function checks to make sure we are getting a proper subset and 
    avoids 0 pixels on the x or y axis. 
    
    Note: Input shape dimensions and dataarray v. dataset changes things so input needs to be a dataarray w 
          2 dimensions (x,y)
    
    Input:
    da = xarray DataArray to be subset
    polarx = x coordinates to subset by in polar stereographic projection
    polary = y coordinates to subset by in polar stereographic projection
    
    Output:
    ls_sub = subset xarray DataArray
    
    '''
    # ***Landsat shape dimensions are one fewer than they are for LandsatCalibration [0,1] not [1,2], no .to_array() or Band
    ls_sub = da.sel(y=slice(polary[1],polary[0]),x=slice(polarx[0],polarx[1]))

    # Check for right dimensions because y order changes sometimes
    if (ls_sub.x.shape[0]==0) & (ls_sub.y.shape[0]==0):
        # print ('L8 x and y shapes are 0')
        ls_sub = da.sel(y=slice(polary[0],polary[1]),x=slice(polarx[1],polarx[0]))
    elif ls_sub.y.shape[0]==0:
        # print ('L8 y shape is 0')
        ls_sub = da.sel(y=slice(polary[0],polary[1]),x=slice(polarx[0],polarx[1]))
    elif ls_sub.x.shape[0]==0:
        # print ('L8 x shape is 0')
        ls_sub = da.sel(y=slice(polary[1],polary[0]),x=slice(polarx[1],polarx[0]))
    # print(ls_sub.shape)
    
    return ls_sub

##########################

def lsat_reproj(old_cs,new_cs,lbox):
    '''
    Reprojects a bounding box from an old coordinate system to a new one, and checks
    for round-trip transformation errors. The resulting bounding box coordinate order
    may be flipped if the input coordinates indicate an inverted orientation. 
    Diagnostic information is printed, and the transformed bounding box is returned.

    Parameters
    ----------
    old_cs : str
        The Proj4 or EPSG string for the original (source) coordinate system.
    new_cs : str
        The Proj4 or EPSG string for the target coordinate system.
    lbox : list or tuple of float
        A bounding box specified as [ULX, LRY, LRX, ULY] in the old coordinate system.

    Returns
    -------
    bbox : list of tuples
        The transformed bounding box in the new coordinate system.
    checkbox : numpy.ndarray
        An array of the round-trip check coordinates.
    '''
    
    test_threshold = 0.5
    
    # Create a transform object to convert between coordinate systems
    inProj = Proj(init=old_cs)
    outProj = Proj(init=new_cs)
    
    ULX,LRY,LRX,ULY = lbox
    
    # Check if bounding box likely crosses or is near the IDL
    crosses_idl = abs(ULX - LRX) > 180 or abs(ULX) > 170 or abs(LRX) > 170

    [lULX,lLRX], [lULY,lLRY] =  transform(inProj,outProj,[ULX,LRX], [ULY,LRY], always_xy=True)
    [cULX,cLRX], [cULY,cLRY] =  transform(outProj,inProj,[lULX,lLRX], [lULY,lLRY], always_xy=True)
    [lLLX,lURX], [lLLY,lURY] =  transform(inProj,outProj,[ULX,LRX], [LRY,ULY], always_xy=True)
    [cLLX,cURX], [cLLY,cURY] =  transform(outProj,inProj,[lLLX,lURX], [lLLY,lURY], always_xy=True)

    if LRY>ULY:
        bbox = [(lULX,lLLY),(lLLX,lULY),(lLRX,lURY),(lURX,lLRY)]
    else:
        bbox = [(lULX,lULY),(lLLX,lLLY),(lLRX,lLRY),(lURX,lURY)]

    checkbox = np.array([cULX,cULY,cLRX,cLRY])
    if not crosses_idl and np.linalg.norm(checkbox - np.array([ULX,ULY,LRX,LRY])) > test_threshold:
        print(f"Round-trip transformation error 1 of {np.linalg.norm(checkbox - np.array([ULX,ULY,LRX,LRY]))}")
    
    checkbox = np.array([cLLX,cLLY,cURX,cURY])
    if not crosses_idl and np.linalg.norm(checkbox - np.array([ULX,LRY,LRX,ULY])) > test_threshold:
        print(f"Round-trip transformation error 2 of {np.linalg.norm(checkbox - np.array([ULX,LRY,LRX,ULY]))}")
    
    return bbox,checkbox
        
##########################

def sub_to_point(row, lat_nm, lon_nm, dist, source_crs, target_crs, SST_calibrated):
    # Create search area around THIS specific seal point (in lat/lon)
    ilat = row[lat_nm]
    ilon = row[lon_nm]
    lat_add = km_to_decimal_degrees(dist, ilat, direction='latitude')
    lon_add = km_to_decimal_degrees(dist, ilat, direction='longitude')
    
    # Create bounding box in lat/lon
    bboxV = (ilon-lon_add, ilat-lat_add, ilon+lon_add, ilat+lat_add)
    
    # Reproject bounding box to EPSG:3031
    sbox, checkbox = lsat_reproj(source_crs, target_crs, 
                                      (bboxV[0], bboxV[1], bboxV[2], bboxV[3]))
    
    # Create polygon for cropping
    polygon = Polygon([
        (sbox[0][0], sbox[0][1]), 
        (sbox[3][0], sbox[3][1]), 
        (sbox[2][0], sbox[2][1]), 
        (sbox[1][0], sbox[1][1]),
        (sbox[0][0], sbox[0][1])  
    ])
    
    # Get min/max boundaries
    minx, miny, maxx, maxy = polygon.bounds
    polarx = [minx, maxx]
    polary = [miny, maxy]
    
    # Subset to the small area around this seal point
    SST_sub = subset_img(SST_calibrated, polarx, polary)
    
    # Crop to exact polygon
    SST_sub = crop_xarray_dataarray_with_polygon(SST_sub, polygon)
    
    # Now compute statistics for THIS specific area
    lsat = np.around(np.nanmean(SST_sub), 2)
    lstd = np.around(np.nanstd(SST_sub), 2)

    return lsat, lstd, SST_sub
        
##########################

def center_pix(SST_sub, x_pt, y_pt):
    # Get center pixel value
    center_val = SST_sub.sel(x=x_pt, y=y_pt, method="nearest")
    
    # Extract the x/y coordinate values as plain floats
    center_x = float(center_val.x.values)
    center_y = float(center_val.y.values)
    
    # Extract the actual value
    if isinstance(center_val, xr.Dataset):
        center_value = float(center_val['band_data'].values.item())
    else:
        center_value = float(center_val.values.item())
        
##########################

def crop_xarray_dataarray_with_polygon(dataarray, polygon):
    """
    Crop an xarray.DataArray using a polygon.
    
    Parameters:
    - dataarray: xarray.DataArray with x and y coordinates.
    - polygon: Shapely Polygon object defining the crop area.
    
    Returns:
    - Cropped xarray.DataArray.
    """
    # Generate a 2D array of shapely Point objects for each grid point
    lon, lat = np.meshgrid(dataarray.x.values, dataarray.y.values)
    points = np.vectorize(Point)(lon, lat)
    
    # Create a mask where points within the polygon are True
    mask_func = np.vectorize(polygon.contains)
    mask = mask_func(points)
    
    # Convert the mask to an xarray.DataArray
    mask_da = xr.DataArray(mask, dims=["y", "x"], coords={"y": dataarray.y, "x": dataarray.x})
    
    # Apply the mask to the dataarray, cropping to the polygon
    # Use where method with drop=True to drop values outside the polygon
    cropped_dataarray = dataarray.where(mask_da, drop=True)
    
    return cropped_dataarray

##########################

def km_to_decimal_degrees(km, latitude, direction='latitude'):
    """
    Convert a distance in kilometers to decimal degrees of latitude or longitude,
    given a specific latitude.

    Parameters
    ----------
    km : float
        The distance in kilometers to be converted.
    latitude : float
        The latitude (in decimal degrees, from -90 to +90) where the conversion
        is being applied. Used only if direction='longitude'.
    direction : str, optional
        Either 'latitude' or 'longitude'. Determines whether to convert
        km to decimal degrees of latitude or longitude. Default is 'latitude'.

    Returns
    -------
    float
        The approximate decimal degrees that correspond to the given distance in km
        at the specified latitude (for longitude) or globally (for latitude).

    Notes
    -----
    1° latitude ~ 111.32 km everywhere on Earth.
    1° longitude ~ 111.32 km * cos(latitude), which is why the
        conversion depends on the specified latitude for 'longitude'.
    This function uses a spherical Earth approximation and is not exact
    at very high latitudes or for large distances.

    Examples
    --------
    >>> # Convert 10 km to decimal degrees of latitude (anywhere)
    >>> km_to_decimal_degrees(10, latitude=0, direction='latitude')
    0.0898...

    >>> # Convert 10 km to decimal degrees of longitude at latitude 69°S
    >>> km_to_decimal_degrees(10, latitude=-69, direction='longitude')
    0.2515...
    """
    if direction.lower() == 'latitude':
        # 1 degree of latitude ≈ 111.32 km (on average)
        deg = km / 111.32
    elif direction.lower() == 'longitude':
        # 1 degree of longitude ≈ 111.32 km * cos(lat)
        deg = km / (111.32 * math.cos(math.radians(latitude)))
    else:
        raise ValueError("direction must be 'latitude' or 'longitude'")
    return deg

##########################

def crosses_idl(coords):
    '''
    Determine if the set of coordinates crosses the International Dateline
    
    Variables:
    coords = list of lon, lat tuples
    
    Output:
    True or False
    '''
    
    for (lon1, lat1), (lon2, lat2) in zip(coords, coords[1:]):
        if abs(lon1 - lon2) >= 180:
            return True
    return False

##########################

def split_polygon_at_idl(coords):
    '''
    Split a polygon that crosses the IDL into two polygons
    
    Variables:
    coords = list of lon, lat tuples
    
    Output:
    Two Polygon objects (western and eastern hemispheres)
    '''
    from shapely.geometry import Polygon, MultiPolygon
    
    # Shift all longitudes to 0-360 range
    shifted_coords = [(lon + 360 if lon < 0 else lon, lat) for lon, lat in coords]
    
    try:
        shifted_poly = Polygon(shifted_coords)
        return shifted_poly, None
    except:
        return None, None

##########################
    
def check_overlap_with_idl_handling(lsatpoly, modis_coords):
    '''
    Check overlap between Landsat and MODIS polygons, handling IDL crossing
    
    Variables:
    lsatpoly = Shapely Polygon for Landsat scene
    modis_coords = list of lon, lat tuples for MODIS granule
    
    Output:
    percent_dif = fraction of Landsat polygon not covered by MODIS
    '''
    
    # Check if MODIS polygon crosses IDL
    if crosses_idl(modis_coords):
        # Try shifted coordinate system (0-360)
        shifted_modis = [(lon + 360 if lon < 0 else lon, lat) for lon, lat in modis_coords]
        
        # Get Landsat bounds
        lsat_bounds = lsatpoly.bounds  # (minx, miny, maxx, maxy)
        
        # Check if Landsat is near IDL
        if lsat_bounds[0] < -170 or lsat_bounds[2] > 170:
            # Shift Landsat polygon too
            lsat_coords = list(lsatpoly.exterior.coords)
            shifted_lsat = [(lon + 360 if lon < 0 else lon, lat) for lon, lat in lsat_coords]
            try:
                shifted_lsatpoly = Polygon(shifted_lsat)
                shifted_modis_poly = Polygon(shifted_modis)
                percent_dif = shifted_lsatpoly.difference(shifted_modis_poly).area / shifted_lsatpoly.area
                return percent_dif
            except:
                return 1.0  # Failed to create polygons
        else:
            # Landsat not near IDL, so this shouldn't match
            return 1.0
    else:
        # Normal case - no IDL crossing
        try:
            pgon = Polygon(modis_coords)
            percent_dif = lsatpoly.difference(pgon).area / lsatpoly.area
            return percent_dif
        except:
            return 1.0

##########################

def plot_geotiff(filepath):
    # Open the geotiff file
    with rio.open(filepath) as src:
        # Reproject the dataset to lat/lon
        transform, width, height = rio.warp.calculate_default_transform(
            src.crs, 'EPSG:4326', src.width, src.height, *src.bounds)
        kwargs = src.meta.copy()
        kwargs.update({
            'crs': 'EPSG:4326',
            'transform': transform,
            'width': width,
            'height': height
        })

        # Read the data and reproject
        with rio.MemoryFile() as memfile:
            with memfile.open(**kwargs) as dst:
                rio.warp.reproject(
                    source=rio.band(src, 1),
                    destination=rio.band(dst, 1),
                    src_transform=src.transform,
                    src_crs=src.crs,
                    dst_transform=transform,
                    dst_crs='EPSG:4326',
                    resampling=rio.enums.Resampling.nearest
                )
                data = dst.read(1)

    # Plot the data
    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw={'projection': ccrs.PlateCarree()})
    ax.set_global()
    ax.add_feature(cfeature.COASTLINE)
    ax.gridlines(draw_labels=True, dms=True, x_inline=False, y_inline=False)
    ax.imshow(data, transform=ccrs.PlateCarree(), origin='upper', extent=dst.bounds, cmap='viridis')
    ax.set_title(os.path.basename(filepath))
    plt.show()


##########################

def create_geotiff_dropdown(directory):
    # Create a dropdown widget with all GeoTIFF files in the directory
    tif_files = [f for f in os.listdir(directory) if f.endswith('.tif')]
    dropdown = Dropdown(options=tif_files, description='Select a file:')
    
    # Update function to plot based on the selected file
    def update_plot(selected_file):
        plot_geotiff(os.path.join(directory, selected_file))
    
    interact(update_plot, selected_file=dropdown)

##########################

# Preprocess to add time dimension and the file name to open_mfdataset for landsat using the filename
def add_time_dim(ds):
    lstr = ds.encoding["source"].split("LC0",1)[1]
    times = pd.to_datetime(lstr[14:22]+lstr[38:44], format='%Y%m%d%H%M%S')
    idee = ds.encoding["source"].split("/")[8][:-4] # The first number depends on how many subdirectories the file is in
    return ds.assign_coords(time=times,ID=idee)


# In[39]:


# Atmospheric correction and production of SST
'''
Functions to find the matching MODIS water vapor image for atmospheric correction and production of SST.
Open_MODIS finds and downloads the closest MODIS water vapor image to a specific landsat image. Get_wv
aligns and subsets the modis image grid to landsat using MODISlookup and subsamples and extracts the data 
onto the Landsat grid using uniqueMODIS
'''

def open_MODIS(ls_scene, scene, modout_path):
    '''
    Search MOD/MDY07 atmospheric data and open water vapor for data collected closest in time to 
    Landsat scene. Handles scenes that cross the International Date Line.
    Tries multiple MODIS granules until a valid one is found.
    
    Input:
    ls_scene = xarray dataset with Landsat scene
    modout_path = directory path for MODIS data
    scene = STAC catalog item
    
    Output:
    mod07 = xarray dataset with MODIS (MOD/MDY07) water vapor 
    modfilenm = MODIS filename for image used in atm correction
    '''

    # Get spatial extent of Landsat scene in lat/lon
    west, south, east, north = scene.metadata['bbox']
    
    # Check if bbox crosses IDL
    crosses_idl_flag = (west > east) or (west > 170) or (east < -170)
    
    if crosses_idl_flag:
        # Split into TWO bounding boxes - one on each side of the IDL
        bbox_west = (west, south, 180.0, north)
        bbox_east = (-180.0, south, east, north)
        
        search_bboxes = [bbox_west, bbox_east]
        
        # For polygon overlap checks, use shifted coordinates (0-360)
        west_shifted = west if west >= 0 else west + 360
        east_shifted = east if east >= 0 else east + 360
        lsatpoly_shifted = Polygon([
            (west_shifted, south),
            (west_shifted, north),
            (east_shifted, north),
            (east_shifted, south),
            (west_shifted, south)
        ])
        lsatpoly = lsatpoly_shifted
        lsatpoly_360 = lsatpoly  # Already in 0-360
        
    else:
        # Normal case - no IDL crossing
        mbbox = (west, south, east, north)
        search_bboxes = [mbbox]
        
        # Create polygon in -180/180
        lsatpoly = Polygon([
            (west, south),
            (west, north),
            (east, north),
            (east, south),
            (west, south)
        ])
        
        # Also create a 0-360 version for when MODIS crosses the IDL
        west_shifted = west if west >= 0 else west + 360
        east_shifted = east if east >= 0 else east + 360
        lsatpoly_360 = Polygon([
            (west_shifted, south),
            (west_shifted, north),
            (east_shifted, north),
            (east_shifted, south),
            (west_shifted, south)
        ])

    ls_time = pd.to_datetime(ls_scene.time.values)
    calc_dt = datetime.strptime(ls_time.strftime('%Y-%m-%d %H:%M:%S'), '%Y-%m-%d %H:%M:%S')
    start_dt = (calc_dt + timedelta(days=-0.5)).strftime('%Y-%m-%d %H:%M:%S')
    end_dt = (calc_dt + timedelta(days=0.5)).strftime('%Y-%m-%d %H:%M:%S')
    
    # Search for MODIS data using all bounding boxes
    results = []
    for bbox in search_bboxes:
        try:
            res_mod = earthaccess.search_data(
                short_name='MOD07_L2',
                bounding_box=bbox,
                temporal=(start_dt, end_dt)
            )
            res_myd = earthaccess.search_data(
                short_name='MYD07_L2',
                bounding_box=bbox,
                temporal=(start_dt, end_dt)
            )
            results.extend(res_mod)
            results.extend(res_myd)
        except Exception as e:
            print(f"Error searching with bbox {bbox}: {e}")
            continue
    
    print(f'{len(results)} TOTAL granules')

    # Accept only granules that overlap at least 100% with Landsat
    best_grans = []
    for granule in results:
        try:
            granule['umm']['SpatialExtent']['HorizontalSpatialDomain']['Geometry']['GPolygons']
        except Exception as error:
            print(error)
            continue
            
        for num in range(len(granule['umm']['SpatialExtent']['HorizontalSpatialDomain']['Geometry']['GPolygons'])):
            try:
                map_points = [(xi['Longitude'], xi['Latitude']) for xi in 
                             granule['umm']['SpatialExtent']['HorizontalSpatialDomain']['Geometry']['GPolygons'][num]['Boundary']['Points']]
                
                # Check if this MODIS granule crosses the IDL
                modis_lons = [p[0] for p in map_points]
                modis_lon_span = max(modis_lons) - min(modis_lons)
                modis_crosses_idl = modis_lon_span > 180
                
                # Use the appropriate Landsat polygon
                if modis_crosses_idl:
                    # MODIS crosses IDL - use 0-360 coordinate system for both
                    map_points_shifted = [(lon if lon >= 0 else lon + 360, lat) for lon, lat in map_points]
                    
                    try:
                        modis_poly = Polygon(map_points_shifted)
                        percent_dif = lsatpoly_360.difference(modis_poly).area / lsatpoly_360.area
                    except Exception as e:
                        percent_dif = 1.0
                else:
                    # MODIS doesn't cross IDL - use normal -180/180 coordinates
                    try:
                        modis_poly = Polygon(map_points)
                        percent_dif = lsatpoly.difference(modis_poly).area / lsatpoly.area
                    except Exception as e:
                        percent_dif = 1.0
                
                if percent_dif == 0.0:
                    best_grans.append(granule)
                    continue
                    
            except Exception as error:
                print(f'Error processing granule polygon: {error}')
    
    # Remove duplicates (since we might search same granule from both boxes)
    unique_grans = []
    seen_ids = set()
    duplicates_removed = []
    
    for gran in best_grans:
        gran_id = gran['umm']['GranuleUR']
        if gran_id not in seen_ids:
            unique_grans.append(gran)
            seen_ids.add(gran_id)
        else:
            # Get filename of duplicates
            modis_filename = "Unknown"
            try:
                for url_info in gran['umm']['RelatedUrls']:
                    if 'URL' in url_info and '.hdf' in url_info['URL']:
                        modis_filename = url_info['URL'].split('/')[-1]
                        break
            except:
                modis_filename = gran_id
            
            duplicates_removed.append(modis_filename)
    
    best_grans = unique_grans
    print(f'{len(best_grans)} TOTAL granules w overlap (after deduplication)')
    
    if len(best_grans) == 0:
        raise ValueError(f"No MODIS granules found that overlap with Landsat scene. Scene bbox: {scene.metadata['bbox']}")

    # Sort granules by time difference (closest first)
    Mdates = [pd.to_datetime(granule['umm']['TemporalExtent']['RangeDateTime']['BeginningDateTime']) 
              for granule in best_grans]
    time_diffs = [abs(d - pytz.utc.localize(pd.to_datetime(ls_time))) for d in Mdates]
    sorted_indices = sorted(range(len(time_diffs)), key=lambda i: time_diffs[i])
    
    # Try up to 5 closest MODIS granules until find a valid one
    max_attempts = min(5, len(best_grans))
    
    for attempt in range(max_attempts):
        ind = sorted_indices[attempt]
        time_diff = time_diffs[ind]
        
        print(f'\nAttempt {attempt+1}/{max_attempts}: Time difference = {time_diff}')
        
        try:
            # Download MODIS data
            data_links = [granule.data_links(access="external") for granule in best_grans[ind:ind+1]]
            netcdf_list = [g._filter_related_links("USE SERVICE API")[0].replace(".html", ".nc4") 
                           for g in best_grans[ind:ind+1]]
            file_handlers = earthaccess.download(netcdf_list, modout_path, provider='NSIDC')

            # Open MODIS data
            mod_list = os.listdir(modout_path)
            mod_list = [file for file in mod_list if file[-3:]=='nc4']
            
            if len(mod_list) == 0:
                print(f"  ✗ No nc4 file downloaded, skipping...")
                continue
                
            modfilenm = mod_list[0]
            
            # Decompress
            os.rename(f'{modout_path}/{modfilenm}', f'{modout_path}/{modfilenm}.gz')
            with gzip.open(f'{modout_path}/{modfilenm}.gz', 'rb') as f_in:
                with open(f'{modout_path}/{modfilenm}', 'wb') as f_out:
                    f_out.write(f_in.read())

            # Open and validate
            mod07 = xr.open_dataset(f'{modout_path}/{modfilenm}')
            mod07 = mod07.rio.write_crs('epsg:4326')
            
            # Get MODIS data
            modis_lon_vals = mod07.Longitude.values
            modis_lat_vals = mod07.Latitude.values
            wv_data = mod07['Water_Vapor'].values
            valid_mask = np.isfinite(wv_data) & (wv_data > 0)
            
            valid_percent = 100 * np.sum(valid_mask) / valid_mask.size
            
            if np.sum(valid_mask) == 0:
                print(f"  ✗ MODIS granule has no valid data")
                os.remove(f'{modout_path}/{modfilenm}')
                os.remove(f'{modout_path}/{modfilenm}.gz')
                continue
            
            # Get Landsat bounds
            ls_lon_min, ls_lat_min, ls_lon_max, ls_lat_max = scene.metadata['bbox']
            ls_crosses_idl = (ls_lon_max < ls_lon_min) or (ls_lon_max - ls_lon_min > 180)
            
            # Check if MODIS crosses IDL
            modis_lon_span = modis_lon_vals.max() - modis_lon_vals.min()
            modis_crosses_idl = modis_lon_span > 180
            
            # Shift to 0-360 if either crosses IDL
            if ls_crosses_idl or modis_crosses_idl:
                modis_lon_shifted = modis_lon_vals.copy()
                modis_lon_shifted[modis_lon_shifted < 0] += 360
                
                ls_lon_min_shifted = ls_lon_min if ls_lon_min >= 0 else ls_lon_min + 360
                ls_lon_max_shifted = ls_lon_max if ls_lon_max >= 0 else ls_lon_max + 360
            else:
                modis_lon_shifted = modis_lon_vals
                ls_lon_min_shifted = ls_lon_min
                ls_lon_max_shifted = ls_lon_max
            
            # Get MODIS data extent (only valid data)
            modis_lon_min = modis_lon_shifted[valid_mask].min()
            modis_lon_max = modis_lon_shifted[valid_mask].max()
            modis_lat_min = modis_lat_vals[valid_mask].min()
            modis_lat_max = modis_lat_vals[valid_mask].max()
            
            print(f"  MODIS: lon [{modis_lon_min:.2f}, {modis_lon_max:.2f}], lat [{modis_lat_min:.2f}, {modis_lat_max:.2f}], {valid_percent:.1f}% valid")
            print(f"  Landsat: lon [{ls_lon_min_shifted:.2f}, {ls_lon_max_shifted:.2f}], lat [{ls_lat_min:.2f}, {ls_lat_max:.2f}]")
            
            # Check coverage
            buffer = 2.0  # degrees
            lon_covered = (modis_lon_min - buffer <= ls_lon_min_shifted) and (ls_lon_max_shifted <= modis_lon_max + buffer)
            lat_covered = (modis_lat_min - buffer <= ls_lat_min) and (ls_lat_max <= modis_lat_max + buffer)
            
            if not (lon_covered and lat_covered):
                print(f"  ✗ MODIS doesn't cover Landsat scene")
                os.remove(f'{modout_path}/{modfilenm}')
                os.remove(f'{modout_path}/{modfilenm}.gz')
                continue
            
            # Count points within Landsat extent
            ls_mask = (
                (modis_lon_shifted >= ls_lon_min_shifted - buffer) &
                (modis_lon_shifted <= ls_lon_max_shifted + buffer) &
                (modis_lat_vals >= ls_lat_min - buffer) &
                (modis_lat_vals <= ls_lat_max + buffer) &
                valid_mask
            )
            
            points_in_scene = np.sum(ls_mask)
            
            if points_in_scene < 10:
                print(f"  ✗ Only {points_in_scene} MODIS points over Landsat scene")
                os.remove(f'{modout_path}/{modfilenm}')
                os.remove(f'{modout_path}/{modfilenm}.gz')
                continue
            
            print(f"  ✓ Valid! {points_in_scene} MODIS points over Landsat scene")
            
            # Clean up compressed file
            os.remove(f'{modout_path}/{modfilenm}.gz')
            os.remove(f'{modout_path}/{modfilenm}')
            
            return mod07, modfilenm
            
        except Exception as e:
            print(f"  ✗ Error processing granule: {e}")
            # Clean up any files that might exist
            try:
                if 'modfilenm' in locals():
                    if os.path.exists(f'{modout_path}/{modfilenm}'):
                        os.remove(f'{modout_path}/{modfilenm}')
                    if os.path.exists(f'{modout_path}/{modfilenm}.gz'):
                        os.remove(f'{modout_path}/{modfilenm}.gz')
            except:
                pass
            continue
    
    # If we get here, all attempts failed
    raise ValueError(
        f"Could not find valid MODIS granule for Landsat scene after {max_attempts} attempts. "
        f"Scene bbox: {scene.metadata['bbox']}"
    )

    
##########################

def find_MODIS(lonboundsC, latboundsC, ls_scene, buffer=2.0, min_points=10, max_attempts=5):
    """
    Finds the MODIS GHRSST L2P (Terra/Aqua) scene most closely coincident to a Landsat scene.
    Adds IDL handling + multi-granule screening similar to open_MODIS().

    Inputs
    ------
    lonboundsC : (west, east)  (degrees)
    latboundsC : (south, north) (degrees)
    ls_scene   : xarray Dataset for one Landsat scene
    buffer     : degrees padding for coverage check
    min_points : minimum # valid MODIS points within buffered Landsat extent
    max_attempts : try this many closest-in-time candidates until one passes screening

    Returns
    -------
    mod_scene : xarray Dataset of MODIS SST
    mod_granule_ur : granule UR (filename-ish)
    time_dif : timedelta between Landsat time and MODIS start time
    """

    west, east = float(lonboundsC[0]), float(lonboundsC[1])
    south, north = float(latboundsC[0]), float(latboundsC[1])

    # --- IDL detection ---
    crosses_idl_flag = (west > east) or (west > 170) or (east < -170)
    
    if crosses_idl_flag:
        bbox_west = (west, south, 180.0, north)
        bbox_east = (-180.0, south, east, north)
        search_bboxes = [bbox_west, bbox_east]
    
        west_shifted = west if west >= 0 else west + 360
        east_shifted = east if east >= 0 else east + 360
        lsatpoly_shifted = Polygon([
            (west_shifted, south),
            (west_shifted, north),
            (east_shifted, north),
            (east_shifted, south),
            (west_shifted, south)
        ])
        lsatpoly = lsatpoly_shifted
        lsatpoly_360 = lsatpoly  # Already in 0-360
    else:
        search_bboxes = [(west, south, east, north)]
        
        # Create polygon in -180/180
        lsatpoly = Polygon([
            (west, south),
            (west, north),
            (east, north),
            (east, south),
            (west, south)
        ])
        
        # Also create a 0-360 version for when MODIS crosses IDL
        west_shifted = west if west >= 0 else west + 360
        east_shifted = east if east >= 0 else east + 360
        lsatpoly_360 = Polygon([
            (west_shifted, south),
            (west_shifted, north),
            (east_shifted, north),
            (east_shifted, south),
            (west_shifted, south)
        ])

    # --- Time window ---
    ls_time = pd.to_datetime(ls_scene.time.values)
    calc_dt = datetime.strptime(ls_time.strftime('%Y-%m-%d %H:%M:%S'), '%Y-%m-%d %H:%M:%S')
    start_dt = (calc_dt + timedelta(days=-0.5)).strftime('%Y-%m-%d %H:%M:%S')
    end_dt   = (calc_dt + timedelta(days=0.5)).strftime('%Y-%m-%d %H:%M:%S')

    # --- Search Terra + Aqua over all relevant bboxes ---
    results = []
    for bbox in search_bboxes:
        try:
            res_t = earthaccess.search_data(
                short_name='MODIS_T-JPL-L2P-v2019.0',
                bounding_box=bbox,
                temporal=(start_dt, end_dt),
            )
            res_a = earthaccess.search_data(
                short_name='MODIS_A-JPL-L2P-v2019.0',
                bounding_box=bbox,
                temporal=(start_dt, end_dt),
            )
            results.extend(res_t)
            results.extend(res_a)
        except Exception as e:
            print(f"Error searching with bbox {bbox}: {e}")
            continue

    print(f"{len(results)} TOTAL MODIS granules (raw search results)")

    # --- Overlap screening (IDL-aware) ---
    best_grans = []
    for granule in results:
        try:
            gpolys = granule['umm']['SpatialExtent']['HorizontalSpatialDomain']['Geometry']['GPolygons']
        except Exception as e:
            continue
    
        for gp in gpolys:
            try:
                map_points = [(pt['Longitude'], pt['Latitude']) for pt in gp['Boundary']['Points']]
    
                # Check if this MODIS granule crosses the IDL
                modis_lons = [p[0] for p in map_points]
                modis_lon_span = max(modis_lons) - min(modis_lons)
                modis_crosses_idl = modis_lon_span > 180
                
                # Use appropriate Landsat polygon
                if modis_crosses_idl:
                    # MODIS crosses IDL - use 0-360 coordinate system for both
                    map_points_shifted = [(lon if lon >= 0 else lon + 360, lat) for lon, lat in map_points]
                    try:
                        modis_poly = Polygon(map_points_shifted)
                        percent_dif = lsatpoly_360.difference(modis_poly).area / lsatpoly_360.area
                    except Exception:
                        percent_dif = 1.0
                else:
                    # MODIS doesn't cross IDL - use normal -180/180 coordinates
                    try:
                        modis_poly = Polygon(map_points)
                        percent_dif = lsatpoly.difference(modis_poly).area / lsatpoly.area
                    except Exception:
                        percent_dif = 1.0
    
                if percent_dif == 0.0:
                    best_grans.append(granule)
                    break
    
            except Exception as e:
                continue

    # --- Deduplicate (searching two bboxes can return the same granule twice) ---
    unique_grans = []
    seen = set()
    for g in best_grans:
        gid = g['umm'].get('GranuleUR', None)
        if gid is None:
            continue
        if gid not in seen:
            unique_grans.append(g)
            seen.add(gid)

    best_grans = unique_grans
    print(f"{len(best_grans)} TOTAL granules w overlap (after deduplication)")

    if len(best_grans) == 0:
        raise ValueError(f"No MODIS granules found that overlap Landsat extent. "
                         f"lonbounds={lonboundsC}, latbounds={latboundsC}")

    # --- Sort by time difference (closest first), then try multiple until valid ---
    Mdates = [
        pd.to_datetime(g['umm']['TemporalExtent']['RangeDateTime']['BeginningDateTime'])
        for g in best_grans
    ]
    time_diffs = [abs(d - pytz.utc.localize(pd.to_datetime(ls_time))) for d in Mdates]
    sorted_idx = sorted(range(len(time_diffs)), key=lambda i: time_diffs[i])

    max_attempts = min(max_attempts, len(best_grans))

    # Landsat bounds in a consistent longitude system for coverage checks
    ls_lon_min, ls_lon_max = west, east
    ls_lat_min, ls_lat_max = south, north
    ls_crosses_idl = crosses_idl_flag

    for attempt in range(max_attempts):
        i = sorted_idx[attempt]
        g = best_grans[i]
        time_dif = time_diffs[i]
        gran_ur = g['umm']['GranuleUR'] 
        print(f"\nAttempt {attempt+1}/{max_attempts}: {gran_ur}  (Δt={time_dif})")
    
        try:
            fh = earthaccess.open([g])[0]
            mod_scene = xr.open_dataset(fh)
            mod_scene = mod_scene.rio.write_crs("epsg:4326", inplace=True)
    
            # Find SST variable
            sst_var_candidates = ["sea_surface_temperature", "sst", "Sea_Surface_Temperature"]
            sst_name = next((v for v in sst_var_candidates if v in mod_scene.variables), None)
            if sst_name is None:
                print("  ✗ Could not find SST variable; skipping.")
                continue
    
            sst = mod_scene[sst_name].values
            sst = sst.squeeze(axis=0)
            
            print(f"\n  === QA Diagnostics ===")
            print(f"  SST finite: {np.sum(np.isfinite(sst))} / {sst.size} ({100*np.sum(np.isfinite(sst))/sst.size:.1f}%)")
            
            # Check quality_level
            if "quality_level" in mod_scene.variables:
                ql = mod_scene["quality_level"].values
                if ql.ndim == 3 and ql.shape[0] == 1:
                    ql = ql.squeeze(axis=0)  # ✅ Squeeze here too
                
                print(f"  Quality level distribution:")
                for level in range(6):
                    count = np.sum(ql == level)
                    pct = 100 * count / ql.size
                    print(f"    Level {level}: {count} pixels ({pct:.1f}%)")
                
                for level in range(6):
                    mask = (ql == level) & np.isfinite(sst)
                    count = np.sum(mask)
                    if count > 0:
                        print(f"    Level {level} with finite SST: {count} pixels")
            
            # Accept Level 1 or higher (questionable quality acceptable for Antarctica)
            valid_mask = np.isfinite(sst)
            
            if "quality_level" in mod_scene.variables:
                valid_mask = valid_mask & (ql >= 1)  # Changed to >= 1
                print(f"  After QA (quality >= 1): {np.sum(valid_mask)} valid pixels")
            
            print(f"  ======================\n")
    
    
            if np.sum(valid_mask) == 0:
                print("  ✗ No valid SST values after QA; skipping.")
                continue
    
            # Get lon/lat
            if "lon" in mod_scene.coords:
                mod_lon = mod_scene["lon"].values
            elif "longitude" in mod_scene.variables:
                mod_lon = mod_scene["longitude"].values
            else:
                print("  ✗ Could not find lon; skipping.")
                continue
    
            if "lat" in mod_scene.coords:
                mod_lat = mod_scene["lat"].values
            elif "latitude" in mod_scene.variables:
                mod_lat = mod_scene["latitude"].values
            else:
                print("  ✗ Could not find lat; skipping.")
                continue
    
            # Handle 1D vs 2D coordinates
            if mod_lon.ndim == 1 and mod_lat.ndim == 1:
                LON, LAT = np.meshgrid(mod_lon, mod_lat)
            else:
                LON, LAT = mod_lon, mod_lat
    
            # Shift if needed
            if crosses_idl_flag:
                LON_shifted = LON.copy()
                LON_shifted[LON_shifted < 0] += 360
                ls_lon_min_shift = ls_lon_min if ls_lon_min >= 0 else ls_lon_min + 360
                ls_lon_max_shift = ls_lon_max if ls_lon_max >= 0 else ls_lon_max + 360
            else:
                LON_shifted = LON
                ls_lon_min_shift, ls_lon_max_shift = ls_lon_min, ls_lon_max
    
            # Get min/max from valid data only
            valid_lons = LON_shifted[valid_mask]
            valid_lats = LAT[valid_mask]
            
            if len(valid_lons) == 0:
                print("  ✗ No valid coordinates; skipping.")
                continue
                
            mod_lon_min = valid_lons.min()
            mod_lon_max = valid_lons.max()
            mod_lat_min = valid_lats.min()
            mod_lat_max = valid_lats.max()
    
            lon_covered = (mod_lon_min - buffer <= ls_lon_min_shift) and (ls_lon_max_shift <= mod_lon_max + buffer)
            lat_covered = (mod_lat_min - buffer <= ls_lat_min) and (ls_lat_max <= mod_lat_max + buffer)
    
            print(f"  MODIS: lon [{mod_lon_min:.2f}, {mod_lon_max:.2f}] lat [{mod_lat_min:.2f}, {mod_lat_max:.2f}]")
            print(f"  Landsat: lon [{ls_lon_min_shift:.2f}, {ls_lon_max_shift:.2f}] lat [{ls_lat_min:.2f}, {ls_lat_max:.2f}]")
    
            if not (lon_covered and lat_covered):
                print("  ✗ Doesn't cover Landsat; skipping.")
                continue
    
            # Count points
            in_scene = (
                (LON_shifted >= ls_lon_min_shift - buffer) &
                (LON_shifted <= ls_lon_max_shift + buffer) &
                (LAT >= ls_lat_min - buffer) &
                (LAT <= ls_lat_max + buffer) &
                valid_mask
            )
            points_in_scene = int(np.sum(in_scene))
    
            if points_in_scene < min_points:
                print(f"  ✗ Only {points_in_scene} points; skipping.")
                continue
    
            print(f"  ✓ Valid! {points_in_scene} MODIS points")
            return mod_scene, gran_ur, time_dif
    
        except Exception as e:
            print(f"  ✗ Error: {e}")
            import traceback
            traceback.print_exc()
            continue


    raise ValueError(
        f"Could not find a valid MODIS SST granule after {max_attempts} attempts. "
        f"lonbounds={lonboundsC}, latbounds={latboundsC}"
    )


##########################  

# Notes for changes - MODISlookup2 doesn't need to output lat/lon, but if want to do the check, 
# can take the lat/lon check out of aligne and do it in 
def get_wv(ls_scene,mod07,spacing,param,scene,interp=0):
    '''
    Aligns and resamples MODIS water vapor data to match the spatial resolution and 
    alignment of a given Landsat scene. The function optionally applies interpolation 
    to improve the data quality.

    Parameters:
    ls_scene (xarray.Dataset): The Landsat scene dataset containing spatial coordinates.
    mod07 (xarray.Dataset): The MODIS dataset containing water vapor data and coordinates (MOD/MDY07).
    spacing (list): Desired spatial resolution (y, x) for alignment with MODIS data in meters.
    param (str): Parameter name for the desired dataset within the MODIS file.
    scene: 
    interp (int): Controls interpolation mode - 0 for none, 1 for bicubic kriging interpolation.

    Returns:
    WV_xr (xarray.DataArray): The processed xarray data array containing the Landsat-aligned 
                              and resampled water vapor data from MODIS.

    Note:
    This function is also very similar to get_sst used in LandsatCalibration; any changes here should consider potential 
    impacts there - may need to be copied/generalized.

    The function performs several key operations:
    1. Defines the bounding box for the Landsat scene based on its spatial coordinates.
    2. Extracts the relevant water vapor data from the MODIS dataset using the specified parameter key.
    3. Validates the geographic coordinate ranges (latitude and longitude) of the MODIS data.
    4. Applies PyGMT interpolation if requested to generate a smoother water vapor data surface.
    5. Utilizes a lookup function to align MODIS data indices with the Landsat grid based on the specified spatial resolution.
    6. Aligns and resamples the MODIS data to match the Landsat scene's grid and spatial resolution.
    7. Adjusts the coordinate system of the output to ensure compatibility with further processing or analysis.

    Difference: no bicubic spline interpolation in LsatCalib during the upsampling, don't set new indexes at the end
    '''
    # Read in desired variables
    ULX = ls_scene.x[0] 
    ULY = ls_scene.y[0]  
    LRX = ls_scene.x[-1] 
    LRY = ls_scene.y[-1] 
    box = [ULX,LRX,ULY,LRY]

    # Apply QC
    data, qa_stats = apply_modis_qa(mod07)
    
    #Extract desired datasets from MODIS file from lookup key (automatically scaled by xarray so no need to do it here)
    # data = mod07[param].values # use this if don't QC
    lat, lon = mod07.Latitude, mod07.Longitude
    #data.attributes()

    # ***Need to use climatology to retrieve quantile data from this area
    
    # # Get rid of low outliers from over ice, cutoff for 98.5%
    # outlier = np.quantile(data[np.isfinite(data)],0.015) #0.015
    # mask2 = np.ones(data.shape)
    # mask2[data<outlier] = np.nan
    # data = np.around(mask2*data,decimals=5)

    # Interpolate using PyGMT
    if interp==1: 
        try:
            grid = interpMOD(data,lat,lon,scene)
            # Produce indicies for aligning MODIS pixel subset to match Landsat image at 4000m (or 300)resolution
            indiciesMOD,lines,samples,lat,lon = MODISlookup(mod07,ls_scene,box,spacing,scene,interpgrid=grid)
            data = grid.values
        except Exception as e: 
            print(e)
            print (f'atm correction of {ls_scene.id.values} failed')
            raise
        
    else:
        # Produce indicies for aligning MODIS pixel subset to match Landsat image at 4000m (or 300)resolution
        indiciesMOD,lines,samples,lat,lon = MODISlookup(mod07,ls_scene,box,spacing,scene)

    # Align and resample MODIS WV to Landsat at indicated spacing with correct axes
    dataOutWV_xr = alignMODIS(data,lat,lon,param,indiciesMOD,lines,samples,mod07,ls_scene,spacing)
    
    # # Resample WV to Landsat resolution and interpolate with B-spline
    # # Need to use 0.1k (this samples at .1 of the grid)
    # # Output of shape fits and need to adjust x and y coords cuz are wrong
    ups_factor = 30/spacing[0]
    WV_upsample = pygmt.grdsample(grid=dataOutWV_xr, spacing=f'{ups_factor}k', interpolation='c')

    # Put into Xarray
    # Sometimes spacing works properly with -1 and sometimes not
    latnew = np.arange(dataOutWV_xr.latitude[0],dataOutWV_xr.latitude[-1]+1,(dataOutWV_xr.latitude[-1]-dataOutWV_xr.latitude[0])/(WV_upsample.shape[0]-1))
    if (WV_upsample.shape[0]!=latnew.shape[0]):
        latnew = np.arange(dataOutWV_xr.latitude[0],dataOutWV_xr.latitude[-1]+1,(dataOutWV_xr.latitude[-1]-dataOutWV_xr.latitude[0])/(WV_upsample.shape[0]))

    # Put into Xarray
    latnew = ls_scene.y[:WV_upsample.shape[0]].values
    lonnew = ls_scene.x[:WV_upsample.shape[1]].values
    if dataOutWV_xr.latitude[0]!=latnew[0]:
        print('Aligned y dim needs to start with the same coordinate as ls_scene')
    if dataOutWV_xr.longitude[0]!=lonnew[0]:
        print('Aligned x dim needs to start with the same coordinate as ls_scene')
    
    WV_xr = xr.DataArray(WV_upsample,name='SST',dims=["y","x"], coords={"latitude": (["y"],latnew), "longitude": (["x"],lonnew)})

    WV_xr = WV_xr.rio.write_crs("epsg:3031", inplace=True)
    WV_xr = WV_xr.rename({'longitude':'x','latitude':'y'})
    
    return WV_xr
    
##########################  

def get_sst(ls_scene,mod07,spacing,param):
    '''
    ***This is copied in LandsatCalibration, modifications have been made but some may tranfer
    
    Create MODIS files aligned and subsampled to Landsat
    
    Variables:
    ls_scene = xarray dataset of a Landsat scene
    mod07 = xarray datarray with MODIS L2 SST data
    spacing = list of desired spatial resolution of output data from the alignment of MODIS to Landsat in y and x (e.g.,[300,-300])
    param = string for desired dataset from MODIS file
    
    Output:
    WV_xr = xarray dataarray of Landsat aligned and upsampled modis data from desired dataset
    
    Differences from NLSST: scene is not a parameter (used for test_gridcoords), SST gets extracted differently into data/lat/lon
    
    '''
    # Read in desired variables and paths
    
    uniqWV = []

    ULX = ls_scene.x[0] 
    ULY = ls_scene.y[0]
    LRX = ls_scene.x[-1]
    LRY = ls_scene.y[-1] 
    box = [ULX,LRX,ULY,LRY]
    
    #Extract desired datasets from MODIS file
    if param == 'sea_surface_temperature': 
        data = mod07[0,:,:]
        lat, lon = mod07.lat, mod07.lon
    else: 
        data = mod07[param].values
        lat, lon = mod07.Latitude, mod07.Longitude    

    # Produce indicies for aligning MODIS pixel subset to match Landsat image at 4000m (or 300)resolution
    indiciesMOD,lines,samples = MODISsstlookup(mod07,ls_scene,box,spacing)

    # Align MODIS SST to Landsat on slightly upsampled grid # have the option to output `uniqImgWV` if want to know range of data
    dataOutWV_xr = alignMODIS(data,lat,lon,param,indiciesMOD,lines,samples,mod07,ls_scene,spacing)

    # Resample MODIS to Landsat resolution and interpolate with B-spline
    # Output of shape fits and need to adjust x and y coords cuz are wrong
    ups_factor = 30/spacing[0]
    WV_upsample = pygmt.grdsample(grid=dataOutWV_xr, spacing=f'{ups_factor}k') # ,interpolation='b' if prefer to interpolate with bspline but don't think it is useful here

    # Put into Xarray
    latnew = ls_scene.y[:WV_upsample.shape[0]].values
    lonnew = ls_scene.x[:WV_upsample.shape[1]].values
    if dataOutWV_xr.latitude[0]!=latnew[0]:
        print('Aligned y dim needs to start with the same coordinate as ls_scene')
    if dataOutWV_xr.longitude[0]!=lonnew[0]:
        print('Aligned x dim needs to start with the same coordinate as ls_scene')
    
    WV_xr = xr.DataArray(WV_upsample,name='SST',dims=["y","x"], coords={"latitude": (["y"],latnew), "longitude": (["x"],lonnew)})
    WV_xr = WV_xr.rio.write_crs("epsg:3031", inplace=True)
    WV_xr = WV_xr.rename({'longitude':'x','latitude':'y'})
    WV_xr = WV_xr.set_index(x='x')
    WV_xr = WV_xr.set_index(y='y')
    
    return WV_xr
    
##############

def apply_modis_qa(mod07, WV='Water_Vapor'):
    """
    Essential MODIS MOD07 quality control - removes redundancy
    
    QC Criteria:
    1. Water_Vapor: not fill value (-9999) and valid
    2. Quality_Assurance_Infrared: useful=1 AND confidence>=1 (marginal or better)
    3. Cloud_Mask (bits 6-7): Land/Water = water (00) only, rejecting coastal mixed (01), desert/ice (10), and land (11)
    """
    wv_data = mod07[WV].values.copy()
    total_pixels = wv_data.size
    
    combined_mask = np.ones_like(wv_data, dtype=bool)
    qa_stats = {'total': total_pixels}
    
    
    # 1. Water_Vapor valid
    wv_valid = (wv_data != -9999) & np.isfinite(wv_data) & (wv_data > 0)
    combined_mask &= wv_valid
    qa_stats['wv_rejected'] = np.sum(~wv_valid)
    print(f"  After WV check: {np.sum(combined_mask)} valid ({100*np.sum(combined_mask)/total_pixels:.1f}%)")
    
    
    # 2. Quality_Assurance_Infrared (Byte 0 only)
    qa_byte0 = mod07['Quality_Assurance_Infrared'][:, :, 0].values  # ✅ Fixed: define qa_byte0 here
    qa_byte0 = qa_byte0.astype(np.int32)
        
    usefulness = (qa_byte0 >> 0) & 0b1
    confidence = (qa_byte0 >> 1) & 0b11
    
    qa_ir_valid = (usefulness == 1) & (confidence >= 1)
    combined_mask &= qa_ir_valid
    qa_stats['qa_rejected'] = np.sum(~qa_ir_valid)
    print(f"  After QA_IR check: {np.sum(combined_mask)} valid ({100*np.sum(combined_mask)/total_pixels:.1f}%)")
    
    
    # 3. Masking for land, coastal because mixed and throws off WV values by a lot, desert because likely land ice
    cloud_mask = mod07['Cloud_Mask'].values.copy()
        
    # Convert signed to unsigned
    cloud_unsigned = np.where(cloud_mask < 0, cloud_mask + 256, cloud_mask).astype(np.int32)
    
    # Bits 6-7: Land/Water Flag (shift right by 6 to get these bits)
    land_water = (cloud_unsigned >> 6) & 0b11
    
    print(f"  Cloud_Mask Land/Water unique: {np.unique(land_water)}")
    print(f"    0=Water: {np.sum(land_water==0)}, 1=Coastal: {np.sum(land_water==1)}, 2=Desert: {np.sum(land_water==2)}, 3=Land: {np.sum(land_water==3)}")
    
    # Keep water (0) only
    is_valid_surface = (land_water == 0)  # ✅ Fixed: use is_valid_surface consistently
    combined_mask &= is_valid_surface
    qa_stats['land_rejected'] = np.sum(~is_valid_surface)
    print(f"  After Land/Water check: {np.sum(combined_mask)} valid ({100*np.sum(combined_mask)/total_pixels:.1f}%)")
    
    
    # Apply mask
    wv_data[~combined_mask] = np.nan
    qa_stats['valid'] = np.sum(np.isfinite(wv_data))
    qa_stats['valid_pct'] = 100 * qa_stats['valid'] / total_pixels
    
    print(f"  → Final MODIS QC: {qa_stats['valid_pct']:.1f}% valid ({qa_stats['valid']}/{total_pixels} pixels)")
    
    return wv_data, qa_stats
    
#######

def interpMOD(data, lat, lon, scene, target_spacing_m=1500, buffer_deg=2.0):
    """
    Interpolate spatial water vapor data using PyGMT over the Landsat scene extent.
    """
    
    # Check if lat/lon are xarray DataArrays or numpy arrays
    if hasattr(lat, 'values'):
        lat_vals = lat.values
        lon_vals = lon.values
    else:
        lat_vals = lat
        lon_vals = lon
    
    # Get Landsat scene extent from metadata
    lon_min, lat_min, lon_max, lat_max = scene.metadata['bbox']
    
    # Check if LANDSAT crosses IDL
    landsat_crosses_idl = (lon_max < lon_min) or (lon_max - lon_min > 180)
    
    # Check if MODIS data crosses IDL
    modis_lon_span = lon_vals.max() - lon_vals.min()
    modis_crosses_idl = modis_lon_span > 180
    
    # If EITHER crosses, shift BOTH to 0-360
    crosses_idl = landsat_crosses_idl or modis_crosses_idl
    
    # Shift coordinates BEFORE masking
    if crosses_idl:
        if landsat_crosses_idl and modis_crosses_idl:
            print("⚠️  Both Landsat and MODIS cross IDL! Shifting to 0-360 range")
        elif landsat_crosses_idl:
            print("⚠️  Landsat crosses IDL! Shifting to 0-360 range")
        else:
            print("⚠️  MODIS crosses IDL! Shifting to 0-360 range")
        
        # Shift MODIS longitude array
        lon_shifted = lon_vals.copy()
        lon_shifted[lon_shifted < 0] += 360
        
        # Shift Landsat bounds
        if lon_min < 0:
            lon_min += 360
        if lon_max < 0:
            lon_max += 360
        
        # print(f"Shifted Landsat extent: lon [{lon_min:.4f}, {lon_max:.4f}]")
        # print(f"Shifted MODIS extent: lon [{lon_shifted.min():.4f}, {lon_shifted.max():.4f}]")
    else:
        lon_shifted = lon_vals
    
    print(f"Landsat extent: lon [{lon_min:.4f}, {lon_max:.4f}], lat [{lat_min:.4f}, {lat_max:.4f}]")
    
    # Create mask for valid data within Landsat extent
    
    valid_mask = np.isfinite(data) & (data > 0)
    spatial_mask = (
        (lon_shifted >= lon_min - buffer_deg) &
        (lon_shifted <= lon_max + buffer_deg) &
        (lat_vals >= lat_min - buffer_deg) &
        (lat_vals <= lat_max + buffer_deg)
    )
    
    combined_mask = valid_mask & spatial_mask
    points_in_scene = np.sum(combined_mask)
    # print(f"MODIS points within Landsat extent: {points_in_scene}")
    
    if points_in_scene < 10:
        raise ValueError(
            f"Too few MODIS points overlap Landsat scene: {points_in_scene}. "
            f"Cannot interpolate reliably."
        )
    
    # Create DataFrame with explicit float64 dtype
    df = pd.DataFrame({
        'longitude': lon_shifted[combined_mask].astype(np.float64),
        'latitude': lat_vals[combined_mask].astype(np.float64),
        'water_vapor': data[combined_mask].astype(np.float64)
    })
    
    # print(f"Valid MODIS data points for interpolation: {len(df)}")
    print(f"Mean water vapor: {df.water_vapor.mean():.4f}")
    
    # Get data extent
    lon_data_min = df.longitude.min()
    lon_data_max = df.longitude.max()
    lat_data_min = df.latitude.min()
    lat_data_max = df.latitude.max()
    
    print(f"Data extent: lon [{lon_data_min:.4f}, {lon_data_max:.4f}], lat [{lat_data_min:.4f}, {lat_data_max:.4f}]")
    
    # Decide interpolation extent based on data density
    # First, calculate a preliminary data density estimate
    avg_lat_prelim = (lat_min + lat_max) / 2
    spacing_lat_deg_prelim = target_spacing_m / 111000.0
    spacing_lon_deg_prelim = target_spacing_m / (111000.0 * np.cos(np.radians(avg_lat_prelim)))
    interp_buffer = buffer_deg  # degrees
    
    # Preliminary grid size using Landsat extent
    lon_grid_min_prelim = lon_min - interp_buffer
    lon_grid_max_prelim = lon_max + interp_buffer
    lat_grid_min_prelim = lat_min - interp_buffer
    lat_grid_max_prelim = lat_max + interp_buffer
    
    nx_prelim = int(np.round((lon_grid_max_prelim - lon_grid_min_prelim) / spacing_lon_deg_prelim))
    ny_prelim = int(np.round((lat_grid_max_prelim - lat_grid_min_prelim) / spacing_lat_deg_prelim))
    total_cells_prelim = nx_prelim * ny_prelim
    data_density_prelim = total_cells_prelim / len(df)
    
    print(f"Preliminary data density: 1 MODIS point per {data_density_prelim:.1f} grid cells")
    
    # If data is too sparse, use actual data extent instead of Landsat extent
    if data_density_prelim > 200:
        print(f"⚠️  Sparse data detected! Using actual data extent to avoid extrapolation.")
        # Use data extent with smaller buffer
        interp_buffer = 0.5  # Smaller buffer for sparse data
        lon_grid_min = lon_data_min - interp_buffer
        lon_grid_max = lon_data_max + interp_buffer
        lat_grid_min = lat_data_min - interp_buffer
        lat_grid_max = lat_data_max + interp_buffer
    else:
        # Use Landsat extent for interpolation grid (with 2 degree buffer)
        lon_grid_min = lon_min - interp_buffer
        lon_grid_max = lon_max + interp_buffer
        lat_grid_min = lat_min - interp_buffer
        lat_grid_max = lat_max + interp_buffer
    
    print(f"Interpolation extent: lon [{lon_grid_min:.4f}, {lon_grid_max:.4f}], lat [{lat_grid_min:.4f}, {lat_grid_max:.4f}]")    
    
    avg_lat = (lat_grid_min + lat_grid_max) / 2
    
    # Convert spacing from meters to degrees
    spacing_lat_deg = target_spacing_m / 111000.0
    spacing_lon_deg = target_spacing_m / (111000.0 * np.cos(np.radians(avg_lat)))
    
    print(f"Target spacing: {target_spacing_m}m")
    print(f"  → lon: {spacing_lon_deg:.8f}°")
    print(f"  → lat: {spacing_lat_deg:.8f}°")
    
    # Calculate grid dimensions based on LANDSAT extent (not data extent)
    nx = int(np.round((lon_grid_max - lon_grid_min) / spacing_lon_deg))
    ny = int(np.round((lat_grid_max - lat_grid_min) / spacing_lat_deg))
    nx = max(nx, 2)
    ny = max(ny, 2)
    
    # Adjust max bounds to fit exactly
    lon_grid_max_adj = lon_grid_min + nx * spacing_lon_deg
    lat_grid_max_adj = lat_grid_min + ny * spacing_lat_deg
    
    region = [lon_grid_min, lon_grid_max_adj, lat_grid_min, lat_grid_max_adj]
    
    total_cells = nx * ny
    data_density = total_cells / len(df)

    # After calculating data_density:
    print(f"Interpolation grid: {nx} x {ny} = {total_cells:,} cells")
    print(f"Data density: 1 MODIS point per {data_density:.1f} grid cells")
    
    spacing_str = f"{spacing_lon_deg:.15f}/{spacing_lat_deg:.15f}"
    
    # Blockmedian preprocessing
    print("Running blockmedian...")
    df_blocked = pygmt.blockmedian(
        data=df[['longitude', 'latitude', 'water_vapor']],
        region=region,
        spacing=spacing_str,
    )
    
    points_reduced = len(df) - len(df_blocked)
    # print(f"Blockmedian: {len(df)} → {len(df_blocked)} points ({points_reduced} duplicates removed)")

    if len(df_blocked) < 10:
        raise ValueError(f"Blockmedian left too few points: {len(df_blocked)}")
    
    # Surface interpolation
    print("Running surface interpolation...")
    grid = pygmt.surface(
        data=df_blocked,
        region=region,
        spacing=spacing_str,
        tension=0.95,
    )
    
    print(f"✓ Interpolation complete. Output grid: {grid.shape}")

    # Grid check:
    print(f"Grid data range: {float(grid.values.min()):.6f} to {float(grid.values.max()):.6f}")
    # print(f"Grid has NaN: {np.isnan(grid.values).sum()} / {grid.values.size} pixels")
    # print(f"Grid valid pixels: {np.sum(~np.isnan(grid.values))}")
    
    # Rename coordinates
    grid = grid.rename({'x': 'lon', 'y': 'lat'})

    # # Shift lon back to -180/180 if crosses_idl
    # if crosses_idl:
    #     print("Shifting grid coordinates back to -180/180 range")
    #     lon_coords = grid.lon.values.copy()
    #     lon_coords[lon_coords > 180] -= 360
    #     grid = grid.assign_coords({'lon': lon_coords})
    #     print(f"Final lon range: {grid.lon.min().values:.2f} to {grid.lon.max().values:.2f}")

    if crosses_idl:
        lon_coords = grid.lon.values.copy()
        grid_span = lon_coords.max() - lon_coords.min()
        
        # If grid spans more than 180°, something went wrong
        if grid_span > 180:
            print(f"WARNING: Grid spans {grid_span:.2f}°, likely wrapping issue")
        
        # For IDL-crossing scenes, keep in 0-360 to maintain continuity
        if grid_span < 20:  # Typical Landsat scene is 5-15° wide
            print(f"Grid spans {grid_span:.2f}° - keeping in 0-360 space for IDL continuity")
            print(f"Grid lon range: {lon_coords.min():.2f} to {lon_coords.max():.2f}")
            # Don't shift back - keep as-is
        else:
            print(f"Grid spans {grid_span:.2f}° - unexpected, keeping as-is")
        
        grid = grid.assign_coords({'lon': lon_coords})
    else:
        # For non-IDL scenes, coordinates are already in -180/180
        print(f"Grid lon range: {grid.lon.min().values:.2f} to {grid.lon.max().values:.2f}")
    
    return grid

##########################

def MODISlookup(mod07,ls_scene,box,spacing,scene,interpgrid=None):
    '''
    Look up indices for aligning MODIS product to the Landsat grid
    # Modified from http://stackoverflow.com/questions/2922532/obtain-latitude-and-longitude-from-a-geotiff-file 
    # and Shane Grigsby

    Variables:    
    mod07 = xarray with MODIS data with crs 4326 assigned
    ls_scene =  Landsat xarray DataArray
    box = list with [left easting,right easting,top northing,bottom northing]
    spacing = desired pixel size for extraction, list of [east/west, north/south] 
          (recommend choosing a number that allows for fast calculations and even division by 30)
    scene = 
    interpgrid = xarray of mod07 data that has been through interpolation in PyGMT (optional)

    Output:
    indiciesMOD = indicies used to project MODIS pixels to match Landsat pixels
    lines = number of lines in Landsat file/MODIS output shape
    samples = number of samples in Landsat file/MODIS output shape
    lon,lat = 2D lon and lat coordinates for grid
    '''
    test_threshold = 5
    
    if interpgrid is None:
        lat, lon = mod07.Latitude.values, mod07.Longitude.values
    else:
        lat, lon = interpgrid.lat, interpgrid.lon
        lon, lat = np.meshgrid(lon,lat)

    # Test lat is in correct range
    if ~((lat <= 90) & (lat >= -90)).all():
        print('MODIS latitude not between -90 and 90')
    # Test lon is in correct range
    if ~((lon <= 180) & (lon >= -180)).all():
        print('MODIS longitude not between -180 and 180')

    # Get the existing coordinate system
    old_cs = ls_scene.rio.crs # 'epsg:3031'
    new_cs = mod07.rio.crs # 'epsg:4326'

    # Create a transform object to convert between coordinate systems
    inProj = Proj(init=old_cs)
    outProj = Proj(init=new_cs)

    # Parse coordinates and spacing to different variables
    west,east,north,south = box
    ewspace,nsspace = spacing

    # Setting up grid, x coord from here to here at this spacing, mesh grid makes 2D
    samples = len(np.r_[west:east+1:ewspace])
    lines = len(np.r_[north:south-1:nsspace])#ns space is -300, could also do 30 instead of 300, but would just have duplicate pixels
    if lines==0:
        lines = len(np.r_[south:north-1:nsspace])

    # x1, y1 = np.meshgrid(np.r_[west:east:ewspace],np.r_[north:south:nsspace]) # offset by 1 meter to preserve shape
    ewdnsamp = int(spacing[0]/30)
    nsdnsamp = int(spacing[1]/30)
    
    # Set up coarser sampling and check to make sure is in the same orientation as the original Landsat grid
    xresamp = ls_scene.x.isel(x=slice(None, None, ewdnsamp)).values
    if xresamp[0]!=ls_scene.x.values[0]:
        xresamp = ls_scene.x.isel(x=slice(None, None, -ewdnsamp)).values
        # print('x resample reversed')
    yresamp = ls_scene.y.isel(y=slice(None, None, nsdnsamp)).values
    if yresamp[0]!=ls_scene.y.values[0]:
        yresamp = ls_scene.y.isel(y=slice(None, None, -nsdnsamp)).values
        # print('y resample reversed')
    x1, y1 = np.meshgrid(xresamp,yresamp)
    LScoords = np.vstack([x1.ravel(),y1.ravel()]).T
    if (LScoords[0,0]!=ls_scene.x.values[0]) |  (LScoords[0,1]!=ls_scene.y.values[0]):
        raise Exception('Landsat coordinates do not match expected during MODIS lookup')

    # Ravel so ND can lookup easily
    # Convert from LS map coords to lat lon --> x = lon, y = lat (usually?)

    ###Make into test
    # Test that reprojection is working correctly on first and last grid point using round-trip transformation
    xs1, ys1 =  transform(inProj,outProj,LScoords[0,0], LScoords[0,1], radians=True, always_xy=True)
    xsl1, ysl1 =  transform(outProj,inProj,xs1, ys1, radians=True, always_xy=True)
    if np.linalg.norm(np.array([xsl1, ysl1]) - LScoords[0,:]) > test_threshold:
        print(f"Round-trip transformation error for point {LScoords[0,:]}, {np.linalg.norm(np.array([xsl1, ysl1]) - LScoords[0,:])}")
    else:
        # If passes, run on entire grid
        xs, ys =  transform(inProj,outProj,LScoords[:,0], LScoords[:,1], radians=True, always_xy=True)
    ###
    
    # Produce landsat reprojected to lat/lon and ensure lat is in 0 column
    # Test: landsat data is in correct orientation as long as lat is in col 0 and lon in col 1
    grid_coords = test_gridcoords(xs,ys,scene)

    # Test that lines and samples match grid_coords
    if len(grid_coords) != lines*samples:
        raise Exception(f'Size of grid coordinates do not match low resolution Landsat dims: {len(grid_coords)} vs. {lines*samples}. Check that spacing is negative for y')
    MODIS_coords = np.vstack([lat.ravel(),lon.ravel()]).T
    MODIS_coords *= np.pi / 180. # to radians

    # Build lookup, haversine = calc dist between lat,lon pairs so can do nearest neighbor on sphere - if did utm it would be planar
    MOD_Ball = BallTree(MODIS_coords,metric='haversine') #sklearn library
    distanceMOD, indiciesMOD= MOD_Ball.query(grid_coords, dualtree=True, breadth_first=True)
        
    return indiciesMOD,lines,samples,lat,lon

##########################           

def MODISsstlookup(mod07, ls_scene, box, spacing):
    '''
    Look up atmospheric consituents from MODIS product for each Landsat pixel
    '''
    test_threshold = 5 
    
    lat, lon = mod07.lat, mod07.lon
    
    # Test lat is in correct range
    if ~((lat <= 90) & (lat >= -90)).all():
        print('MODIS latitude not between -90 and 90')
    # Test lon is in correct range
    if ~((lon <= 180) & (lon >= -180)).all():
        print('MODIS longitude not between -180 and 180')

    # Get the existing coordinate system
    old_cs = ls_scene.rio.crs
    new_cs = mod07.rio.crs

    # Create a transform object to convert between coordinate systems
    inProj = Proj(init=old_cs)
    outProj = Proj(init=new_cs)

    # Parse coordinates and spacing
    west, east, north, south = box
    ewspace, nsspace = spacing

    # Setting up grid
    samples = len(np.r_[west:east+1:ewspace])
    lines = len(np.r_[north:south-1:nsspace])
    if lines == 0:
        lines = len(np.r_[south:north-1:nsspace])
        
    ewdnsamp = int(spacing[0]/30)
    nsdnsamp = int(spacing[1]/30)

    # Set up coarser sampling
    xresamp = ls_scene.x.isel(x=slice(None, None, ewdnsamp)).values
    if xresamp[0] != ls_scene.x.values[0]:
        xresamp = ls_scene.x.isel(x=slice(None, None, -ewdnsamp)).values
        
    yresamp = ls_scene.y.isel(y=slice(None, None, nsdnsamp)).values
    if yresamp[0] != ls_scene.y.values[0]:
        yresamp = ls_scene.y.isel(y=slice(None, None, -nsdnsamp)).values

    x1, y1 = np.meshgrid(xresamp, yresamp)
    LScoords = np.vstack([x1.ravel(), y1.ravel()]).T
    if (LScoords[0,0] != ls_scene.x.values[0]) | (LScoords[0,1] != ls_scene.y.values[0]):
        raise Exception('Landsat coordinates do not match expected during MODIS lookup')

    # Test round-trip transformation
    xs1, ys1 = transform(inProj, outProj, LScoords[0,0], LScoords[0,1], radians=True, always_xy=True)
    xsl1, ysl1 = transform(outProj, inProj, xs1, ys1, radians=True, always_xy=True)
    if np.linalg.norm(np.array([xsl1, ysl1]) - LScoords[0,:]) > test_threshold:
        print(f"Round-trip transformation error for point {LScoords[0,:]}, {np.linalg.norm(np.array([xsl1, ysl1]) - LScoords[0,:])}")
    else:
        xs, ys = transform(inProj, outProj, LScoords[:,0], LScoords[:,1], radians=True, always_xy=True)
    
    # Convert back to degrees to see what you actually got
    xs_deg = xs * 180 / np.pi
    ys_deg = ys * 180 / np.pi
    print(f"Transformed coords in degrees:")
    print(f"  xs_deg: {xs_deg.min():.2f} to {xs_deg.max():.2f}")
    print(f"  ys_deg: {ys_deg.min():.2f} to {ys_deg.max():.2f}")

    # Produce landsat reprojected to lat/lon and ensure lat is in 0 column
    grid_coords = test_gridcoords_calib(xs, ys)
    
    # Test that lines and samples match grid_coords
    if len(grid_coords) != lines*samples:
        raise Exception(f'Size of grid coordinates do not match low resolution Landsat dims: {len(grid_coords)} vs. {lines*samples}')
    
    # Filter out nan coordinates before building balltree
    lat_vals = lat.values.ravel()
    lon_vals = lon.values.ravel()
    
    # Create valid mask
    valid_mask = np.isfinite(lat_vals) & np.isfinite(lon_vals)
    
    # Filter to valid coordinates only
    lat_valid = lat_vals[valid_mask]
    lon_valid = lon_vals[valid_mask]
    
    if len(lat_valid) == 0:
        raise ValueError("No valid MODIS SST coordinates found")
    
    MODIS_coords = np.vstack([lat_valid, lon_valid]).T
    MODIS_coords *= np.pi / 180.
    
    # Build lookup
    MOD_Ball = BallTree(MODIS_coords, metric='haversine')
    distanceMOD, indiciesMOD_valid = MOD_Ball.query(grid_coords, dualtree=True, breadth_first=True)
    
    # Map back to original indices (accounting for NaN filtering)
    # Get original indices of valid points
    valid_indices = np.where(valid_mask)[0]
    
    # Map the BallTree indices back to original array indices
    indiciesMOD = valid_indices[indiciesMOD_valid]
    
    return indiciesMOD, lines, samples

##########################

def test_gridcoords(xs,ys,scene):
    '''
    Test to ensure grid lat and lon are not swapped during reprojection and output grid coordinates
    that have been raveled and stacked for input into BallTree
    
    Variables:
    xs = 1D radians representing longitude 
    ys = 1D radians representing latitude
    scene = catalog item for landsat image
    
    Output:
    grid_coords = two columns of x/y radian pairs representing lon/lat
    '''
    
    # Convert radians to lat/lon
    x_check = xs * 180. / np.pi
    y_check = ys * 180. / np.pi
    
    # We know lat is ys and lon is xs if this is true so goes in 0 column position to match MODIS
    if ((-90 <= y_check) & (y_check <= -60)).all() & ~((-90 <= x_check) & (x_check <= -60)).all():
        grid_coords = np.vstack([ys.ravel(),xs.ravel()]).T # note y / x switch (i.e., lat long convention)
        # print('Latitude in proper position')

    # A small subset of data have lat and lon that falls between -60 and -90 so test if the landsat metadata confirms that
    elif ((-90 <= y_check) & (y_check <= -60)).all():
        llons = np.array((float(scene.metadata['bbox'][0]), float(scene.metadata['bbox'][2])))
        # ys is latitude if true here
        if ((-90 <= llons) & (llons <= -60)).all():
            grid_coords = np.vstack([ys.ravel(),xs.ravel()]).T # note y / x switch (i.e., lat long convention)
            # print('Latitude in proper position')
        # xs is latitude if not and goes in 0 column position
        else:
            grid_coords = np.vstack([xs.ravel(),ys.ravel()]).T 
            print('Latitude in wrong position')

    # Otherwise xs is latitude and goes in 0 column position
    else:
        grid_coords = np.vstack([xs.ravel(),ys.ravel()]).T
        print('Latitude in wrong position')
    
    return grid_coords

##########################

def test_gridcoords_calib(xs,ys):
    '''
    Test to ensure grid lat and lon are not swapped during reprojection and output grid coordinates
    that have been raveled and stacked for input into BallTree. There is some uncertainty only when the image is
    taken between -60 and -90 longitude because lat and lon can have the same values.
    
    Variables:
    xs = 1D radians representing longitude 
    ys = 1D radians representing latitude
    
    Output:
    grid_coords = two columns of x/y radian pairs representing lon/lat
    
    Differences from NLSST: elif is different than NLSST pipeline
    '''
    
    # Convert radians to lat/lon
    x_check = xs * 180. / np.pi
    y_check = ys * 180. / np.pi
    
    # We know lat is ys and lon is xs if this is true so goes in 0 column position to match MODIS
    if ((-90 <= y_check) & (y_check <= -60)).all() & ~((-90 <= x_check) & (x_check <= -60)).all():
        grid_coords = np.vstack([ys.ravel(),xs.ravel()]).T # note y / x switch (i.e., lat long convention)
        print('Latitude in proper position')

    # A small subset of data have lat and lon that falls between -60 and -90 so test if the landsat metadata confirms that
    elif ((-90 <= y_check) & (y_check <= -60)).all():
        # xs is latitude if not and goes in 0 column position
        grid_coords = np.vstack([ys.ravel(),xs.ravel()]).T 
        print('Latitude in uncertain position, may be incorrect')

    # Otherwise xs is latitude and goes in 0 column position
    else:
        grid_coords = np.vstack([xs.ravel(),ys.ravel()]).T
        print('Latitude in wrong position')
    
    return grid_coords

##########################

def alignMODIS(data,lat,lon,param,indiciesMOD,lines,samples,mod07,ls_scene,spacing):
    '''
    Align MODIS image to Landsat and resample at indicated spacing
    
    Variables:
    data =
    lat = 
    lon = 
    param =
    indiciesMOD =
    lines = 
    samples =
    mod07 = 
    ls_scene =
    spacing =
    
    Output:
    dataOut_xr = 
    
    Not currently set, but can also output: 
    uniqImg = uniq MODIS atm values within area of Landsat image
    '''
    test_threshold = 5
    
    # Check to ensure lat/lon and data have compatible shapes
    if (np.shape(lat)== np.shape(lon)== np.shape(data))==False:
        raise Exception("Error in creating indicies, lat/lon and data shapes do not match")
        
    # Extract MODIS data into Landsat grid and gather unique data values
    dataOut,uniqWV = uniqueMODIS(data,param,indiciesMOD,lines,samples)
    
    # Check grid directionality and create matching x/y for new grid
    # Define the source and target coordinate reference systems (CRS)
    src_crs = mod07.rio.crs #'epsg:4326'  MODIS
    target_crs = ls_scene.rio.crs #crs[6:] # 'epsg:3031' Landsat

    # Create a PyProj transformer
    transformer = pyproj.Transformer.from_crs(src_crs, target_crs, always_xy=True)
    transformer_test = pyproj.Transformer.from_crs(target_crs, src_crs, always_xy=True)

    # Test that reprojection is working correctly on first and last modis grid point
    xm1,xm2 = lon[0,0],lon[-1,-1]
    ym1,ym2 = lat[0,0],lat[-1,-1]
    xx,yy = [xm1,xm2], [ym1,ym2]
    xs1, ys1 =  transformer.transform(xx,yy)
    xsl1, ysl1 = transformer_test.transform(xs1, ys1)
    for i,n in enumerate(xsl1):
        if np.linalg.norm(np.array([xsl1[i], ysl1[i]]) - [xx[i],yy[i]]) > test_threshold:
            print(f"Round-trip transformation error for this scene, {np.linalg.norm(np.array([xsl1[i], ysl1[i]]) - xx[i],yy[i])}")
    
    # Spacing to create x and y parameters at the correct spacing
    redy = int(abs(spacing[0]/30))
    redx = int(abs(spacing[1]/30))

    # From SST# Set up coarser sampling and check to make sure is in the same orientation as the original Landsat grid
    # xgrid = ls_scene.x.isel(x=slice(None, None, redx)).values
    # if xgrid[0]!=ls_scene.x.values[0]:
    #     xgrid = ls_scene.x.isel(x=slice(None, None, -redx)).values
    # ygrid = ls_scene.y.isel(y=slice(None, None, redy)).values
    # if ygrid[0]!=ls_scene.y.values[0]:
    #     ygrid = ls_scene.y.isel(y=slice(None, None, -redy)).values
    # if (xgrid[0]!=ls_scene.x.values[0]) |  (ygrid[0]!=ls_scene.y.values[0]):
    #     raise Exception('Landsat coordinates do not match expected during MODIS lookup')

    #From LandsatCalib
    # Set up coarser sampling grid to match spacing and check to make sure is in the same orientation as the original Landsat grid
    xgrid = ls_scene.x.values[0::redx]
    if len(xgrid)==1:
        xgrid = ls_scene.x.values[0::-redx]
    if xgrid[0]!=ls_scene.x.values[0]:
        xgrid = np.flip(xgrid)
        print ('Align x flip')
    ygrid = ls_scene.y.values[0::redy]
    if len(ygrid)==1:
        ygrid = ls_scene.y.values[0::-redy]
    if ygrid[0]!=ls_scene.y.values[0]:
        ygrid = np.flip(ygrid)
        print ('Align y flip')
    if (xgrid[0]!=ls_scene.x.values[0]) |  (ygrid[0]!=ls_scene.y.values[0]):
        raise Exception(f'Landsat coordinates do not match expected during MODIS align')
    
    # Create xarray from numpy array
    dataOut_xr = xr.DataArray(dataOut,name='SST',dims=["y","x"], coords={"latitude": (["y"],ygrid), "longitude": (["x"],xgrid)})
    
    return dataOut_xr

##########################

def uniqueMODIS(data,param,indiciesMOD,lines,samples):
    '''
    Extracts data values and unique values from desired MODIS dataset that corresponds to Landsat file
    No scaling needed - xarray automatically scales for you
    # Modified from http://stackoverflow.com/questions/2922532/obtain-latitude-and-longitude-from-a-geotiff-file
    
    Variables: 
    data = array with MOD07 data in crs 4326 assigned 
    param =  string for desired dataset from MODIS file
    indiciesMOD = indicies output for neighest neighbor query from MODIS to Landsat coordinates
    lines = number of lines in Landsat file/MODIS output shape
    samples = number of samples in Landsat file/MODIS output shape
    
    Output:
    dataOut = MODIS atm image subset and aligned to Landsat image pixels
    uniq = uniq MODIS atm values within area of Landsat image
    #counts = count for each unique value in subset
    '''
    # Convert from K to C
    KtoC = -273.15
    
    # Reproject data from MODIS into corresponding postions for Landsat pixels for the desired dataset
    # Remove unrealistic data/outliers
    # Scaling has already been automatically done by xarray
    if param == 'sea_surface_temperature':  
        #Extract desired datasets from MODIS file from lookup key
        # Move to adjusted grid and rescale data
        dataOut = np.reshape(np.array(data.values.ravel())[indiciesMOD],(lines,samples)) + KtoC #* # to scale?
        dataOut[dataOut < -3.5] = np.nan
    elif param == 'Water_Vapor':
        dataOut = np.reshape(np.array(data.ravel())[indiciesMOD],(lines,samples))
        dataOut[dataOut < 0] = np.nan
        MODimg = np.array(data)
        MODimg[MODimg < 0] = np.nan
    elif param == 'Total_Ozone':
        dataOut = np.reshape(np.array(data.ravel())[indiciesMOD],(lines,samples))
        dataOut[dataOut < 225] = np.nan
        dataOut[dataOut > 430] = np.nan
        MODimg = np.array(data)
        MODimg[MODimg < 0] = np.nan

    # Get unique values for datasets within Landsat extent
    #uniq, inverse, counts= np.unique(dataOut, return_inverse=True, return_counts=True)
    uniq = set(dataOut[np.isfinite(dataOut)])
    
    return dataOut,uniq # Can also output MODimg and inverse and counts if desired


# In[28]:


# Functions for deriving SST retrieval coefficients
'''
These functions help to derive the SST monthly correction coefficients
prep_retrieval prepares the inputs for running the multiple regression that determines the coefficients, including
converting ERA-5 specific humidity data to total column water vapor in spec_hu_to_tcwv. Derive retrieval then takes
the inputs and runs an OLS multiple regression to derive the coefficients.

Functions to search, open, and analyze Landsat scenes.
Search_stac finds the Landsat scene based on user parameters, 
plot_search plots the locations of the landsat scenes from the search,
landsat_to_xarray takes one of those scenes and puts all bands into an xarray,
and create_masks produces cloud/ice/water masks for the scene. Subset_img 
subsets a landsat scene with coordinates that have been reprojected from lat/lon
and may be flipped in which is larger in the pair. Lsat_reproj can be used to reproject
while ensuring x and y pairs don't get flipped (common converting between espg 3031 and wgs84.
'''
def prep_retrieval(atmpath,prefix,spec_hu_file):
    '''
    Create the inputs for the SST algorithm using the atmospheric column inputs and outputs from 
    the MODTRAN model runs for Landsat. Uses specific humidity to calculate total column water vapor 
    for the retrieval multiple regression.

    Units for ERA5 specific humidity listed here under main variables: https://cds.climate.copernicus.eu/datasets/reanalysis-era5-pressure-levels?tab=overview
    
    Variable:
    atmpath = directory path where MODTRAN outputs are stored (str)
    prefix = beginning of file path name for MODTRAN outputs - files created by Andy Harris (str) 
    spec_hu_file = file path for era5 input file for MODTRAN that includes specific humidity (str)
    
    Intermediates:
    modtran_lut = pandas dataframe of MODTRAN outputs
    modtran_atm = pandas dataframe of era5 atmopheric columns for input into MODTRAN
    
    Outputs:
    modtran_lut = pandas dataframe of MODTRAN outputs with total column water vapor [cm] added as 
                  a column
    '''
    
    # Open and concatenate MODTRAN outputs for SST algorithm
    # Get file paths
    modtr_list = os.listdir(atmpath)
    modtr_list = [file for file in modtr_list if file.startswith(prefix)]
    modtr_list.sort()

    # Open into pandas and concatenate
    df_list = []
    lut_cols = ['wind spd [m/s]','Surface T[K]','TOA T[K]','transmittance','jakobian']
    
    if len(modtr_list)>1:
        for mfile in modtr_list:
            df_list.append(pd.read_csv(f'{atmpath}/{mfile}', sep=' ',header=None,names=lut_cols))

        modtran_lut = pd.concat(df_list, ignore_index=True)
    else:
        mfile = modtr_list[0]
        modtran_lut = pd.read_csv(f'{atmpath}/{mfile}', sep=' ',header=None,names=lut_cols)

    # Remove last row only if it duplicates the second-to-last row
    print(f"MODTRAN LUT before cleaning: {modtran_lut.shape[0]} rows")
    
    if modtran_lut.shape[0] >= 2 and modtran_lut.iloc[-1].equals(modtran_lut.iloc[-2]):
        modtran_lut = modtran_lut.iloc[:-1].reset_index(drop=True)
        print(f"  Removed duplicate last row")
    
    print(f"MODTRAN LUT after cleaning: {modtran_lut.shape[0]} rows")
    
    modtran_lut['TCWV [cm]'] = np.nan
    
    # Open atm profiles for input of water vapor specific humidity
    atm_cols = ['Altitude [km]', 'pressure [hPa]', 'temp [K]', 'spec humidity [kg/kg]']
    modtran_atm = pd.read_csv(f'{atmpath}/{spec_hu_file}', sep='\t',header=None,names=atm_cols)

    # Ensure profiles match outputs
    expected_profiles = modtran_atm.shape[0] // 37
    actual_profiles = modtran_lut.shape[0]
    
    if expected_profiles != actual_profiles:
        print(f"⚠️  WARNING: Mismatch between profiles and outputs!")
        print(f"   Atmospheric profiles: {modtran_atm.shape[0]} rows = {expected_profiles} profiles")
        print(f"   MODTRAN outputs: {actual_profiles} rows")
        
        # Trim atmospheric data to match
        if expected_profiles > actual_profiles:
            modtran_atm = modtran_atm.iloc[:actual_profiles * 37]
            print(f"   Trimmed atm profiles to {modtran_atm.shape[0]} rows")
    
    # Run integral to get tcwv
    modtran_lut = spec_hu_to_tcwv(modtran_lut,modtran_atm)

    tcwvmin = modtran_lut['TCWV [cm]'].min()
    tcwvmax = modtran_lut['TCWV [cm]'].max()
    print (f'TCWV min: {tcwvmin}, max: {tcwvmax}')
    
    return modtran_lut

##########################

def concat_modtran_months(months,atmpath):
    # Create a list to store the DataFrame for each month in the window.
    modtran_list = []

    n = 0
    
    for mo in months:
        TCWV_input_file = atmpath / f"TCWV_{mo}.csv"
        if os.path.isfile(TCWV_input_file):
            # print(f"  Month {mo}: retrieval input exists")
            modtran = pd.read_csv(TCWV_input_file)
        else:
            spec_hu_file = f"modtran_atmprofiles_{mo}.txt"
            modtran_output_file = f"modtran_atmprofiles_{mo}.bts+tau+dbtdsst.txt"
            modtran = prep_retrieval(atmpath, modtran_output_file, spec_hu_file)
            modtran.to_csv(TCWV_input_file, index=False)
        
        # QC: Remove all ERA-5 atm profiles associated with sea ice (SST <- 1.688 C)
        modtran = modtran[modtran['Surface T[K]'] >= (-1.688 + 273.15)]

        print(f'{mo}: {modtran.shape[0]}')
        
        modtran_list.append(modtran)

        n = n + modtran.shape[0]
    
    # Concatenate the DataFrames for the three months in the window.
    modtran_lut = pd.concat(modtran_list, ignore_index=True)

    return modtran_lut,n

##########################

def derive_coeffs(atmpath,simTOA_transformer,simWV_transformer,simT_transformer):
    # Derive retrieval coefficiencts from MODTRAN files - 3 month rolling window
    months = ['01','02','03','04','05','06','07','08','09','10','11','12']
    atmcor = {}  # To store the regression results for each middle month.
    
    # Loop over months by index so we can get the previous and next month via modulo arithmetic.
    for i, middle_month in enumerate(months):
        # Determine the rolling window months: previous, current, and next (with wrap-around)
        prev_month = months[(i - 1) % 12]
        next_month = months[(i + 1) % 12]
        window_months = [prev_month, middle_month, next_month]
        
        print(f"Processing rolling window for middle month {middle_month}")
        
        modtran_lut,_ = concat_modtran_months(window_months,atmpath)
    
        modtran_lut_norm = modtran_lut
    
        modtran_lut_norm['Surface T[K]'] = simT_transformer.transform(modtran_lut[['Surface T[K]']])
        modtran_lut_norm['TOA T[K]'] = simTOA_transformer.transform(modtran_lut[['TOA T[K]']])
        modtran_lut_norm['TCWV [cm]'] = simWV_transformer.transform(modtran_lut[['TCWV [cm]']])
        
        # Run the regression using your derive_retrieval function on the concatenated DataFrame.
        retrieval_results = derive_retrieval(modtran_lut_norm)
        a1 = np.around(retrieval_results.params.toa, 2)
        a2 = np.around(retrieval_results.params.tcwv_toa, 2)
        a3 = np.around(retrieval_results.params.Intercept, 2)
        r2 = np.around(retrieval_results.rsquared, 2)
        
        pa1 = np.around(retrieval_results.pvalues[1], 3)
        pa2 = np.around(retrieval_results.pvalues[2], 3)
        pa3 = np.around(retrieval_results.pvalues[0], 3)
        
        # Store the regression coefficients and R2 for the middle month.
        atmcor[middle_month] = {"a1": a1, "a2": a2, "a3": a3}
        print(f"Rolling for month {middle_month}: toa = {a1}, tcwv_toa = {a2}, Intercept = {a3}, R2 = {r2}")
        print(f"p-values for month {middle_month}: toa = {pa1}, tcwv_toa = {pa2}, Intercept = {pa3}")
    
    print(retrieval_results.summary())
    
    return atmcor,retrieval_results,modtran_lut_norm

##########################

def derive_retrieval(modtran_lut):
    '''
    Derive the retrieval coefficients from the atmospheric column inputs and outputs to the MODTRAN
    model runs for Landsat using multiple regression. 
    
    Variables:
    modtran_lut = pandas dataframe that includes columns for surface temperature [K], top of 
                  atmosphere brightness temperature [K], total column water vapor [cm]
    
    Outputs:
    results = multiple regression summary and derived coefficients (ak) for retrieval atmospheric correction
    '''
    # Run OLS multiple regression to derive atmospheric correction coefficients
    df_newnames = modtran_lut.rename(columns={'Surface T[K]': 'surface', 'TOA T[K]': 'toa', 'TCWV [cm]': 'tcwv'})
    df_newnames['tcwv_toa'] = df_newnames['tcwv']*df_newnames['toa']
    results = smf.ols('surface ~ toa + tcwv_toa', data=df_newnames).fit()
 
    return results

##########################

def derive_retrieval_ransac(modtran_lut, residual_threshold=1.0, max_trials=100, random_state=42):
    """
    Derive the retrieval coefficients from the atmospheric column inputs and outputs to the MODTRAN
    model runs for Landsat using RANSAC regression.
    
    Variables:
    -----------
    modtran_lut : pandas DataFrame
        A dataframe that includes columns for surface temperature [K], top of atmosphere 
        brightness temperature [K], and total column water vapor [cm]. Expected column names are:
            'Surface T[K]', 'TOA T[K]', and 'TCWV [cm]'
    
    residual_threshold : float, default 1.0
        Maximum residual for a data point to be classified as an inlier.
        
    max_trials : int, default 100
        Maximum number of iterations for the RANSAC algorithm.
        
    random_state : int, default 42
        Random seed for reproducibility.
    
    Returns:
    --------
    results : dict
        Dictionary containing the estimated coefficients with keys:
            'Intercept', 'toa', 'tcwv_toa'
    """
    # Rename columns to standard names
    df_newnames = modtran_lut.rename(columns={
        'Surface T[K]': 'surface',
        'TOA T[K]': 'toa',
        'TCWV [cm]': 'tcwv'
    })

    # Remove rows with any NaN values
    df_newnames = df_newnames.dropna()
    
    # Create the interaction term
    df_newnames['tcwv_toa'] = df_newnames['tcwv'] * df_newnames['toa']
    
    # Prepare the predictor matrix and response vector
    X = df_newnames[['toa', 'tcwv_toa']]
    y = df_newnames['surface']
    
    # Set up the RANSAC regression with a base LinearRegression estimator
    base_estimator = LinearRegression()
    ransac = RANSACRegressor(estimator=base_estimator,
                             max_trials=max_trials,
                             residual_threshold=residual_threshold,
                             random_state=random_state)
    
    # Fit the model
    ransac.fit(X, y)
    
    # Extract the fitted parameters
    # The intercept and coefficients are stored in the underlying estimator.
    params = {
        'Intercept': ransac.estimator_.intercept_,
        'toa': ransac.estimator_.coef_[0],
        'tcwv_toa': ransac.estimator_.coef_[1]
    }
    
    return params


##########################

def derive_retrieval_odr(modtran_lut):
    """
    Derive the retrieval coefficients from the atmospheric column inputs and outputs 
    to the MODTRAN model runs for Landsat using Orthogonal Distance Regression.
    
    Parameters:
    -----------
    modtran_lut : pandas DataFrame
        A dataframe that includes columns for surface temperature [K],
        top-of-atmosphere brightness temperature [K], and total column water vapor [cm].
        Expected column names:
            'Surface T[K]', 'TOA T[K]', and 'TCWV [cm]'
    
    Returns:
    --------
    results : dict
        Dictionary containing the estimated coefficients:
            'Intercept', 'toa', and 'tcwv_toa'
    """
    # Rename columns to standard names
    df_newnames = modtran_lut.rename(columns={
        'Surface T[K]': 'surface',
        'TOA T[K]': 'toa',
        'TCWV [cm]': 'tcwv'
    })
    
    # Remove any rows that contain NaNs
    df_newnames = df_newnames.dropna()
    
    # Create the interaction term
    df_newnames['tcwv_toa'] = df_newnames['tcwv'] * df_newnames['toa']
    
    # Prepare the independent variables and the dependent variable.
    # For ODR, X must be an array of shape (n_predictors, n_points)
    X = df_newnames[['toa', 'tcwv_toa']].values.T  # shape: (2, n)
    y = df_newnames['surface'].values              # shape: (n,)
    
    # Define the linear model function for ODR.
    # beta[0] is the intercept, beta[1] is the coefficient for 'toa', and beta[2] for 'tcwv_toa'
    def linear_model(beta, x):
        return beta[0] + beta[1] * x[0] + beta[2] * x[1]
    
    # Create an ODR Model object
    model = odr.Model(linear_model)
    
    # Prepare the data for ODR. (If you have measurement errors, you can pass them via sx and sy)
    data = odr.RealData(X, y)
    
    # Create an ODR instance with an initial guess for the parameters.
    # Here we use an initial guess: [0.0, 1.0, 1.0]
    odr_instance = odr.ODR(data, model, beta0=[0.0, 1.0, 1.0])
    
    # Run the ODR regression.
    out = odr_instance.run()
    
    # Return the parameters in a dictionary similar to the statsmodels output.
    results = {
        'Intercept': out.beta[0],
        'toa': out.beta[1],
        'tcwv_toa': out.beta[2],
        # Optionally, you can also return diagnostics:
        'sum_square': out.sum_square,
        'res_var': out.res_var
    }
    
    return results

##########################

def spec_hu_to_tcwv(modtran_lut, modtran_atm, atm_levels=37):
    '''
    Calculate total column water vapor by integrating across all atmospheric pressure levels
    using hydrostatic approximation.

    Output:
    modtran_lut = original dataframe with added TCWV column in [cm]
    '''
    g = 9.80665  # gravity [m/s^2]

    m = 0
    for y in tqdm(range(modtran_lut.shape[0])):
        r = m + atm_levels
        df = modtran_atm.iloc[m:r]

        tcwv_pa = 0
        for i in range(1, len(df)):
            p0 = df['pressure [hPa]'].iloc[i-1] * 100  # convert to Pa
            p1 = df['pressure [hPa]'].iloc[i] * 100
            q0 = df['spec humidity [kg/kg]'].iloc[i-1]
            q1 = df['spec humidity [kg/kg]'].iloc[i]
            dq = (q0 + q1) / 2  # trapezoidal average
            dp = p1 - p0        # pressure difference
            tcwv_pa += dq * dp  # integral sum: q * dp

        # Final TCWV in kg/m^2 → mm (same numerically) → cm
        tcwv_kg_m2 = tcwv_pa / g
        tcwv_cm = tcwv_kg_m2 / 10

        modtran_lut.loc[y, 'TCWV [cm]'] = tcwv_cm
        m = r

    return modtran_lut


# In[29]:


# Functions to produce SST with atmospheric correction
'''
Functions to produce SST with atmospheric correction
apply_retrieval preps the masks and thermal data then runs the entire retrieval correction and calibration pipeline in lsatAtmCorr,
lsatAtmCorr calculates top of atmosphere brightness temperatures from thermal digital numbers data in TOA_BT and applies 
the atmospheric correction to get absolute temperatures [C] using retrieval.
'''

##########################

def TOA_BT(ls_thermal,scene):
    '''
    Calculate TOA radiance and brightness temperature using MTL json
    
    ls_thermal = xarray dataset of Landsat thermal data computed
    scene = catalog item for landsat scene
    
    Using equations from https://www.usgs.gov/landsat-missions/using-usgs-landsat-level-1-data-product
    '''
    
    # Calculate radiances using MTL data
    s3 = boto3.client("s3")

    # Extract bucket and key for json MTL file
    # Example: bucket = "usgs-landsat" ; key = "collection02/level-1/standard/oli-tirs/2019/002/113/LC08_L1GT_002113_20190206_20201016_02_T2/LC08_L1GT_002113_20190206_20201016_02_T2_MTL.json"
    s3_url = scene['MTL.json'].metadata['alternate']['s3']['href']
    bucket = s3_url.split('/')[2].strip()
    key = s3_url.split(bucket)[1].strip()[1:]

    # Get MLT data
    res = s3.get_object(Bucket=bucket, Key=key, RequestPayer="requester")
    MTL = res["Body"].read().decode("utf-8")

    # Get important constants from MTL
    ind = MTL.find('K1_CONSTANT_BAND_10')
    K1_10 = float(MTL[ind+23:ind+31])
    ind = MTL.find('K2_CONSTANT_BAND_10')
    K2_10 = float(MTL[ind+23:ind+32])
    ind = MTL.find('RADIANCE_MULT_BAND_10')
    ML10 = float(MTL[ind+25:ind+35])
    ind = MTL.find('RADIANCE_ADD_BAND_10')
    AL10 = float(MTL[ind+24:ind+31])

    # Mask no data 
    DN_masked = ls_thermal.where(ls_thermal != 0)

    # Top of Atmosphere radiance for Band 10
    Llambda = ML10 * DN_masked + AL10

    # Top of Atmosphere brightness temperature for Band 10
    T10 = K2_10 / np.log((K1_10 / Llambda) + 1)
    return T10

##########################

def retrieval(toa, wv, a1, a2, a3, simT_transformer, simTOA_transformer, simWV_transformer):
    '''
    Calculates the surface temperature.

    Variables:
    toa = calculated TOA [K] (xarray.DataArray)
    wv = total column water vapor (xarray.DataArray)
    ak = derived retrieval coefficients

    Output:
    SST = sea surface temperature [C] (xarray.DataArray)
    '''
    # Convert toa DataArray to numpy, preserving shape.
    toa_arr = toa.values
    original_toa_shape = toa_arr.shape
    # Reshape to a column vector for the transformer
    toa_norm = simTOA_transformer.transform(toa_arr.reshape(-1, 1))
    toa_norm = toa_norm.reshape(original_toa_shape)
    
    # Convert wv DataArray to numpy array and transform similarly.
    original_wv_shape = wv.shape
    wv_norm = simWV_transformer.transform(wv.reshape(-1, 1))
    wv_norm = wv_norm.reshape(original_wv_shape)
    
    # Calculate normalized SST using the retrieval coefficients and water vapor
    SST_norm = a3 + a1 * toa_norm + a2 * wv_norm * toa_norm
    
    # Inverse transform SST: again reshape as needed.
    original_sst_shape = SST_norm.shape
    SST_norm_flat = SST_norm.reshape(-1, 1)
    SST_flat = simT_transformer.inverse_transform(SST_norm_flat)
    SST = SST_flat.reshape(original_sst_shape) - 273.15
    
    # Optionally, convert the result back into an xarray.DataArray,
    # preserving the original toa coordinates and dimensions:
    SST = xr.DataArray(SST, coords=toa.coords, dims=toa.dims)
    
    return SST

##########################

def lsatAtmCorr(ls_thermal,scene,mask,modwv,a1,a2,a3,simT_transformer,simTOA_transformer,simWV_transformer):
    '''
    Applies atmospheric correction to top of atmosphere (TOA) brightness temperatures and converts to 
    absolute temperature for Landsat thermal images. Uses a derived coefficients for a non-linear
    sea surface temperature algorithm (retrieval) and water vapor from MODIS to produce the atmospheric 
    correction.

    Variables:
    ls_thermal = xarray dataset of Landsat thermal band computed 
    scene = STAC catalog item for the Landsat scene
    modwv = Water Vapor array (or other varrying parameter) from MODIS that has been 
            processed to the same dimensionality and pixel size as the Landsat image
    ak = derived retrieval coefficients 

    Previously defined:
    ak = derived retrieval coefficients

    Output: SST 2D array and GTiff of atmospheric corrected absolute temperatures [C]

    '''
    T10 = TOA_BT(ls_thermal,scene)
    T10 = mask * T10
    SST = retrieval(T10,modwv,a1,a2,a3,simT_transformer,simTOA_transformer,simWV_transformer)
    SST = SST.compute()
        
    return SST

##########################

def apply_retrieval(ls_thermal,scene,mask,WV_xr,atmcor,simT_transformer,simTOA_transformer,simWV_transformer):
    '''
    Use MODIS water vapor and landsat DN in retrieval algorithm to derive sea surface temperature.
    
    Variables:
    ls_scene = xarray dataset of Landsat scene  
    scene = STAC catalog item for the Landsat scene
    WV_xr = xarray dataarray of MODIS water vapor values matching timing of the landsat scene  
    atmcor = dictionary of derived retrieval coefficients for all months
    
    Outputs:
    SST = multiple regression summary and derived coefficients (ak) for retrieval atmospheric correction
    Also saves a cloud-optimized geotiff of SST
    '''
    try:
        wv2 = mask*WV_xr.values
        wv3 = mask*np.around(wv2,decimals=5) # usually w2 but skippping outliers for now

        means = np.nanmean(wv3)
        print (f'Mean water vapor value is: {means}, min: {np.nanmin(wv3)}, max: {np.nanmax(wv3)}')
        
        # Select appropriate atmospheric correction coefficients
        month = scene.metadata['datetime'].month
        a_mo = f'{month}'.zfill(2)
        a1 = atmcor[a_mo]['a1']
        a2 = atmcor[a_mo]['a2']
        a3 = atmcor[a_mo]['a3']
        
        # Apply atmospheric correction to get absolute temps in C
        SST = lsatAtmCorr(ls_thermal,scene,mask,wv3,a1,a2,a3,simT_transformer,simTOA_transformer,simWV_transformer)

        return SST
        
    except Exception as e: 
        print(e)
        print (f'atm correction of {ls_scene.id.values} failed')

