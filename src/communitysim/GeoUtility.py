""" Geo Utility functions """
from osgeo import ogr
from osgeo import osr
from osgeo import gdal, gdal_array

import numpy as np

def LatLonToUTM(
    latitude: float, 
    longitude: float, 
    inEPSG: int=4326, 
    outEPSG: int=26910):
    inSR = osr.SpatialReference()
    inSR.ImportFromEPSG(inEPSG)
    outSR = osr.SpatialReference()
    outSR.ImportFromEPSG(outEPSG)

    point = ogr.Geometry(ogr.wkbPoint)
    point.AddPoint(latitude, longitude)
    point.AssignSpatialReference(inSR)
    point.TransformTo(outSR)

    return point.GetX(), point.GetY()

def UTMArrayToGrid(inputArray, width: int, height: int):
    dataset = gdal_array.OpenArray(inputArray)
    gridOptions = gdal.GridOptions(width=width, height=height)
    gridOutput = gdal.Grid('gridOutput', dataset, options=gridOptions)

    return gridOutput

coords = np.array([[0,1],[0.5,1],[1.25,1]])

width = 3
height = 3

gridOutput = UTMArrayToGrid(coords, width, height)
print(gridOutput)