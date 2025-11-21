"""Geo Utility functions"""

import pyproj
from repast4py.space import BoundingBox as bb
from repast4py.space import ContinuousPoint as cpt


# Create a transformer object for the desired UTM zone
def latlon_to_utm(latitude, longitude) -> tuple:
    """Convert latitude and longitude to UTM coordinates.

    Arguments:
        latitude: Latitude in decimal degrees.
        longitude: Longitude in decimal degrees.
    Returns:
        A tuple of UTM coordinates.
    """
    utm_zone = int((longitude + 180) / 6) + 1
    proj_string = f"+proj=utm +zone={utm_zone} +ellps=WGS84 +datum=WGS84 +units=m +no_defs"
    transformer = pyproj.Transformer.from_crs("EPSG:4326", proj_string)

    # Convert latitude and longitude to UTM coordinates
    easting, northing = transformer.transform(latitude, longitude)
    return easting, northing


def pointInBounds(point: cpt, bounds: bb) -> bool:
    """Check if a point is within the bounds."""
    xInBounds = point.x >= bounds.xmin and point.x < (bounds.xmin + bounds.xextent)
    yInBounds = point.y >= bounds.ymin and point.y < (bounds.ymin + bounds.yextent)
    zInBounds = point.z == 0 or (point.z >= bounds.zmin and point.z < (bounds.zmin + bounds.zextent))

    return xInBounds and yInBounds and zInBounds
