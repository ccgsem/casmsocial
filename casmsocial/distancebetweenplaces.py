# -*- coding: utf-8 -*-
"""
Created on Mon Oct  4 14:05:37 2021

@author: GMAAYAN
"""

import argparse
import pandas as pd
import numpy as np
import sys
import utm
from math import sqrt
from scipy.spatial import distance

import time


def dist_cartesian(x1, y1, x2, y2):
    dx = x2 - x1
    dy = y2 - y1
    
    return sqrt(dx*dx + dy*dy)

def distance_coords(coord1, coord2):
    return dist_cartesian(coord1[0], coord1[1], coord2[0], coord2[1])

def getPlaceIDAndCoordsFromFile(file):
    df = pd.read_csv(file)
    
    trimmed_df = df[['sp_id','latitude','longitude']]
    return trimmed_df

def getListOfNearbyPlaces(p1_id, coord1, places, dist):
    res = []
    for p_id, coord2 in places.items():
        d = distance(coord1, coord2)
        if d <= dist and p_id != p1_id:
            res.append(p_id)
            
    return res
    
    
    
def makeDictOfPlacesInDistance(places, dist):
    res = {p_id: getListOfNearbyPlaces(p_id, coord, places, dist) 
           for p_id, coord in places.items()}
    
    return res
    
def printNearbyPlaces2(outfile, places_in_distance):
    with open(outfile, 'w') as f:
        for p_id, other_places in places_in_distance.items():
            line = "{},{}\n".format(p_id, ':'.join(other_places))
            f.write(line)
            
def getUTM(lat, lon):
    x, y, z, u = utm.from_latlon(lat, lon)
    return (x, y)

def getAllDistances(coords):
    return distance.cdist(coords, coords, 'euclidean')

def findPlacesInDistance(pids, distances, max_dist):
    res = {pid: pidsInDistance(pid, pids, dists, max_dist) for pid, dists in zip(pids, distances)}
    return res

def pidsInDistance(pid, pids, dists, max_dist):
    res = []
    for pid2, dist in zip(pids, dists):
        if pid2 != pid and dist <= max_dist:
            res.append(pid2)
    return res

def placesInRange(X, Y, places, max_dist):
    # start_time = time.time()
    places['dX'] = places['X'] - X
    places['dY'] = places['Y'] - Y

    places['dist'] = np.sqrt((places['dX']**2) + (places['dY']**2))

    inRange = places.loc[places['dist'] <= max_dist, 'sp_id'].tolist()

    # print("This took {}ms to run".format((time.time() - start_time)*1000))
    return inRange

def printNearbyPlaces(outfile, sp_id, other_places):
    line = "{},{}\n".format(sp_id, ':'.join(str(other_id) for other_id in other_places))
    outfile.write(line)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Takes list of places and '
                                     'finds the distance between them.')
    parser.add_argument('-i', action='append', required=True)
    parser.add_argument('-o', required=True)
    parser.add_argument('-d')
    args = parser.parse_args()
    
    place_df = pd.DataFrame()
    for f in args.i:
        place_df = place_df.append(getPlaceIDAndCoordsFromFile(f))
        
    if len(place_df) == 0:
        sys.exit(1)
    
    place_df = place_df.astype({'sp_id': int, 'latitude': float, 'longitude': float})
    pids = place_df['sp_id']
    coords = [getUTM(lat, lon) for lat, lon in zip(place_df['latitude'], place_df['longitude'])]
    xs, ys = zip(*coords)
    place_df['X'] = xs
    place_df['Y'] = ys
    place_df = place_df.drop('latitude', 1)
    place_df = place_df.drop('longitude', 1)
    
    # place_dict = {row[0]: getUTM(row[1], row[2]) for row in 
    #               zip(place_df['sp_id'], place_df['latitude'], place_df['longitude'])}
    
    max_dist = args.d if args.d is not None else 500
    
    print("Starting to find distances.")

    with open(args.o, 'w') as outfile:
        outfile.write('sp_id,nearby_places\n')
        for row in zip(place_df["sp_id"], place_df["X"], place_df["Y"]):
            start_time = time.time()
            printNearbyPlaces(outfile, row[0], placesInRange(row[1],row[2],place_df, max_dist))
            print("One iteration took {}ms".format((time.time() - start_time)*1000))
    # places_in_destance = {row[0]: placesInRange(row[1], row[2], place_df, max_dist) for row in zip(place_df["sp_id"], place_df["X"], place_df["Y"])}

    # distances = getAllDistances(coords)

    # assert len(distances) == len(pids)

    # places_in_distance = findPlacesInDistance(pids, distances, max_dist)

    # places_in_distance = makeDictOfPlacesInDistance(place_dict, dist)
    
    print("Found distances. Printing to file.")
    
    # printNearbyPlaces(args.o, places_in_distance)
             
