from .config import *

from . import access

"""These are the types of import we might expect in this file
import pandas
import bokeh
import seaborn
import matplotlib.pyplot as plt
import sklearn.decomposition as decomposition
import sklearn.feature_extraction"""

"""Place commands in this file to assess the data you have downloaded. How are missing values encoded, how are outliers encoded? What do columns represent, makes rure they are correctly labeled. How is the data indexed. Crete visualisation routines to assess the data (e.g. in bokeh). Ensure that date formats are correct and correctly timezoned."""


def data():
    """Load the data from access and ensure missing values are correctly encoded as well as indices correct, column names informative, date and times correctly formatted. Return a structured data structure such as a data frame."""
    df = access.data()
    raise NotImplementedError

def query(data):
    """Request user input for some aspect of the data."""
    raise NotImplementedError

def view(data):
    """Provide a view of the data that allows the user to verify some aspect of its quality."""
    raise NotImplementedError

def labelled(data):
    """Provide a labelled set of data ready for supervised learning."""
    raise NotImplementedError


"""
PRACTICAL 2 FUNCTIONS
"""

import geopy
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import osmnx as ox

# figure out how to make this work on import, for now added to bottom
# from access import refactor_osm_data, access_data

def count_pois_near_coordinates_simplified(latitude: float, longitude: float, tags: dict, pois_df, distance_km: float = 1.0) -> dict:
    """
    Count Points of Interest (POIs) near a given pair of coordinates within a specified distance.
    Args:
        latitude (float): Latitude of the location.
        longitude (float): Longitude of the location.
        tags (dict): A dictionary of OSM tags to filter the POIs (e.g., {'amenity': True, 'tourism': True}).
        pois_df (pd.DataFrame): DataFrame containing the POIs. I ADDED THIS.
        distance_km (float): The distance around the location in kilometers. Default is 1 km.
    Returns:
        dict: A dictionary where keys are the OSM tags and values are the counts of POIs for each tag.
    """

    # step 1: filter pois_df to only POIs within specified distance
    pois_df['distance'] = pois_df.apply(lambda x: geopy.distance.geodesic((latitude, longitude), (x.latitude, x.longitude)).km, axis=1)
    pois_df = pois_df[pois_df['distance'] <= distance_km]
    pois_df = pois_df.drop(columns=['distance'])

    # step 2: count POIs for each tag
    poi_counts = {}

    for tag in tags:
        if tag in pois_df.columns:
            poi_counts[tag] = pois_df[tag].notnull().sum()
    else:
        poi_counts[tag] = 0

    return poi_counts

def osm_counts_per_town_df(locations_dict: dict, pois_df, tags: dict):
    """Function to return OSM feature count dict given a dict of locations to check.
    :param locations_dict: dictionary of key location name, value lat/long tuple
    :param pois_df: DataFrame of OSM data
    :param tags: dict tags
    """
    osm_counts_per_town = {
        "town": []
    }

    for tag in tags.keys():
        osm_counts_per_town[tag] = []

    for location, (latitude, longitude) in locations_dict.items():
        osm_counts = count_pois_near_coordinates_simplified(latitude, longitude, tags, pois_df)
        osm_counts_per_town["town"].append(location)
        for tag in tags.keys():
            osm_counts_per_town[tag].append(osm_counts[tag] if tag in osm_counts else 0)

    osm_counts_df = pd.DataFrame.from_dict(osm_counts_per_town)
    return osm_counts_df

def osm_counts_for_town_df(location: str, latitude: float, longitude: float, pois_df, tags:dict):
    """Function to return OSM feature count dict given a dict of locations to check.
    :param location: location name
    :param latitude: float latitude
    :param longitude: float longitude
    :param pois_df: DataFrame of OSM data
    :param tags: dict of tags
    """
    osm_counts_for_town = {
        "town": []
    }

    for tag in tags.keys():
        osm_counts_for_town[tag] = []

    osm_counts = count_pois_near_coordinates_simplified(latitude, longitude, tags, pois_df)
    osm_counts_for_town["town"].append(location)
    for tag in tags.keys():
        osm_counts_for_town[tag].append(osm_counts[tag] if tag in osm_counts else 0)

    osm_counts_df = pd.DataFrame.from_dict(osm_counts_for_town)
    return osm_counts_df

def fetch_osm_given_places(locations_dict: dict, bound: float = 0.02):
    """ Fetch OSM feature count given places and their locations in a dictionary.
    :param locations_dict: dictionary of key location name, value lat/long tuple
    :param bound: length of bounding square in km for accessing place data
    """

    # ASSERTING there is at least one entry in locations_dict
    if not (len(locations_dict) > 0):
        raise ValueError("fetch_osm_given_places() must be fed a non-empty locations_dict.")

    first_place, (first_latitude, first_longitude) = list(locations_dict.items())[0]

    pois = access_data(first_latitude, first_longitude, bound=bound)
    pois_df = refactor_osm_data(pois)
    overall_osm_df = osm_counts_per_town_df(locations_dict, pois_df)

    print("-------------------------")
    print(f"{first_place} processed.")
    print("-------------------------")

    for place, (latitude, longitude) in locations_dict.items():

        if place != first_place:

            pois = access_data(latitude, longitude, bound=bound)
            pois_df = refactor_osm_data(pois)
            osm_df = osm_counts_for_town_df(place, latitude, longitude, pois_df)

            overall_osm_df_index = overall_osm_df[overall_osm_df["town"] == place].index
            overall_osm_df.iloc[overall_osm_df_index] = osm_df[osm_df["town"] == place]

            print("-------------------------")
            print(f"{place} processed.")
            print("-------------------------")

    return overall_osm_df

def augment_osm_data_with_latitude_longitude(osm_df, locations_dict: dict):
    """Augment a given df with latitudes and longitudes using locations_dict as a map.
    :param osm_df: DataFrame of OSM data
    :param locations_dict: dictionary of key location name, value lat/long tuple
    """
    latitudes = {location: latitude for location, (latitude, longitude) in locations_dict.items()}
    longitudes = {location: longitude for location, (latitude, longitude) in locations_dict.items()}

    osm_df["latitude"] = osm_df["town"].map(latitudes)
    osm_df["longitude"] = osm_df["town"].map(longitudes)

    return osm_df

def feature_correlations_with_london_distance(exploration_df):
    """
    Function to visualise correlation of features with distance of place from London.
    :param exploration_df: df with location data
    """

    # Corr matrices
    fig, ax = plt.subplots(figsize=(10,5))
    correlation_values = {
        "amenity": exploration_df["km_from_london"].corr(exploration_df['amenity']),
        "tourism": exploration_df["km_from_london"].corr(exploration_df['tourism']),
        "shop": exploration_df["km_from_london"].corr(exploration_df['shop'])
    }

    plt.bar(range(len(correlation_values)), list(correlation_values.values()), align='center')
    plt.xticks(range(len(correlation_values)), list(correlation_values.keys()))
    plt.xlabel("Feature")
    plt.ylabel("Correlation")
    plt.title("Correlation of features with distance from London in km")

    plt.show()

    # Scatter graph
    fig, ax = plt.subplots(figsize=(8,10))

    a, b = np.polyfit(exploration_df["km_from_london"], exploration_df['amenity'], 1)
    ax.scatter(exploration_df["km_from_london"], exploration_df['amenity'], color="blue", label="amenity")
    ax.plot(exploration_df["km_from_london"], a*exploration_df["km_from_london"]+b, color="blue")

    a, b = np.polyfit(exploration_df["km_from_london"], exploration_df['tourism'], 1)
    ax.scatter(exploration_df["km_from_london"], exploration_df['tourism'], color="green", label="tourism")
    ax.plot(exploration_df["km_from_london"], a*exploration_df["km_from_london"]+b, color="green")

    a, b = np.polyfit(exploration_df["km_from_london"], exploration_df['shop'], 1)
    ax.scatter(exploration_df["km_from_london"], exploration_df['shop'], color="red", label="shop")
    ax.plot(exploration_df["km_from_london"], a*exploration_df["km_from_london"]+b, color="red")


    plt.xlabel("Distance from London (km)")
    plt.ylabel("Count")
    plt.title("Count of features against distance from London in km")
    plt.legend()

    plt.show()

def plot_area_wrt_buildings(place_name: str, latitude: float, longitude: float, buildings_in_area_df, box_width: float, box_height: float, with_addr_color:str ="green", without_addr_color:str ="red"):
    """
    Function to plot a map of buildings in area, using color to distinguish buildings with and without addresses.
    :param place_name: str name of place
    :param latitude: float latitude
    :param longitude: float longitude
    :param buildings_in_area_df: dataframe of buildings
    :param box_width: float width of box in degrees
    :param box_height: float height of box in degrees
    """

    north = latitude + box_height/2
    south = latitude - box_width/2
    west = longitude - box_width/2
    east = longitude + box_width/2

    graph = ox.graph_from_bbox(north, south, east, west)

    # Retrieve nodes and edges
    nodes, edges = ox.graph_to_gdfs(graph)

    area = ox.geocode_to_gdf(place_name)

    fig, ax = plt.subplots()

    # Plot the footprint
    area.plot(ax=ax, facecolor="white")

    # Plot street edges
    edges.plot(ax=ax, linewidth=1, edgecolor="dimgray")

    ax.set_xlim([west, east])
    ax.set_ylim([south, north])
    ax.set_xlabel("longitude")
    ax.set_ylabel("latitude")

    # Plot all buildings with addresses in area as green, and without as red
    # aspect=1 is to debug ValueError: aspect must be finite and positive
    buildings_without_addresses = buildings_in_area_df[~((buildings_in_area_df['addr:housenumber'].isna()) & (buildings_in_area_df['addr:street'].isna()) & (buildings_in_area_df['addr:postcode'].isna()))]
    buildings_with_addresses = buildings_in_area_df[(buildings_in_area_df['addr:housenumber'].isna()) & (buildings_in_area_df['addr:street'].isna()) & (buildings_in_area_df['addr:postcode'].isna())]

    buildings_with_addresses.plot(ax=ax, color=with_addr_color, alpha=0.7, markersize=10, label="address present", aspect=1)
    buildings_without_addresses.plot(ax=ax, color=without_addr_color, alpha=0.7, markersize=10, label="address incomplete", aspect=1)

    plt.tight_layout()

"""
FROM OTHER MODULES
"""

def access_data(
        latitude: float, 
        longitude: float, 
        tags: dict = {
            "amenity": True,
            "buildings": True,
            "historic": True,
            "leisure": True,
            "shop": True,
            "tourism": True,
            "religion": True,
            "memorial": True
        }, 
        bound: float = 0.02
        ):
    """Function to access data from OSM given a latitude, longitude and bounding square length.
    :param latitude: latitude of centre of bounding square
    :param longitude: longitude of centre of bounding square
    :param bound: length of bounding square in degrees
    """
    box_width = bound
    box_height = bound
    north = latitude + box_height/2
    south = latitude - box_width/2
    west = longitude - box_width/2
    east = longitude + box_width/2

    pois = ox.geometries_from_bbox(north, south, east, west, tags)

    return pois

def refactor_osm_data(pois):
    """Function to refactor GeoDataFrame into DataFrame for usage.
    :param pois: GeoDataFrame of OSM data
    """
    pois_df = pd.DataFrame(pois)
    pois_df['latitude'] = pois_df.apply(lambda row: row.geometry.centroid.y, axis=1)
    pois_df['longitude'] = pois_df.apply(lambda row: row.geometry.centroid.x, axis=1)

    return pois_df