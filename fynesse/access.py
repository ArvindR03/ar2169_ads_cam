from .config import *

"""These are the types of import we might expect in this file
import httplib2
import oauth2
import tables
import mongodb
import sqlite"""

import requests
import pymysql
import csv

# This file accesses the data

"""Place commands in this file to access the data electronically. Don't remove any missing values, or deal with outliers. Make sure you have legalities correct, both intellectual property and personal data privacy rights. Beyond the legal side also think about the ethical issues around this data. """

def data():
    """Read the data from the web or local file, returning structured format such as a data frame"""
    raise NotImplementedError

def hello_world():
  """Helper hello world function."""
  print("Hello from the data science library!")

def download_price_paid_data(year_from, year_to):
    """
    Download the UK Price Paid data in chunks.
    :param year_from: starting year (inclusive).
    :param year_to: ending year (inclusive).
    """
    # Base URL where the dataset is stored 
    base_url = "http://prod.publicdata.landregistry.gov.uk.s3-website-eu-west-1.amazonaws.com"
    """Download UK house price data for given year range"""
    # File name with placeholders
    file_name = "/pp-<year>-part<part>.csv"
    for year in range(year_from, (year_to+1)):
        print (f"Downloading data for year: {year}")
        for part in range(1,3):
            url = base_url + file_name.replace("<year>", str(year)).replace("<part>", str(part))
            response = requests.get(url)
            if response.status_code == 200:
                with open("." + file_name.replace("<year>", str(year)).replace("<part>", str(part)), "wb") as file:
                    file.write(response.content)

def create_connection(user, password, host, database, port=3306):
    """ Create a database connection to the MariaDB database
        specified by the host url and database name.
    :param user: username
    :param password: password
    :param host: host url
    :param database: database name
    :param port: port number
    :return: Connection object or None
    """
    conn = None
    try:
        conn = pymysql.connect(user=user,
                               passwd=password,
                               host=host,
                               port=port,
                               local_infile=1,
                               db=database
                               )
        print(f"Connection established!")
    except Exception as e:
        print(f"Error connecting to the MariaDB Server: {e}")
    return conn

def housing_upload_join_data(conn, year):
  """
  Upload housing data using a given connection for a given year.
  :param conn: connection
  :param year: year int
  """
  start_date = str(year) + "-01-01"
  end_date = str(year) + "-12-31"

  cur = conn.cursor()
  print('Selecting data for year: ' + str(year))
  cur.execute(f'SELECT pp.price, pp.date_of_transfer, po.postcode, pp.property_type, pp.new_build_flag, pp.tenure_type, pp.locality, pp.town_city, pp.district, pp.county, po.country, po.latitude, po.longitude FROM (SELECT price, date_of_transfer, postcode, property_type, new_build_flag, tenure_type, locality, town_city, district, county FROM pp_data WHERE date_of_transfer BETWEEN "' + start_date + '" AND "' + end_date + '") AS pp INNER JOIN postcode_data AS po ON pp.postcode = po.postcode')
  rows = cur.fetchall()

  csv_file_path = 'output_file.csv'

  # Write the rows to the CSV file
  with open(csv_file_path, 'w', newline='') as csvfile:
    csv_writer = csv.writer(csvfile)
    # Write the data rows
    csv_writer.writerows(rows)
  print('Storing data for year: ' + str(year))
  cur.execute(f"LOAD DATA LOCAL INFILE '" + csv_file_path + "' INTO TABLE `prices_coordinates_data` FIELDS TERMINATED BY ',' OPTIONALLY ENCLOSED by '\"' LINES STARTING BY '' TERMINATED BY '\n';")
  print('Data stored for year: ' + str(year))

  conn.commit()

"""
PRACTICAL 2 FUNCTIONS
"""

import osmnx as ox
import pandas as pd

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

# convert this to pymysql

# def get_data_from_db_within_sqkm(longitude: float, latitude: float, year_from:int = 2020, year_to:int = 2024):
#     """
#     Function to query all houses in a 1km x 1km region around the centre of Cambridge that have been part of housing transactions since 2020.
#     :param longitude: float longitude
#     :param latitude: float latitude
#     :param year_from: int year from
#     :param year_to: int year to (inclusive)
#     """

#     query_string = f"""
#     USE `ads_2024`;
#     select * from
#     (select * from `postcode_data` where longitude BETWEEN {longitude - 0.005} AND {longitude + 0.005} and latitude between {latitude - 0.005} and {latitude + 0.005})
#     as po
#     inner join pp_data as pp
#     on po.postcode = pp.postcode
#     where pp.date_of_transfer BETWEEN '{year_from}-01-01' AND '{year_to}-12-31';
#     """

#     pp_query = %sql $query_string
#     pp_df = pd.DataFrame(pp_query)
#     return pp_df

def get_buildings_in_area_from_osm(latitude: float, longitude: float, bound:float = None):
    """
    Function to get information about buildings in a given area from OpenStreetMap.
    :param latitude: float latitude
    :param longitude: float longitude
    :param bound: float bounding box size in degrees, defaults inner functions to 1km width
    """

    tags = {
        "addr": ["housenumber", "street", "postcode"],
        "building": True,
        "geometry": True
    }
    if bound:
        pois_df = access_data(latitude, longitude, bound=bound, tags=tags)
    else:
        pois_df = access_data(latitude, longitude, tags=tags)

    # add their square metre area to the df
    pois_df["area_sq_m"] = pois_df.to_crs(epsg=32630)["geometry"].area

    return pois_df

def match_pp_osm_dfs(pp_df, osm_df):
    """
    Function to merge UK Price Paid and OSM dataframes.
    :param pp_df: df UKPP data
    :param osm_df: df OSM data
    """

    # remove duplicate columns
    pp_df = pp_df.loc[:, ~pp_df.columns.duplicated()]

    # drop nans for necessary columns
    osm_df = osm_df.dropna(subset=["addr:street", "addr:postcode", "addr:housenumber"])
    pp_df = pp_df.dropna(subset=["street", "postcode", "primary_addressable_object_name"])

    # remove uppercase and spaces from street names
    osm_df["addr:street"] = osm_df["addr:street"].str.lower().str.replace(" ", "")
    pp_df["street"] = pp_df["street"].str.lower().str.replace(" ", "")

    # remove spacing and make postcode uppercase
    osm_df["addr:postcode"] = osm_df["addr:postcode"].str.upper().str.replace(" ", "")
    pp_df["postcode"] = pp_df["postcode"].str.upper().str.replace(" ", "")

    # remove spacings in house numbers if present
    osm_df["addr:housenumber"] = osm_df["addr:housenumber"].str.replace(" ", "")
    pp_df["primary_addressable_object_name"] = pp_df["primary_addressable_object_name"].str.replace(" ", "")
    pp_df["secondary_addressable_object_name"] = pp_df["secondary_addressable_object_name"].fillna("").str.replace(" ", "")

    # checking with priority to primary then secondary
    pp_df["housenumber"] = pp_df.apply(lambda row: row["secondary_addressable_object_name"] if row["secondary_addressable_object_name"].isdigit() else row["primary_addressable_object_name"], axis=1)

    output_df = pd.merge(osm_df, pp_df, left_on=["addr:postcode", "addr:street", "addr:housenumber"], right_on=["postcode", "street", "housenumber"], how="inner")
    
    return output_df