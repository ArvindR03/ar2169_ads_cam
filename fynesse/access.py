from .config import *

import requests
import pymysql
import csv
import zipfile
import os
import osmnx as ox
import pandas as pd
import io
import geopandas as gpd
import osmium
import math
from shapely import wkt
import json
import numpy as np
from shapely.geometry import Point
import yaml
import string
import geojson
from shapely.geometry import Polygon, Point
from rapidfuzz import fuzz
from rapidfuzz.process import extract
import random
import topojson as tp

"""
FUNCTIONS FOR THE MAIN ADS ASSESSMENT
"""

# -------------
# TASK 1
# 1.1) functions
# -------------

# DONE
def download_census_data(code, base_dir=''):
    """
    TAKEN FROM PREVIOUS PRACTICAL.
    Function to download 2021 census data from nomisweb.co.uk given the TS code.
    
    Example usage:
    `download_census_data('TS007')`

    :param code: string TS code e.g. 'TS007'
    :param base_dir: alternative base directory for local storage of census files
    """
    url = f'https://www.nomisweb.co.uk/output/census/2021/census2021-{code.lower()}.zip'
    extract_dir = os.path.join(base_dir, os.path.splitext(os.path.basename(url))[0])

    if os.path.exists(extract_dir) and os.listdir(extract_dir):
        print(f"Files already exist at: {extract_dir}.")
        return

    os.makedirs(extract_dir, exist_ok=True)
    response = requests.get(url)
    response.raise_for_status()

    with zipfile.ZipFile(io.BytesIO(response.content)) as zip_ref:
        zip_ref.extractall(extract_dir)

    print(f"Files extracted to: {extract_dir}")

# DONE - helper
def csv_url_to_dataframe(url):
    """
    Helper function.
    Given a url to a csv file, download it (if not local) and return it as a pandas dataframe.
    :param url: string url
    :returns: the DataFrame
    """
    df = pd.read_csv(url)
    return df

# DONE
def fetch_files(url, local_filename, chunk_size=1048576):
    """
    Given an internet url to a file/folder destination, download it into local memory.
    Use streaming in requests module to allow large files e.g. ~1.5GB osm data.
    
    Example usage:
    `fetch_files("some url", "elections/election.csv")`

    :param url
    :param local_filename
    :param chunk_size: size to write to memory each time (default 1MB)
    """
    response = requests.get(url, stream=True) # we write stream to allow large files
                                              # such as the osm.pbf file
    print("Files successfully fetched.")

    packet_number = 1
    with open(local_filename, 'wb') as file:
        for packet in response.iter_content(chunk_size):
            file.write(packet)
            # print packet number every 100, and inc counter (just for printing)
            if packet_number % 100 == 0:
                print(f"{packet_number} packets written.")
            packet_number += 1

    print("Success.")

# DONE
def unzip_folder(folder_loc, destination_loc):
    """
    Unzip a local folder to a destination. (Similar to practical method.)
    
    :param folder_loc: place of zip folder
    :param destination_loc: place of destination extraction
    """
    with zipfile.ZipFile(folder_loc, 'r') as z:
        z.extractall(destination_loc)

# DONE - helper
def get_processes(conn):
    """
    Helper function.
    Get processes running on the cloud db.
    To analyze if database is throttling.
    :param conn
    """
    cur = conn.cursor()
    cur.execute("SHOW FULL PROCESSLIST;")
    processes = cur.fetchall()
    print([i for i in processes])

# DONE - helper
def kill_processes(processes, conn):
    """
    Helper function.
    Kill off processes given ids.
    Helps mitigate throttling.
    :param processes: list of int ids.
    :param conn
    """
    cur = conn.cursor()
    for process in processes:
        cur.execute(f"KILL {process};")

# -------------
# TASK 1
# 1.2) functions
# -------------

# DONE - helper
def get_credentials_from_yaml(filename="credentials.yaml"):
    """
    Helper function.
    Fetch and save credentials into variables from a given file.
    :param filename
    :returns: the username, poassword, url and port strings
    """
    with open(filename) as file:
        credentials = yaml.safe_load(file)
        username = credentials["username"]
        password = credentials["password"]
        url = credentials["url"]
        port = credentials["port"]
    return username, password, url, port

# DONE - helper
def save_credentials(username, password, url, port):
    """
    Helper function.
    Save credentials into variables from function call.
    :param username
    :param password
    :param url
    :param port
    :returns: params in order
    """
    return username, password, url, port

# DONE
def create_connection_pymysql(user, password, host, database, port=3306):
    """ Create a database connection to the MariaDB database
        specified by the host url and database name.
    TAKEN FROM PREVIOUS PRACTICAL
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

# DONE
def terminate_connection_pymysql(conn):
    """
    Given a PYMYSQL connection, terminate the connection safely.
    :param conn
    """
    conn.close()
    print("Connection terminated!")

# DONE
def load_census_data(code, level='msoa'):
    """
    TAKEN FROM PREVIOUS PRACTICAL.
    Function to read in local census CSV files, and return a pandas DataFrame.
    :param code: string TS code e.g. 'TS007'
    :param level: the specified level to read (CSV).
    :returns: the dataframe containing census data.
    """
    return pd.read_csv(f'census2021-{code.lower()}/census2021-{code.lower()}-{level}.csv')

# DONE - helper
def get_geodataframe(folder_name):
    """
    Helper function.
    Given a location of a file, construct and return a GeoDataFrame.
    :param folder_name: name of the local folder
    :returns: the GeoDataFrame object
    """
    return gpd.read_file(folder_name)

# DONE
def preprocess_ts007_census_dataframe(ts_007_df, normalize=True):
    """
    Preprocess raw census data in dataframe, and return cleaned dataframe with processed columns.
    INSPIRED BY PREVIOUS PRACTICAL.

    :param age_df
    :param normalize: boolean to whether or not normalize row counts to make them proportion
    :returns: cleaned DataFrame
    """
    # Preparing the columns we want
    ts_007_df = ts_007_df.drop(ts_007_df.columns[[0,3,4,10,16,23,28,34,45,61,77,88,99,115]], axis=1).set_index('geography')
    columns = ["geography code"] + [str(i) for i in range(100)]
    ts_007_df.columns = columns

    if normalize:
        columns_to_normalize = columns.copy()
        columns_to_normalize.remove("geography code")
        ts_007_df['sum'] = ts_007_df[columns_to_normalize].sum(axis=1)
        ts_007_df[columns_to_normalize] = ts_007_df[columns_to_normalize].div(ts_007_df['sum'], axis=0)
        ts_007_df = ts_007_df.drop(columns=['sum'])

    return ts_007_df

# DONE
def preprocess_ts062_census_dataframe(ts_062_df, normalize=True):
    """
    Preprocess raw census data in dataframe, and return cleaned dataframe with processed columns.
    :param age_df
    :param normalize: boolean to whether or not normalize row counts to make them proportion
    :returns: cleaned DataFrame
    """
    ts_062_df = ts_062_df.drop(ts_062_df.columns[[0]], axis=1).set_index('geography')
    columns = ["geography code","total","l1-3","l4-6","l7","l8-9","l10-11","l12","l13","l14","l15"]
    ts_062_df.columns = columns

    if normalize:
        columns_to_normalize = columns.copy()
        columns_to_normalize.remove("geography code")
        ts_062_df[columns_to_normalize] = ts_062_df[columns_to_normalize].div(ts_062_df['total'], axis=0)

        ts_062_df = ts_062_df.drop(columns=['total']) # as the total becomes redundant when we normalize within rows

    return ts_062_df

# DONE
def preprocess_ts066_census_dataframe(ts_066_df, normalize=True):
    """
    Preprocess raw census data in dataframe, and return cleaned dataframe with processed columns.
    :param age_df
    :param normalize: boolean to whether or not normalize row counts to make them proportion
    :returns: cleaned DataFrame
    """
    ts_066_df = ts_066_df.drop(ts_066_df.columns[[0,6,7,8,9,10,11,12,13,14,18,19,20,21,22,23,24,25,26]], axis=1).set_index('geography')
    columns = [
       'geography code',
       'total',
       'active (exc students)',
       'active (exc students): in employment',
       'active (exc students): unemployed',
       'active students',
       'active students: in employment',
       'active students: unemployed',
       'inactive',
       'inactive: retired',
       'inactive: student',
       'inactive: looking after home or family',
       'inactive: long-term sick or disabled',
       'inactive: other'
    ]
    ts_066_df.columns = columns

    if normalize:
        columns_to_normalize = columns.copy()
        columns_to_normalize.remove("geography code")
        ts_066_df[columns_to_normalize] = ts_066_df[columns_to_normalize].div(ts_066_df['total'], axis=0)

        ts_066_df = ts_066_df.drop(columns=['total']) # as the total becomes redundant when we normalize within rows

    return ts_066_df

# DONE
def preprocess_geo_coordinates_geo_df(geo_df, epsg_to=4326):
    """
    Given fetched geo-coordinates data for census output areas, preprocess the data.
    :param df: fetched data in pandas dataframe
    :param epsg_to: the crs that we want
    :returns; preprocessed GeoDataFrame
    """
    col_rename = {
        'OA21CD': 'coa_code',
        'LSOA21CD': 'super_coa_code',
        'LSOA21NM': 'super_coa_name',
        'LSOA21NMW': 'super_coa_name_welsh',
        'BNG_E': 'bng_east',
        'BNG_N': 'bng_north',
        'GlobalID': 'global_id',
        'SHAPE_Length': 'shape_length',
        'SHAPE_Area': 'shape_area',
        "geometry": "wkt_geom"
    }
    geo_df = geo_df.rename(columns=col_rename)
    # geo_df.dropna(subset=['coa_code', 'wkt_geom', 'shape_area'])

    # note here: mariadb supports spatial data with wkt geometry types for spatial lookup

    # removing special chars from global_id, and casting relevant cols to string just as precaution
    for col in ['coa_code', 'super_coa_code', 'super_coa_name', 'super_coa_name_welsh']:
        geo_df[col] = geo_df[col].astype(str)
    geo_df['global_id'] = geo_df['global_id'].apply(lambda x: str(x).replace("{", "").replace("}", "").replace("'", ""))

    # setting geometry and ensureing same coordinate ref system
    geo_df = geo_df.set_geometry("wkt_geom")
    geo_df = geo_df.to_crs(epsg=epsg_to)
    
    return geo_df

# DONE - helper
class PBFHandler(osmium.SimpleHandler):
    """
    Helper class.
    Convert .osm.pbf to .csv, using the help of the osmium module.
    Inspired by approach to iterate through nodes in this stack overflow article:
    https://stackoverflow.com/questions/45771809/how-to-extract-and-visualize-data-from-osm-file-in-python
    """

    def __init__(self, csv_loc, chunk_size=100000):
        """
        Initialize local variables.
        :param destination_loc: destination to write data to
        :param chunk_size: corresponds to number of nodes to write at a time
        """
        # call osmium init
        super(PBFHandler, self).__init__()
        self.data = [] # we read data into here
        self.csv_loc = csv_loc
        self.chunk_size = chunk_size # chunk lim
        self.chunks_completed = 0 #were using this to print and keep track of headers

    def write_csv(self):
        """
        Write a chunk of data to local csv, and clear inner data.
        """
        mode='w'
        # add headers if not done yet, otherwise mode is append
        if len(self.chunks_completed) > 0:
            mode = 'a'
        with open(self.csv_loc, mode, newline='') as file:
            # use this to align values with headers
            w = csv.DictWriter(file, fieldnames=['id', 'latitude', 'longitude', 'tags'])
            # write the header if its the first chunk
            if self.chunks_completed == 0:
                w.writeheader()
            w.writerows(self.data) # now write the data
        
        # and finally clear all the chunks.
        self.data = []
        self.chunk_size = 0
        self.chunks_completed += 1
        print(f"Chunk {self.chunks_completed} written to csv.")

    def node(self, node):
        """
        Given a node, append relevant data to inner data.
        This function overwrites inner implementation called iteratively by apply_file.
        :param node: 'osmium.osm.types.Node'
        """
        # we filter out empty tags here to speed up search
        l = len(dict(node.tags).keys())
        if l > 0:
            n = {
                'id': node.id,
                'latitude': node.location.lat,
                'longitude': node.location.lon,
                'tags': json.dumps(dict(node.tags))
            }
            self.data.append(n)
            self.current_chunk_size += 1
            # process chunk if full and if we added a chunk
            if len(self.data) == self.chunk_size:
                self.write_csv()

# DONE
def convert_osm_pbf_into_csv(filename, destination_loc, chunk_size=100000):
    """
    Given a .osm.pbf file, load into a local csv file. We do this by instantiating class and applying it.
    
    Example usage:
    `convert_osm_pbf_into_csv("somefile.osm.pbf", "somefile.csv")`

    :param filename: loc of .pbf file
    :param destination_loc: loc of csv to write to
    :param chunk_size: number of nodes to write at a time
    """
    h = PBFHandler(csv_loc=destination_loc, chunk_size=chunk_size)
    h.apply_file(filename) # inbuilt function of the class from osmium
    if len(h.data) != 0:
        h.write_csv()

# DONE
def get_csv_headers_for_osm_large_df_preprocessing(osm_large_df, print_every = 100000):
    """
    Given a non-preprocessed DF, return all the possible features by looking through tags.
    
    Example usage:
    `headers = get_csv_headers_for_osm_large_df_preprocessing(osm_large_df)`
    
    :param osm_large_df
    :param print_every
    :returns: list of string headers
    """
    headers = set([])
    l = len(osm_large_df)
    i = 0
    c = 1

    print(f"{math.ceil(l / print_every)} chunks to process...")

    for _, row in osm_large_df.iterrows():
        tags = list(json.loads(row['tags']).keys())
        filtered_tags = []

        for tag in tags:
            if (tag[0].isupper()) or (tag[0].isdigit()) or (tag[0] == "_") or ((":" in tag) and not (tag == "addr:housenumber" or tag == "addr:street")):
                # tags.remove(tag)
                pass
            else:
                filtered_tags.append(tag)

        tags = filtered_tags

        # filter here as advised to skip nodes with less than 6 tags
        # UNLESS they include 'addr:housenumber' and 'addr:street'
        if len(str(tags)) >= 6 or (('addr:housenumber' in tags) and ('addr:street' in tags)):
            for tag in tags:
                headers.add(tag)
        i += 1
        if (i-1) % print_every == 0:
            print(f"Preprocessed chunk {c}.")
            c += 1

    start_cols = list(osm_large_df.columns)
    start_cols.remove('tags')

    return start_cols + sorted(list(headers))

# DONE
def write_preprocessed_osm_to_csv(osm_large_df, filename, headers, chunksize=10000):
    """
    Given osm data in df (large data), write in chunks to file, preprocessing data.
    Remove tags that have less than 2 values.
    
    Example usage:
    `write_preprocessed_osm_to_csv(osm_large_df, "savehere.csv", ["h1", "h2", ...])`
    
    :param osm_large_df
    :param filename: csv to write to
    :param headers: list of headers to frame new csv with
    :param chunksize: number of rows to add at a time to memory csv
    """
    l = len(osm_large_df)
    chunk_number = math.ceil(l / chunksize)

    # remove duplicates, and id, lat and long in array:
    headers = list(dict.fromkeys(headers))
    for h in set(['id', 'latitude', 'longitude']):
        if h in headers:
            headers.remove(h)

    print(f"Beginning preprocessing and writing to memory, {chunk_number+1} chunks.")

    # manually write the headers to the file
    with open(filename, mode='w', newline='') as file:
        w = csv.writer(file)
        w.writerow(['id', 'latitude', 'longitude'] + headers)

    # then iterate over chunks, adding each at a time
    for chunk_index in range(chunk_number):
        start = chunk_index * chunksize
        end = min((chunk_index + 1) * chunksize, l)

        # getting the chunk data
        chunk = osm_large_df.iloc[start:end]
        chunkdata = []

        # iterate over the chunk and add the data to data
        for _, row in chunk.iterrows():
            rowdata = {
                'id': row['id'],
                'latitude': row['latitude'],
                'longitude': row['longitude']
            }

            tags = json.loads(row['tags']) # load it as a dict in python
            for h in headers:
                rowdata[h] = np.nan if (h == "id" or h == "latitude" or h == "longitude" or h not in tags) else tags[h]
                # note here, just making sure we do not read in any id columns as there are dupes

            number_of_values_present = 0
            for h in headers:
                if rowdata[h] != None and rowdata[h] != np.nan:
                    number_of_values_present += 1

            # i.e. only add a row if it has more than or equal to 2 tags
            if number_of_values_present > 1:
                chunkdata.append(rowdata)

        # only bother if there was actually data in the chunk
        if chunkdata:
            processed_df = pd.DataFrame(chunkdata)
            processed_df.to_csv(filename, mode='a', index=False, header=False)

        print(f"Chunk {chunk_index+1} written to memory (csv).")

# DONE
def get_chosen_headers():
    """
    Return headers chosen from eliminating many irrelevant features and subfeatures of other features
    from the dataset.
    :returns: list of headers
    """
    # looked through the dataset and found a lot of irrelevant features that increase size, so
    # limit headers to reduce size of dataset and choose features that could be important
    # for tasks 1 and 2; was lenient enough to still include features that may not be related
    # but reduced dataset size by roughly ~100 times (previous csv was 29.66GB)
    headers = [
        "academic", "access", "accommodation", "accountant", "addr:housenumber", 
        "addr:street", "alcohol", "amenity", "animal", "bar", "barber", "basketball", 
        "beauty", "bmx", "boat", "books", "box_junction", "braille", "branch", "brand", 
        "brewery", "bridge", "bridleway", "building", "business", "butcher", "cafe", 
        "camera", "capital", "car", "car_service", "cash_withdrawal", "cemetery", 
        "charity", "child", "children", "cinema", "city", "climbing", "clothes", 
        "communication", "company", "constituency", "constriction", "consulate", 
        "courthouse", "currency", "dock", "doctors", "eat_in", "education", "embassy", 
        "fitness_centre", "flag", "franchise", "golf", "government", "goods", "health", 
        "healthcare", "highway", "historic", "industrial", "kiosk", "knowledge", 
        "laboratory", "memorial", "office", "parking", "pharmacy", "power", 
        "public_building", "public_transport", "recycling", "sauna", "school", 
        "sculpture", "sensor", "service", "shop", "signals", "sinkhole", "station", 
        "swimming_pool", "table_tennis", "toilets", "tourism", "underground", "vehicle", 
        "veteran", "website", "youth_centre", "youth_club", "wreck", "zoo"
    ]
    return headers

# DONE
def set_crs_for_gdfs(gdfs, gdf_geometries, epsg=4326):
    """
    Check and cast to correct coordinate reference system all given GDFs in a list.
    
    Example usage:
    `gdfs = set_crs_for_gdfs([gdf1, gdf2], ["geom", "wkt_geom"])`
    
    :param gdfs: list of geodataframes
    :param gdf_geometries: geometry column names
    :param epsg
    :returns: list of GeoDataFrames
    """
    results = []
    for i in range(len(gdfs)):
        gdf = gdfs[i]
        geom_col = gdf_geometries[i]

        # if there is no geometry, set it first, otherwise translate the crs
        if gdf.crs is None:
            gdf = gdf.set_geometry(geom_col)
            gdf.set_crs(epsg=epsg)
        else:
            gdf.to_crs(epsg=epsg)

        results.append(gdf)
    return results

# DONE
def gdf_from_osm_df(osm_pp_df):
    """
    Given loaded osm data from preprocessed csv, return the geodataframe, making new geometry
    of Point from spahely's wkt, used by geodataframes.
    :param osm_pp_df
    :returns: GeoDataFrame equivalent
    """
    osm_pp_df['geometry'] = osm_pp_df.apply(lambda x: Point(x['longitude'], x['latitude']), axis=1)
    osm_gdf = gpd.GeoDataFrame(osm_pp_df, geometry='geometry')
    osm_gdf.set_crs(epsg=4326, inplace=True)

    return osm_gdf

# DONE
def join_coa_code_to_osm_data(osm_gdf, geo_gdf, epsg=4326):
    """
    Given osm and geo data, spatially join to add coa code to osm data.
    :param osm_gdf
    :param geo_gdf
    :param epsg
    :returns: GeoDataFrame of OSM features with attached geography codes
    """
    # then make sure that the geometry is correct for geo_gdf
    geo_gdf = geo_gdf.set_geometry("wkt_geom")
    
    # make sure that the epsg are the same
    if osm_gdf.crs is None:
        osm_gdf = osm_gdf.set_crs(epsg=epsg, allow_override=True)
    elif osm_gdf.crs.to_epsg != epsg:
        osm_gdf.to_crs(epsg=epsg, inplace=True)
    if geo_gdf.crs is None:
        geo_gdf = geo_gdf.set_crs(epsg=epsg, allow_override=True)
    elif geo_gdf.crs.to_epsg != epsg:
        geo_gdf.to_crs(epsg=epsg, inplace=True)

    # finally spatial join and return result
    osm_gdf = gpd.sjoin(osm_gdf, geo_gdf[['coa_code', 'wkt_geom']], how='left', predicate='within')
    return osm_gdf

# DONE
def combine_all_census_data(dfs, join="inner", on='geography code'):
    """
    Given a list of dataframes of census data that all have `geography_code` features,
    join on this feature. Specify the join.

    Example usage:
    `df = combine_all_census_data([df1, df2, df3])`

    :param dfs: array of dataframes of census data
    :param join: specify the join type, `inner` or `outer`
    :param on: column name to join on:
    :returns: join of all dfs given as DataFrame
    """

    # if no dfs
    if len(dfs) == 0:
        print("no dfs in array")
        return None

    elif len(dfs) == 1:
        return dfs[0]

    # default join behaviour is inner
    if join != 'inner' and join != 'outer':
        join = 'inner'
        print(f"Invalid join type given, defaulting to {join}")

    # purge dfs that do not have geo code
    for df in dfs:
        if on not in df:
            dfs.remove(df)
            print(dfs.index(df) + " index df in submitted array purged, no geo code")

    # merge all dfs and return
    result = dfs[0].copy()
    result = result.reset_index().rename(columns={"index": "geography"})
    for df in dfs[1:]:
        result = result.merge(df, how=join, on=on)

    return result

# DONE
def upload_total_census_data_table_in_chunks(total_census_df, conn, chunk_size=5000, tablename='task1_census_information'):
    """
    Given a census dataframe corresponding to the inner schema, upload to cloud.
    Load it in by rows so you can do chunks at a time.
    
    :param total_census_df
    :param conn
    :param chunk_size
    :param tablename
    """
    cur = conn.cursor()
    l = len(total_census_df)

    print("Clearing table...")
    cur.execute(f"DROP TABLE IF EXISTS {tablename};")
    print("Cleared table.")

    print("Creating table with schema...")
    cur.execute(f"CREATE TABLE IF NOT EXISTS {tablename} (id INT AUTO_INCREMENT PRIMARY KEY NOT NULL, geography VARCHAR(255) NOT NULL, geography_code VARCHAR(255), l1_3 DOUBLE, l4_6 DOUBLE, l7 DOUBLE, l8_9 DOUBLE, l10_11 DOUBLE, l12 DOUBLE, l13 DOUBLE, l14 DOUBLE, l15 DOUBLE, active_exc_students DOUBLE, active_exc_students_in_employment DOUBLE, active_exc_students_unemployed DOUBLE, active_students DOUBLE, active_students_in_employment DOUBLE, active_students_unemployed DOUBLE, inactive DOUBLE, inactive_retired DOUBLE, inactive_student DOUBLE, inactive_looking_after_home_or_family DOUBLE, inactive_long_term_sick_or_disabled DOUBLE, inactive_other DOUBLE) DEFAULT CHARSET=utf8 COLLATE=utf8_bin AUTO_INCREMENT=1 ;")
    print("Created table.")

    print(f"Length of df: {l}")
    print(f"Number of chunks to upload: {math.ceil(l/chunk_size)}")

    print("Uploading to table in cloud...")
    # splitting data from gdf into rows so we can iterate over them with .executemany()
    data = [(
        r['geography'],
        r['geography code'],
        r['l1-3'],
        r['l4-6'],
        r['l7'],
        r['l8-9'],
        r['l10-11'],
        r['l12'],
        r['l13'],
        r['l14'],
        r['l15'],
        r['active (exc students)'],
        r['active (exc students): in employment'],
        r['active (exc students): unemployed'],
        r['active students'],
        r['active students: in employment'],
        r['active students: unemployed'],
        r['inactive'],
        r['inactive: retired'],
        r['inactive: student'],
        r['inactive: looking after home or family'],
        r['inactive: long-term sick or disabled'],
        r['inactive: other']
    ) for _, r in total_census_df.iterrows()]
    query_string = f"INSERT INTO {tablename} (geography, geography_code, l1_3, l4_6, l7, l8_9, l10_11, l12, l13, l14, l15, active_exc_students, active_exc_students_in_employment, active_exc_students_unemployed, active_students, active_students_in_employment, active_students_unemployed, inactive, inactive_retired, inactive_student, inactive_looking_after_home_or_family, inactive_long_term_sick_or_disabled, inactive_other) values (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s);"

    # process by chunks
    for chunk in range(math.ceil(l / chunk_size)):
        start = chunk * chunk_size
        end = min((chunk+1)*chunk_size, l)
        cur.executemany(query_string, data[start:end]) # this executes the same query string, and injects each row data into the string
        print(f"Uploaded chunk {chunk + 1}.")

        conn.commit()

# DONE
def upload_geo_coa_table_in_chunks(geo_coordinates_df, conn, chunk_size=5000, tablename='geo_codes_information'):
    """
    Upload geo code information from the dataframe in chunks of 5000 rows.
    :param geo_coordinates_df
    :param conn
    :param chunk_size: number of rows at a time
    :param tablename: table name in cloud
    """
    cur = conn.cursor()

    # work
    print("Clearing table...")
    cur.execute(f"DROP TABLE IF EXISTS {tablename};")
    print("Cleared table.")

    print("Creating table with schema...")
    cur.execute(f"CREATE TABLE IF NOT EXISTS {tablename} (id INT AUTO_INCREMENT PRIMARY KEY NOT NULL, coa_code VARCHAR(255) NOT NULL, super_coa_code VARCHAR(255), super_coa_name VARCHAR(255), super_coa_name_welsh VARCHAR(255), bng_east DOUBLE, bng_north DOUBLE, global_id VARCHAR(255), shape_length DOUBLE, shape_area DOUBLE, wkt_geom MULTIPOLYGON NOT NULL) DEFAULT CHARSET=utf8 COLLATE=utf8_bin AUTO_INCREMENT=1 ;")
    print("Created table.")

    # not required here
    # print("Saving dataframe to csv...")
    # geo_coordinates_df.to_csv(local_csv_path, header=False, index=False, quoting=2, na_rep='\\N')
    # print("Saved dataframe to csv.")

    l = len(geo_coordinates_df)

    print(f"Length of df: {l}")
    print(f"Number of chunks to upload: {math.ceil(l/chunk_size)}")

    print("Uploading to table in cloud...")
    data = [(
        r['coa_code'],
        r['super_coa_code'],
        r['super_coa_name'],
        r['super_coa_name_welsh'],
        r['bng_east'],
        r['bng_north'],
        r['global_id'],
        r['shape_length'],
        r['shape_area'],
        r['wkt_geom']
    ) for _, r in geo_coordinates_df.iterrows()]
    query_string = f"INSERT INTO {tablename} (coa_code, super_coa_code, super_coa_name, super_coa_name_welsh, bng_east, bng_north, global_id, shape_length, shape_area, wkt_geom) values (%s, %s, %s, %s, %s, %s, %s, %s, %s, ST_GeomFromText(%s, 27700))"

    # working and saving in chunks to cloud
    for chunk in range(math.ceil(l / chunk_size)):
        start = chunk * chunk_size
        end = min((chunk+1)*chunk_size, l)
        cur.executemany(query_string, data[start:end]) # this executes the same query string, and injects each row data into the string
        print(f"Uploaded chunk {chunk + 1}.")

        conn.commit()

# DONE - helper
def save_aug_osm_df_to_csv(aug_osm_df, filename):
    """
    Helper function.
    Save augmented OSM data to csv.
    :param aug_osm_df
    :param filename
    """
    aug_osm_df.to_csv(filename, index=False)

# DONE
def upload_table_to_cloud_from_csv(csv_file_path, conn, tablename = 'osm_national_compressed_table'):
    """
    Dynamically create table with headers from file, and upload with LOCAL DATA LOAD INFILE.
    
    Example usage:
    `upload_table_to_cloud_from_csv("path/to/csv.csv", conn)`
    
    :param csv_file_path
    :param conn
    :param tablename
    """
    cur = conn.cursor()
    
    print("Dropping table if exists...")
    cur.execute("DROP TABLE IF EXISTS " + tablename + ";")
    print("Done.")

    print("Creating schema for table...")
    # we dynamically get the schema string through the following steps:
    # 1) we can read in only the first line of the csv file to get the headers
    with open(csv_file_path, newline='') as file:
        r = csv.reader(file)
        headers = next(r)
        #mandatory_headers = headers[:3]
        feature_headers = headers[3:]

    # 2) add the id, latitude, longitude columns to the inner string for the schema, and then all headers
    # we add a geometry column for spatial search, and make it autogenerated from longitude and latitude
    # NB: the reason we do not use the id as a unique key is that it is not unique!
    schema_string = f"""
    CREATE TABLE IF NOT EXISTS {tablename} (
        id INT NOT NULL,
        latitude DOUBLE NOT NULL,
        longitude DOUBLE NOT NULL,
        {', '.join([f"`{feature}` TEXT" for feature in feature_headers])}
    ) DEFAULT CHARSET=utf8 COLLATE=utf8_bin;
    """

    # 4) now use the schema string to create the schema
    cur.execute(schema_string)
    print("Done.")

    # note: same command as practical 1, but ignore the first row of headers.
    print("Uploading the table from csv...")
    load_string = f"""
    LOAD DATA LOCAL INFILE '{csv_file_path}'
    INTO TABLE {tablename}
    FIELDS TERMINATED BY ','
    OPTIONALLY ENCLOSED BY '"'
    LINES TERMINATED BY '\\n'
    IGNORE 1 ROWS;
    """
    cur.execute(load_string)
    print("Done.")

    conn.commit()

    # verify the number of rows in the cloud db table is what you expected
    print("Verifying cloud db table...")
    cur.execute(f"select count(*) from {tablename}")
    print(f"Number of rows in cloud table {tablename}: {cur.fetchone()[0]}")

# DONE
def add_generated_column_to_table_in_cloud(conn, tablename, new_column, data_type, expression, stored=True):
    """
    Given a table, add a new generated column that is calculated with a given expression.
    
    Example usage:
    `add_generated_column_to_table_in_cloud(conn, "table", "newcol", "INT", "expression", stored=False)`
    
    :param conn
    :param tablename
    :param new_column: name
    :param data_type: string
    ;param expression: string to calculate column vals
    :param stored: whether the values are runtime calculated (false) or stored (true)
    """
    cur = conn.cursor()

    query=f"ALTER TABLE {tablename} ADD COLUMN {new_column} {data_type} AS {expression} {'STORED' if stored else 'VIRTUAL'};"
    cur.execute(query)

    conn.commit()

# DONE
def create_index_on_table_in_cloud(conn, table_name, features, index_name=None, spatial=False):
    """
    Create a BTREE index on a table in the cloud given a connection and the features.
    :param conn
    :param table_name
    :param features: list of string features
    :param index_name: name of index (auto-generated if not submitted)
    :param spatial: boolean whether feature is spatial or not
    """

    if index_name is None or index_name == "":
        index_name = "idx_" + "_".join(features)
        print(f"Index name not given, selecting {index_name} as index name (default)")
    
    cur = conn.cursor()
    if spatial:
        query = f"CREATE SPATIAL INDEX {index_name} ON {table_name}({', '.join(features)})"
    else:
        query = f"CREATE INDEX {index_name} ON {table_name}({', '.join(features)})"
    cur.execute(query)
    conn.commit()
    print(f"{'Spatial i' if spatial else 'I'}ndex {index_name} created on {table_name} in cloud successfully.")

# ------------------------------------------
# TASK 1
# 1.EXT) cloud function wrappers for future tasks
# ------------------------------------------

# DONE - helper
def send_query_to_cloud(query, conn):
    """
    Helper function.
    Wrapper function to send a query to the cloud without returning data, and commit result.
    :param query: string query
    :param conn
    """
    cur = conn.cursor()
    cur.execute(query)
    conn.commit()

# DONE - helper
def alter_table_in_cloud(conn, tablename, action):
    """
    Helper function.
    Given an action, alter the table in the cloud, and commit result.
    :param conn
    :param tablename
    :param action: the remaining portion of the alter table query.
    """
    query = f"ALTER TABLE {tablename} {action};"
    send_query_to_cloud(query, conn)

# DONE - helper
def fetch_query_from_cloud_as_df(query, conn):
    """
    Helper function.
    Wrapper function to fetch and return data from the cloud as a DataFrame.
    :param query: string query
    :param conn
    :returns: DataFrame of result from query
    """
    cur = conn.cursor()
    cur.execute(query)
    rows = cur.fetchall()

    return pd.DataFrame(rows, columns=[c[0] for c in cur.description])

# DONE
def get_entire_table(conn, tablename, headers=None):
    """
    Helper function.
    Get an entire table from the cloud.
    :param conn
    :param tablename
    :param headers: optional list of headers in table
    :returns: DataFrame of result to query
    """
    query = f"SELECT {'*' if headers is None else ', '.join(headers)} FROM {tablename};"
    return fetch_query_from_cloud_as_df(query, conn)

# DONE
def get_osm_features_frequency_by_code(conn, headers, tablename='osm_with_coa_national_compressed_table', group_by='coa_code'):
    """
    Get frequency of features from the augmented osm table grouped by coa_code (by default and recommended).
    :param conn
    :param tablename
    :param group_by: feature name
    :param headers: mandatory list of headers in table to count
    :returns: DataFrame of result table
    """
    if headers is None:
        print("error")
        return None
    
    counts = ', '.join([f'COUNT(CASE WHEN `{header}` != "" THEN 1 END) as `{header}_freq`' for header in headers])

    query = f"""
    SELECT
        {group_by}, {counts}
    FROM {tablename}
    GROUP BY {group_by};
    """
    return fetch_query_from_cloud_as_df(query, conn)

# DONE
def get_response_variable(conn, tablename='total_census_table',response='l15', geo_code_feature='geography_code'):
    """
    Get the response variable from the given table.
    :param conn
    :param tablename
    :param response
    :param geo_code_feature: name of the geocode to map to response var
    :returns: DataFrame of query result
    """
    query = f"SELECT {geo_code_feature}, {response} FROM {tablename}"
    return fetch_query_from_cloud_as_df(query, conn)

# DONE
def get_features_and_response_task1(conn, feature_headers, feature_table='osm_by_features', response_table='task1_census_information', response='l15', feature_geo_code='coa_code', response_goa_code='geography'):
    """
    Join the features and response and return as an entire table for task 1.
    :param conn
    :param feature_headers: the features you want in the table
    :param feature_table
    :param response: name of var
    :param feature_geo_code: the geo code to align for feature table
    :param response_geo_code: the geo code to align for response table
    :returns: features and response combined DataFrame
    """
    if feature_headers is None:
        print("error")
        return None
    
    counts = ', '.join([f'COUNT(CASE WHEN `{header}` != "" THEN 1 END) as `{header}_freq`' for header in feature_headers])

    # use an inner query to inject into the larger query so you can join them together
    inner_query = f"""
    SELECT
        {feature_geo_code}, {counts}
    FROM {feature_table}
    GROUP BY {feature_geo_code}
    """

    # now join the inner query with the osm data and fetch as a dataframe
    query = f"""
    SELECT osm.{feature_geo_code}, {', '.join(["osm.`"+header+"_freq`" for header in feature_headers])}, ce.{response}
    FROM ({inner_query}) as osm
    LEFT JOIN {response_table} as ce
    ON osm.{feature_geo_code}=ce.{response_goa_code};
    """
    return fetch_query_from_cloud_as_df(query, conn)

# DONE
def fetch_query_from_cloud_as_gdf(query, conn, geometry):
    """
    Given a suitable query, return as a geodataframe, with the specified geometry column.
    NB: make sure to fetch geometry as text! e.g. \"select *, ST_AsText(wkt_geom) as wkt_geom_text from geo_coa_table limit 5;\"
    This works for tables where the geometry is stored with MariaDB geoemtry types.

    :param query
    :param conn
    :param geometry: name of geom col in fetched SQL query.
    :returns: GeoDataFrame of result, with submitted geometry
    """
    df = fetch_query_from_cloud_as_df(query, conn)
    df[geometry] = df[geometry].apply(wkt.loads)
    return gpd.GeoDataFrame(df, geometry=geometry)


# -------------
# TASK 2 ACCESS
# -------------

# DONE
def preprocess_ts019_census_dataframe(ts_019_df, normalize=True):
    """
    Preprocess raw census data in dataframe, and return cleaned dataframe with processed columns.
    :param ts_019_df
    :param normalize: boolean to whether or not normalize row counts to make them proportion
    :returns: processed DataFrame
    """
    ts_019_df = ts_019_df.drop(ts_019_df.columns[[0]], axis=1)
    columns = ['geography','geography code','total','not_migrant','student_migrant','uk_migrant','international_migrant']
    ts_019_df.columns = columns
    ts_019_df = ts_019_df.set_index('geography')

    if normalize:
        columns_to_normalize = columns.copy()
        columns_to_normalize.remove("geography")
        columns_to_normalize.remove("geography code")
        ts_019_df[columns_to_normalize] = ts_019_df[columns_to_normalize].div(ts_019_df['total'], axis=0)

        ts_019_df = ts_019_df.drop(columns=['total']) # as the total becomes redundant when we normalize within rows

    return ts_019_df

# DONE
def preprocess_and_join_all_1981_census_dfs(age_census_1981_df, migrant_census_1981_df, employment_census_1981_df, on='mnemonic'):
    """
    Given all census dfs, preprocess and combine accordingly:
    1) rename any columns to make features distinct
    2) normalize values appropriately
    3) join on mnemonic

    :param age_census_1981_df
    :param migrant_census_1981_df
    :param employment_census_1981_df
    :param on: column to join on
    :returns: result DataFrame of combination of census data
    """
    # step 1: rename migrant census columns
    migrant_census_columns = [
        "parliamentary constituency 1983 revision",
        "mnemonic"
    ]
    migrant_census_columns = migrant_census_columns + ["migrant: "+ name for name in migrant_census_1981_df.columns[2:]]

    # step2a: normalize age, migrant and employment data together
    age_census_1981_df['total'] = age_census_1981_df.drop(columns=['parliamentary constituency 1983 revision', 'mnemonic']).sum(axis=1)

    age_census_cols_to_normalize = list(age_census_1981_df.columns)
    age_census_cols_to_normalize.remove("parliamentary constituency 1983 revision")
    age_census_cols_to_normalize.remove("mnemonic")
    age_census_cols_to_normalize.remove("total")
    age_census_1981_df[age_census_cols_to_normalize] = age_census_1981_df[age_census_cols_to_normalize].div(age_census_1981_df['total'], axis=0)

    migrant_census_cols_to_normalize = list(migrant_census_1981_df.columns)
    migrant_census_cols_to_normalize.remove("parliamentary constituency 1983 revision")
    migrant_census_cols_to_normalize.remove("mnemonic")
    migrant_census_1981_df[migrant_census_cols_to_normalize] = migrant_census_1981_df[migrant_census_cols_to_normalize].div(age_census_1981_df['total'], axis=0)
    
    employment_census_cols_to_normalize = list(employment_census_1981_df.columns)
    employment_census_cols_to_normalize.remove("parliamentary constituency 1983 revision")
    employment_census_cols_to_normalize.remove("mnemonic")
    employment_census_1981_df[employment_census_cols_to_normalize] = employment_census_1981_df[employment_census_cols_to_normalize].div(age_census_1981_df['total'], axis=0)

    # step 3: join all on mnemonic column
    temp_df = pd.merge(age_census_1981_df, migrant_census_1981_df.drop(columns=['parliamentary constituency 1983 revision']), how='inner', on='mnemonic')
    result_df = pd.merge(temp_df, employment_census_1981_df.drop(columns=['parliamentary constituency 1983 revision']), how='inner', on='mnemonic')

    return result_df

# DONE
def preprocess_2024_election_df(df):
   """
   Preprocess the 2024 election dataframe for visualization purposes.
   :param df
   :returns: DataFrame of preprocessed election data for 2024.
   """
   # remove non england constituencies, got these from printing unique col vals
   england_constituencies = set(['South East', 'West Midlands', 'North West',
      'East Midlands', 'London', 'Yorkshire and The Humber',
      'East of England', 'South West', 'North East'])
   df = df[df['Region name'].isin(england_constituencies)]

   # remove columns that are irrelevant
   df = df.drop(df.columns[[3,4,5,6,7,8,9,10,14,15,16,18,19,20,21,22,23,24,25,26,27,28,29,30,31]], axis=1)
    
   # preprocess "Result" to categorical 'hold' or 'gained'
   df['Result'] = df['Result'].apply(lambda x: 'Hold' if 'Hold' in x else 'Gained')

   return df

# DONE - helper
def csv_url_to_dataframe_with_encoding(url, encoding='ISO-8859-1'):
    """
    Helper function.
    Given a url to a csv file, download it (if not local) and return it as a pandas dataframe,
    WITH a specific encoding that is not default in the pandas read_csv function.
    :param url: string url
    :param encoding
    :returns: loaded DataFrame
    """
    df = pd.read_csv(url, encoding=encoding)
    return df

# DONE
def preprocess_historical_election_df(df):
    """
    Given a dataframe of historical elections, preprocess the dataframe and return the result.
    :param df
    :returns: preprocessed election DataFrame
    """
    # NB: we could filter out to inly have the elections that we are interested in, but this is
    # against good data science practice of having data!

    # filter for england and wales
    df = df[
        (df['country/region'] != "Ireland") &
        (df['country/region'] != "Scotland") &
        (df['country/region'] != "Northern Ireland")
    ]

    # rename different occurences of Yorkshire and the Humber
    df['country/region'] = df['country/region'].apply(lambda x: "Yorkshire and the Humber" if x in set(['Yorkshire & The Humber', 'Yorkshire and The Humber']) else x)

    # remove empty trailing column
    df = df.drop(columns=['Unnamed: 19'], axis=1)

    # remove trailing spaces in column names (lib_votes had invalid name before this)
    df.columns = df.columns.str.strip()

    return df

# DONE
def load_in_constituency_geometries(year='1983', country='england', area='midlands'):
    """
    Load in geometries for a constituency using the geojson module to process geometries.
    :param year
    :param country
    :param area
    :returns: GeoDataFrame of geometries from file
    """
    path = f"elections/constituency_geometries/{year}_constituencies___{country}__{area}_.geojson"
    with open(path) as file:
        data = geojson.load(file)
    return data

# DONE
def return_combined_area_gdf_constituencies(geometries_dict, ids_df, epsg=4326, on="id"):
    """
    Given loaded geometries and IDs for a place, combine and return a result that can be used as a geo search table.
    :param geometries_dict: 
    :param ids_df
    :param epsg: coordinate ref system, default lat long (4326)
    :param on: the col to join on
    :returns: GeoDataFrame of constituencies and data
    """
    # step 1: convert this loaded dict to a gdf
    columns = [on, 'geom_name', 'geometry']
    geometries_gdf = gpd.GeoDataFrame(columns=columns)

    for feature in geometries_dict["features"]:
        row = {c: [] for c in columns}

        row[on].append(feature["properties"][on])
        row["geom_name"].append(feature["properties"]["Name"])
        
        if on == "id":
            points = [Point(long, lat) for long, lat in feature["geometry"]["coordinates"][0]]
        else:
            points = [Point(long, lat) for long, lat in feature["geometry"]["coordinates"][0][0]]

        row["geometry"].append(Polygon(points))
        geometries_gdf = pd.concat([geometries_gdf, gpd.GeoDataFrame(row)], ignore_index=True)

    geometries_gdf.set_geometry('geometry', inplace=True)
    geometries_gdf.set_crs(epsg=epsg, allow_override=True, inplace=True)

    # step 2: convert the ids to a gdf (and rename col if necessary)
    ids_gdf = gpd.GeoDataFrame(ids_df)

    # step 3: join them on id and return result
    result = geometries_gdf.merge(ids_gdf, on=on, how="right")
    if on != "id":
        result = result.rename(columns={on:"id"})
    return result

# DONE - helper function
def concat_constituency_tables_for_year(gdfs):
    """
    Helper function.
    Return a concatenated geodataframe from a list of geodataframes that are similarly structured.
    :param gdfs
    :returns: single GeoDataFrame of concatenated gdfs given
    """
    if len(gdfs) < 2:
        return gdfs[0] if len(gdfs) == 1 else None
    return pd.concat(gdfs, ignore_index=True)

# DONE
def add_lookup_for_census_into_constituencies_table(constituencies_df, census_df, lookup_colname="lookup_census"):
    """
    Given a table for constituencies, and a table for census data, add to the constituencies table a 
    lookup into the census mnemonic.
    :param constituencies_df
    :param census_df
    :param lookup_colname
    :returns: DataFrame of constituencies
    """
    # step 0: ensure crs
    constituencies_df = constituencies_df.to_crs(epsg=4326)
    
    # step 1: add the lookup col to the gdf
    constituencies_df['fuzzcol'] = constituencies_df['geom_name'].apply(
        lambda x: extract(
            query=x,
            choices=census_df['parliamentary constituency 1983 revision'],
            score_cutoff=80
        )
    )
    constituencies_df[lookup_colname] = constituencies_df['fuzzcol'].apply(lambda vals: max(vals, key=lambda x: x[1])[0] if vals != [] else np.nan)
    constituencies_df.drop(columns=['fuzzcol'], inplace=True)
    # constituencies_df[lookup_colname].fillna(method='ffill', inplace=True)

    # step 2: use spatial matching rather than forward fill
    if constituencies_df[lookup_colname].isna().any():
        df_nans = constituencies_df[constituencies_df[lookup_colname].isna()]
        df_vals = constituencies_df[constituencies_df[lookup_colname].notna()]

        # join with itself to find nearest mathces
        checks = df_nans.sjoin_nearest(
            df_vals[['geometry', lookup_colname]],
            how='left'
        )
        checks = checks.groupby(checks.index).first()

        constituencies_df.loc[
            constituencies_df[lookup_colname].isna(), lookup_colname
        ] = checks[lookup_colname+'_right']

    # step 3: use mnemonic instead of geomnames
    constituencies_df = pd.merge(
        constituencies_df,
        census_df[['parliamentary constituency 1983 revision', 'mnemonic']],
        left_on='lookup_census',
        right_on='parliamentary constituency 1983 revision',
        how='left'
    ).drop(
        columns=['parliamentary constituency 1983 revision', 'lookup_census']
    ).rename(
        columns={'mnemonic': 'lookup_census'}
    )

    # step 4: return
    return constituencies_df

# DONE
def get_response_variable_for_boundary_set(election_historical_df, boundary_set='1974-1979', majority_col=False, party_col=False):
    """
    Assuming that the set of constituency id's and mappings to constituencies are the same within a boundary set,
    return a df of constituency id against response variable i.e. 0 for stronghold, 1 for marginal.
    :param election_historical_df
    :param boundary_set: column in election_historical_df
    :param majority_col: whether or not to include a column for average historical majority
    :param party_col: whether or not to include a column for last party that won
    :returns: response var and other vars as DataFrame
    """
   # step 0: filter for only the boundary set
    df = election_historical_df[election_historical_df["boundary_set"] == boundary_set]

    # step 1: replace all invalid vals with 0 (dataset has empty and-1 vals)
    df = df.fillna(0.0)
    df = df.replace(to_replace=-1.0, value=0.0)

    # step 2: get dict that corresponds to which party won max VOTES in each election
    # we choose votes for better specificity than share rounded to 3dp. in the dataset.
    cols_to_check = ["con_votes", "lib_votes", "lab_votes", "natSW_votes", "oth_votes"]
    constituency_results = {}

    for _, row in df.iterrows():
        votes = {
            c: row[c] for c in cols_to_check
        }
        constituency_id = row["constituency_id"]
        max_votes_party = max(votes, key=lambda x: float(votes[x]))

        if constituency_id not in constituency_results:
            constituency_results[constituency_id] = []
        constituency_results[constituency_id].append(max_votes_party)

    # step 3: process this dict to get response var table
    data = {"constituency_id": [], "response_variable": []}
    for k, vs in constituency_results.items():
        data["constituency_id"].append(k)
        if len(set(vs)) == 1:
            data["response_variable"].append(0) # stronghold
        else:
            data["response_variable"].append(1) # marginal

    result_df = pd.DataFrame(data)

    # optional step 4: get majority column i.e. max minus second max col out of _votes suffix columns
    if majority_col:
        set_cols_correct_suffix = set([i for i in list(df.columns) if i[-6:] == "_votes" and i[:-6] != "total"])
        votes_df = df.copy()[list(set_cols_correct_suffix) + ['constituency_id']]
        votes_values_cols = [c for c in votes_df if c != "constituency_id" and c in set_cols_correct_suffix]
        votes_df[votes_values_cols] = votes_df[votes_values_cols].astype(float)
        
        votes_df['max'] = votes_df[votes_values_cols].max(axis=1)
        votes_df['secondmax'] = votes_df[votes_values_cols].apply(lambda x: x.nlargest(2).values[-1], axis=1)
        votes_df['Majority'] = votes_df['max'] - votes_df['secondmax']
        votes_df = votes_df[['constituency_id', 'Majority']].groupby(['constituency_id']).mean()

        result_df = pd.merge(
            result_df,
            votes_df,
            on='constituency_id',
            how='left'
        )

    # optional step 5: get party column
    if party_col:
        set_cols_correct_suffix = set([i for i in list(df.columns) if i[-6:] == "_votes" and i[:-6] != "total"])
        votes_df = df.copy()[list(set_cols_correct_suffix) + ['constituency_id']]
        votes_values_cols = [c for c in votes_df if c != "constituency_id" and c in set_cols_correct_suffix]
        votes_df[votes_values_cols] = votes_df[votes_values_cols].astype(float)
        parties = {c:c[:-6] for c in set_cols_correct_suffix}

        votes_df['Party'] = votes_df[votes_values_cols].idxmax(axis=1).map(parties)
        votes_df = votes_df[['constituency_id', 'Party']].groupby(['constituency_id']).first()

        result_df = pd.merge(
            result_df,
            votes_df,
            on='constituency_id',
            how='left'
        )

    # step 6: join back the names of constituencies and return, we do the groupby stuff to avoid dupe rows
    returning_df = pd.merge(result_df, df[["constituency_id", "constituency_name"]], how="left", on="constituency_id").groupby(["constituency_id"])
    if not majority_col and not party_col:
        returning_df = returning_df[["constituency_name", "response_variable"]].first().reset_index()
    elif majority_col and not party_col:
        returning_df = returning_df[["constituency_name", "response_variable", "Majority"]].first().reset_index()
    elif not majority_col and party_col:
        returning_df = returning_df[["constituency_name", "response_variable", "Party"]].first().reset_index()
    elif majority_col and party_col:
        returning_df = returning_df[["constituency_name", "response_variable", "Majority", "Party"]].first().reset_index()
    return returning_df

# DONE
def rename_cols_to_fit_mariadb_limit(df):
    """
    Rename columns to fit the 64 char limit of MariaDB.
    :param df
    :returns: DataFrame wth renamed columns
    """
    columns = list(df.columns)
    new_columns = []
    for col in columns:
        new_columns.append("".join(char for char in col if char.isalpha() or char.isdigit()))
    df.columns = [i[:61] for i in new_columns]
    return df

# DONE
def upload_table_to_cloud_from_csv_task2(csv_file_path, conn, tablename = 'constituencies_1974_lookup'):
    """
    Dynamically create table with headers from file, and upload with LOCAL DATA LOAD INFILE.
    Rewritten from task 1 to not include id, lat and long as separate in schema.
    Ignore all comments, this is literally a clone of the other function!
    :param csv_file_path
    :param conn
    :param tablename
    """
    cur = conn.cursor()
    
    print("Dropping table if exists...")
    cur.execute("DROP TABLE IF EXISTS " + tablename + ";")
    print("Done.")

    print("Creating schema for table...")
    # we dynamically get the schema string through the following steps:
    # 1) we can read in only the first line of the csv file to get the headers
    with open(csv_file_path, newline='') as file:
        r = csv.reader(file)
        headers = next(r)
        feature_headers = headers

    # 2) add the id, latitude, longitude columns to the inner string for the schema, and then all headers
    # we add a geometry column for spatial search, and make it autogenerated from longitude and latitude
    schema_string = f"""
    CREATE TABLE IF NOT EXISTS {tablename} (
        {', '.join([f"`{feature}` TEXT" for feature in feature_headers])}
    ) DEFAULT CHARSET=utf8 COLLATE=utf8_bin;
    """

    # 4) now use the schema string to create the schema
    cur.execute(schema_string)
    print("Done.")

    # note: same command as practical 1, but ignore the first row of headers.
    print("Uploading the table from csv...")
    load_string = f"""
    LOAD DATA LOCAL INFILE '{csv_file_path}'
    INTO TABLE {tablename}
    FIELDS TERMINATED BY ','
    OPTIONALLY ENCLOSED BY '"'
    LINES TERMINATED BY '\\n'
    IGNORE 1 ROWS;
    """
    cur.execute(load_string)
    print("Done.")

    conn.commit()

    # verify the number of rows in the cloud db table is what you expected
    print("Verifying cloud db table...")
    cur.execute(f"select count(*) from {tablename}")
    print(f"Number of rows in cloud table {tablename}: {cur.fetchone()[0]}")


# DONE
def latest_data_grouped_on_constituencies(total_census_df, latest_lsoa_constit_lookups):
    """
    Given census data and lookups, convert LSOA's to wards, and average and return the result.
    :param total_census_df
    :param latest_lsoa_constit_lookups
    :returns: merged DataFrame i.e. LSOA -> ward
    """
    return pd.merge(
        total_census_df, 
        latest_lsoa_constit_lookups[['LSOA21CD', 'PCON25CD', 'PCON25NM']], 
        left_on='geography code',
        right_on='LSOA21CD',
        how='left'
    ).drop(
        columns=['geography', 'geography code', 'LSOA21CD']
    ).rename(
        columns={'PCON25CD': 'geography code', 'PCON25NM': 'geography'}
    ).groupby(['geography', 'geography code']).sum().reset_index()

# DONE
def latest_census_data_with_response_var(total_census_df, election_2024_df):
    """
    Given census and election dataframes for 2024, combine to return census data augmented with the response variable.
    :param total_census_df
    :param election_2024_df
    :returns: DataFrame of census with response and other new vars
    """
    df = pd.merge(
        total_census_df,
        election_2024_df[['ONS ID', 'First party', 'Result', 'Majority']],
        left_on='geography code',
        right_on='ONS ID',
        how='left'
    ).drop(
        columns=['ONS ID']
    )
    df['Result'] = df['Result'].apply(
        lambda x: 0 if x == "Hold" else 1
    )
    return df.rename(columns={'Result': 'responsevariable'})

# DONE
def merge_response_to_constituencies_gdf(
    response_vars,
    constituencies_gdf,
    other_cols_to_pass = [('Majority', float), ('Party', str)]
):
    """
    Given a dataframe of response variable (and posibly more variables) calculated from historical
    election data, and a geodataframe of constituencies, attach the variables to the constituencies
    by joining with string matching, and return the result.
    :param response_vars
    :param constituencies_gdf
    :param other_cols_to_pass
    :returns: constituencies GeoDataFrame with response (and other) variable(s)
    """
    result_gdf = constituencies_gdf.copy()

    # step 1: get fuzzy matches
    result_gdf['fuzzcol'] = result_gdf['geom_name'].apply(
        lambda x: extract(
            query=x, 
            choices=response_vars['constituency_name'], 
            score_cutoff=50
        )
    )

    # step 2: get the highest likelihood string match between names
    result_gdf['constit_name_response_vars'] = result_gdf['fuzzcol'].apply(
        lambda fuzzvals: max(fuzzvals, key=lambda x: x[1])[0]
    )

    # step 3: use generated columns to get response variable (and others)
    response_col = []
    other_cols = {
        c[0]: [] for c in other_cols_to_pass
    }
    for _, row in result_gdf.iterrows():
        response_value = response_vars[
            response_vars['constituency_name']==row['constit_name_response_vars']
        ].iloc[0]['response_variable']
        response_col.append(response_value)

        for colname, coltype in other_cols_to_pass:
            matches = response_vars[
                response_vars['constituency_name']==row['constit_name_response_vars']
            ][colname]
            if coltype == float:
                val = matches.mean()
            elif coltype == str:
                # get the highest frequency, and random if tie (i.e. mode)
                maxvals = matches.mode()
                if len(maxvals) > 1:
                    val = random.choice(maxvals)
                else:
                    val = maxvals[0]
            other_cols[colname].append(val)

    result_gdf['response_variable'] = response_col
    for col, vals in other_cols.items():
        result_gdf[col] = vals

    # step 4: clean up unnecessary columns
    result_gdf = result_gdf.drop(
        columns=['constit_name_response_vars', 'fuzzcol']
    )

    # step 4: return result
    return result_gdf
    
# DONE
def pass_response_across_gdfs(
    gdf1,
    gdf2,
    other_cols_to_pass=[('Majority', float), ('Party', str)]
):
    """
    Given two geodataframes of constituencies where the first has the response variable
    (and others if necessary), transfer these variables from the first to the second with
    spatial logic, and return the second geodataframe.
    :param gdf1
    :param gdf2
    :param other_cols_to_pass
    :returns: gdf2 with cols passed from gdf1
    """
    # step 0: get right geom
    gdf1_copy = gdf1.copy().drop(columns=['geometry'])
    gdf1_copy = gpd.GeoDataFrame(
        gdf1_copy, 
        geometry=gpd.points_from_xy(gdf1_copy['Longitude'], gdf1_copy['Latitude'])
    )
    gdf1_copy.set_crs(epsg=4326, inplace=True)
    gdf2.to_crs(epsg=4326, inplace=True)

    # step 1: find nearest constituency vals and add to gdf2
    result_gdf = gpd.sjoin_nearest(
        gdf2,
        gdf1_copy[['geometry', 'response_variable'] + [i[0] for i in other_cols_to_pass]],
        how='left'
    )

    # step 2: drop unnecessary columns 
    result_gdf.drop(columns=['index_right'], inplace=True)

    # step 3: group by place, and reformat vals accordingly
    cols_to_group_by = ['id', 'geom_name', 'geometry', 'Name', 'Latitude', 'Longitude']
    custom_agg = {}
    for name, ctype in [('response_variable', None)] + other_cols_to_pass:
        if ctype is None:
            # if there is a tie for the response variable, choose swing (1)
            custom_agg[name] = lambda x: 1 if len(x.mode()) > 1 else x.mode().iloc[0]
        elif ctype is float:
            custom_agg[name] = 'mean'
        elif ctype is str:
            # similar logic to None, but random choice
            custom_agg[name] = lambda x: x.mode().iloc[0] if len(x.mode()) == 1 else random.choice(x.mode())
    result_gdf = result_gdf.groupby(cols_to_group_by).agg(custom_agg).reset_index()

    # finally return
    return result_gdf

# DONE
def augment_census_with_response_from_gdf(
    census_df,
    gdf,
    cols_to_pass=[('response_variable', None), ('Majority', float), ('Party', str)]
):
    """
    Given census dataframe and gdf with response and other variables, pass these over and return census.
    :param census_df
    :param gdf
    :param cols_to_pass
    :returns: census_df with cols passed to it
    """
    # step 0.1: filter out rows that had no data (i saw this on inspection of data)
    census_df = census_df[census_df['mnemonic'] != 'Column Total']
    census_df = census_df[census_df['mnemonic'].astype(int)<525]

    # step 0.2: make sure it is a gdf
    if type(gdf) != gpd.geodataframe.GeoDataFrame:
        gdf = gpd.GeoDataFrame(gdf, geometry='geometry', crs='EPSG:4326')

    # step 1: use string matching to find matches of names for each row in
    # the census dataset, and interpolate the ones that have no proper matches
    census_df['fuzzcol'] = census_df['parliamentary constituency 1983 revision'].apply(
        lambda x: extract(
            query=x,
            choices=gdf['geom_name'],
            score_cutoff=80
        )
    )
    census_df['geom_name'] = census_df['fuzzcol'].apply(lambda vals: max(vals, key=lambda x: x[1])[0] if vals != [] else np.nan)
    census_df.drop(columns=['fuzzcol'], inplace=True)
    census_df['geom_name'].fillna(method='ffill', inplace=True)
    
    # step 2: for each row, transfer over the response variable to the census_df
    # (and any other variables)
    census_df = pd.merge(
        census_df,
        gdf[['geom_name'] + [c[0] for c in cols_to_pass]],
        on='geom_name',
        how='left'
    ).drop(columns=['geom_name'])

    # step 3: return the result census df
    return census_df

# DONE
def reduce_geometry_resolutions(gdf, epsilon=0.007):
    """
    Helper function.
    Use topojson to reduce resolution of geometries whilst keeping boundaries and topology.
    Uses an approach similar to the following:
    https://gis.stackexchange.com/questions/325766/geopandas-simplify-results-in-gaps-between-polygons
    :param gdf
    :param epsilon: smoothening ratio, recommended 0.007 for UK constituencies
    :returns: gdf with simplified geometry
    """
    return tp.Topology(
        gdf, prequantize=False
    ).toposimplify(epsilon=epsilon).to_gdf()


# DONE
def fetch_assess_data_historical_and_latest(conn):
    """
    Fetch from the cloud census data for both historical and latest tables:
    :param conn
    :returns: both historical and latest data as DataFrames
    """
    # fetching historical data with features and response
    query = "select * from historical_census_constituencies"
    historical_census_df = fetch_query_from_cloud_as_df(query, conn)

    # fetching 2024 data with features and response
    query = "select * from 2024_census_constituencies"
    latest_census_df = fetch_query_from_cloud_as_df(query, conn).rename(columns={'First party':'Party'})

    return historical_census_df, latest_census_df

# DONE
def fetch_constituency_geometries_1983(conn):
    """
    Fetch from the cloud constituency geometries from 1983, for plotting, joined with their features and response.
    :param conn
    :returns: DataFrame of constituencies with geometry, features and response.
    """
    # fetching constituencies joined to their values
    query = "select * from constituencies_1983_with_lookup as co left join historical_census_constituencies as hi on co.lookup_census=hi.mnemonic;"
    constituencies_1983 = fetch_query_from_cloud_as_df(query, conn)
    constituencies_1983['geometry'] = gpd.GeoSeries.from_wkt(constituencies_1983['geometry'])
    constituencies_1983 = gpd.GeoDataFrame(constituencies_1983, geometry='geometry', crs='EPSG:4326')
    constituencies_1983 = constituencies_1983.drop(
        columns=['id', 'geom_name', 'Name', 'lookup_census']
    )

    return constituencies_1983




