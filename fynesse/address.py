from . import access
from sklearn.metrics import r2_score
import random
from sklearn.model_selection import train_test_split, KFold
from sklearn.linear_model import LinearRegression

# DONE
def spatially_cluster_geocode_groupings(coa_code_groupings_indv, n=10):
    """
    Spatially cluster geocode into groups on account of the fact that they are
    ordered by code already in the data.
    :param coa_code_groupings_indv
    :param n: nuber of codes in a grouping
    """
    to_add = []
    to_add_count = 0
    result = []
    for coa_code in coa_code_groupings_indv:
        coa_code = coa_code[0]
        to_add.append(coa_code)
        to_add_count += 1

        if to_add_count == n:
            result.append(to_add)
            to_add = []
            to_add_count = 0
    
    return result

# NB: here I decide to package the model into a class to make it more transparent
#     and encourage good programming principles.
# DONE
class Task1Model:
    """
    Class to enclose the model for Task 1 of the assessment, which will be reusable for
    any decided response variable from the census data in the cloud.

    The model is set as a dictionary of coa clusters corresponding to a machine learning prediction model.
    """

    def __init__(self, conn, osm_features, census_response_variable, coa_code_groupings, model=LinearRegression):
        """
        Set all of the class variables required for the model to operate.
        
        :param conn
        :param osm_features: features of the data to predict with
        :param census_response_variable: name of the response variable to predict on
        :param coa_code_groupings: list of lists corresponding to clusters of coa codes
        :param model: the model to initialize for prediction (allows flexibility in submission to class)
        """
        self.model = {}

        i = 0
        for cluster in coa_code_groupings:
            self.model[i] = {"coa_codes":set(cluster), "model":model()}
            i+=1

        self.response = census_response_variable
        self.features = osm_features
        self.conn = conn

        print("Initialized model.")
        print(type(self.features[0]))

    def initializeModelAndFetchData(self, conn=None, prefetchedData=None):
        """
        Fetch the data for training, testing etc.
        :param conn: pass a new connection if you want to.
        :param prefetchedData: pass in the data if you've fetched it already
        """
        if conn is not None:
            self.conn = conn

        print("Fetching data for model...")

        if prefetchedData is None:
            self.data_df = access.get_features_and_response_task1(
                self.conn,
                self.features,
                response = self.response
            ).dropna(subset=['coa_code'])
        else:
            self.data_df = prefetchedData.dropna(subset=['coa_code'])

        print("Fetched data for model.")
    
    def trainModel(self, model=None):
        """
        Train the model on prefetched data (must run self.initializeModelAndFetchData first).
        
        :param model: submit a saved model if you've already trained it
        """
        if model is None:
            print(f"Training model ({len(self.model)} to train)...")
            i = 0
            for cluster_index, entry in self.model.items():
                coa_codes = entry['coa_codes']
                cluster_model = entry['model']

                data = self.data_df[self.data_df['coa_code'].isin(coa_codes)]

                if len(data) == 0:
                    print(f"Model {i} has no data.")
                else:
                    X = data[[f+"_freq" for f in self.features]]
                    y = data[self.response]

                    cluster_model.fit(X, y)
                i += 1

                if i % 1000 == 0:
                    print(f"Trained {i} submodels.")
            print("Trained model.")
        else:
            print("Saving passed in model.")
            self.model = model

    def predictWithModel(self, latitude, longitude):
        """
        Given a latitude and longitude, return a prediction for the set response variable.
        :param latitude: float
        :param longitude: float
        """
        print(f"Finding coa code for {latitude}, {longitude}...")
        # first get the coa code
        coa_code = self._findCoaCodeGivenLatLong(latitude, longitude)
        print(f"Found coa code {coa_code}.")

        if coa_code == None:
            return None

        # check if the coa code is present in the data, if not then get closest one with data
        if len(self.data_df[self.data_df['coa_code'] == coa_code]) == 0:
            code_len = len(coa_code)
            foundCloseMatch = False
            while not foundCloseMatch:
                code_len -= 1
                prefix = coa_code[:code_len]
                matching_codes = self.data_df[self.data_df['coa_code'].str.startswith(prefix)]
                if len(matching_codes) > 0:
                    coa_code = matching_codes.iloc[0]['coa_code']
                    foundCloseMatch = True
            print(f"No data found for this code, choosing the next best code {coa_code}...")

        
        print("Predicting with coa code...")
        modelToUse = None
        index_used = None
        for cluster_index, entry in self.model.items():
            if coa_code in entry['coa_codes']:
                modelToUse = entry['model']
                index_used = cluster_index

        if modelToUse is None:
            curr = modelToUse
            index = index_used
            while curr is None:
                index += 1
                if index not in self.model:
                    print("looped")
                    index = 0
                if self.model[index]['model'] != None:
                    curr = self.model[index]['model']
            modelToUse = curr
        
        # now get the data given the coa code
        data = self.data_df[self.data_df['coa_code'] == coa_code]
        X = data[[f+"_freq" for f in self.features]]

        # return a prediction given a coa code
        return modelToUse.predict(X)[0]

    def predictWithModel_coa(self, coa_code):
        """
        Given a latitude and longitude, return a prediction for the set response variable.
        This is a helper function to predict with the code (for model analysis).
        :param coa_code
        """
        if coa_code == None:
            return None

        # check if the coa code is present in the data, if not then get closest one with data
        if len(self.data_df[self.data_df['coa_code'] == coa_code]) == 0:
            code_len = len(coa_code)
            foundCloseMatch = False
            while not foundCloseMatch:
                code_len -= 1
                prefix = coa_code[:code_len]
                matching_codes = self.data_df[self.data_df['coa_code'].str.startswith(prefix)]
                if len(matching_codes) > 0:
                    coa_code = matching_codes.iloc[0]['coa_code']
                    foundCloseMatch = True
        
        modelToUse = None
        index_used = None
        for cluster_index, entry in self.model.items():
            if coa_code in entry['coa_codes']:
                modelToUse = entry['model']
                index_used = cluster_index

        if modelToUse is None:
            curr = modelToUse
            index = index_used
            if index is None:
                index = 0
            while curr is None:
                index += 1
                if index not in self.model:
                    print("looped")
                    index = 0
                if self.model[index]['model'] != None:
                    curr = self.model[index]['model']
            modelToUse = curr
        
        # now get the data given the coa code
        data = self.data_df[self.data_df['coa_code'] == coa_code]
        X = data[[f+"_freq" for f in self.features]]

        # return a prediction given a coa code
        return modelToUse.predict(X)[0]

    def nationallyTestModel(self, n=100, coordinates=None):
        """
        Nationally test the model by generating UK coordinates (or using submitted), and
        return the R2 value of the predicted result.
        """
        predictions = []
        actuals = []

        if coordinates is None:
        # way of generating coordinates to test within a England bounding box (I looked on google maps for this)
            lat_min=51.8
            lat_max=52.5
            lon_min=-1.0
            lon_max=1.0
            coordinates = [(random.uniform(lat_min, lat_max), random.uniform(lon_min, lon_max)) for _ in range(n)]

        #coordinates = [(51.582593, -0.026611), (52.851243, -1.259172)]
        # this was for testing, ignore this

        for lat, long in coordinates:
            predictions.append(self.predictWithModel(lat, long))
            coa_code = self._findCoaCodeGivenLatLong(lat, long)
            actuals.append(self.data_df[self.data_df["coa_code"] == coa_code].iloc[0][self.response])

        print(actuals)
        print(predictions)

        # filter out any errors due to lack of data or outside bounding box
        actuals = list(filter(lambda x: x is not None, actuals))
        predictions = list(filter(lambda x: x is not None, predictions))

        return r2_score(actuals, predictions)

        # kfold test the model using all national data

    def _findCoaCodeGivenLatLong(self, lat, long):
        """
        Given a longitude and latitude, query the database to see where the point falls under.
        :param lat: float
        :param long: float
        """
        query = f"""
        SELECT coa_code FROM geo_codes_information
        WHERE ST_Contains(
            wkt_geom,
            ST_GeomFromText(
                CONCAT('POINT(', {long}, ' ', {lat}, ')')
            )
        )
        """
        response = access.fetch_query_from_cloud_as_df(query, self.conn)
        
        # if we have an empty response, then try for lesser accuracy of coordinates, and return none if not found
        if response.empty:
            foundMatch = False
            while not foundMatch:
                if long % 1 == 0 or lat % 1 == 0:
                    return None
                long = float(str(long)[:-1])
                lat = float(str(lat)[:-1])
                query = f"""
                SELECT coa_code FROM geo_codes_information
                WHERE ST_Contains(
                    wkt_geom,
                    ST_GeomFromText(
                        CONCAT('POINT(', {long}, ' ', {lat}, ')')
                    )
                )
                """
                response = access.fetch_query_from_cloud_as_df(query, self.conn)

        return response.iloc[0]['coa_code']
