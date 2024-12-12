from . import access, assess
from sklearn.metrics import r2_score
import random
from sklearn.model_selection import cross_validate, KFold
from sklearn.linear_model import LinearRegression
from sklearn.tree import plot_tree
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.cluster import KMeans

# --------------
# TASK 1
# --------------

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
        Use the next nearest model if the model for the place does not exist.
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
        :param n: number of points to generate
        :param coordinates: optional list of coordinates to test
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

# --------------
# TASK 2
# same principles
# --------------

# DONE
def modelMaxDepth(d=6, n=5, max_features='sqrt'):
    """
    Helper function.
    Frame a wrapper for Random Forest classification that implements features to limit overfitting and complexity of model,
    akin to regularization techniques used for regression problems.
    :param d: max depth of each tree in random forest
    :param n: number of trees in random forest
    :param max_features: size of set of random features to choose at any node in the tree
    :returns: specified RandomForestClassifier initialized model
    """
    return RandomForestClassifier(max_depth=d, n_estimators=n, max_features=max_features)

# DONE
class Task2Model:
    """
    Class to implement a given classification model for the purpose of predicting whether or
    not a constituency is marginal or a stronghold in a given election.

    Has inbuilt functionality to use clustering i.e. K-Means++, with a separate Random Forest
    model for each cluster of the dataset fetched.
    """

    def __init__(
            self, 
            features, 
            conn=None, 
            response='responsevariable', 
            census_table='historical_census_constituencies', 
            model=RandomForestClassifier, 
            year=1983,
            clusters=1
        ):
        """
        Initialize instance variables in the model, and set up model structure.
        :param features: list of string features to use
        :param conn
        :param response
        :param census_table: in the cloud
        :param model: uninitialized model i.e. class/function
        :param year
        :param clusters: number of clusters (1 means no clustering)
        """
        self.model = model
        self.X = None
        self.y = None
        self.features = features
        self.response = response
        self.census_table = census_table
        self.conn = conn
        self.year = year
        self.clusters = clusters > 1
        self.k = clusters

        if self.clusters:
            self.model = {
                i:self.model for i in range(clusters)
            }
            self.Xs = [[] for i in range(clusters)]
            self.ys = [[] for i in range(clusters)]

    def fetchData(self):
        """
        Fetch the data to train on from the cloud, and segment if using clustering.
        """
        if self.conn is not None:
            query = f"select * from {self.census_table};"
            data = access.fetch_query_from_cloud_as_df(query, self.conn)
            data = assess.preprocess_census_df_task2(data, year=self.year)
            self.X = data[self.features]
            self.y = data[self.response]

            if self.clusters:
                # then we use kmeansplusplus to cluster groups of area codes
                km = KMeans(n_clusters=self.k, random_state=12, init='k-means++')
                data = self.X.copy()
                km.fit(data)

                for idx in range(len(km.labels_)):
                    label = km.labels_[idx]
                    self.Xs[label].append(idx)
                    self.ys[label].append(idx)

                self.Xs = [self.X.iloc[indices] for indices in self.Xs]
                self.ys = [self.y.iloc[indices] for indices in self.ys]

        else:
            print("error fetching data: conn is None")

    def initializeAndTrainModel(self, X=None, y=None, Xs=None, ys=None):
        """
        Initialize and train the model, using data if submitted, else fetched and
        stored data within the class. Train each model using cluster data if clustering.
        :param X: optional feature data if no clustering
        :param y: optional response data if no clustering
        :param Xs: optional list of feature data for each cluster if clustering
        :param ys: optional list of response data for each cluster if clustering
        """
        if self.clusters:
            for k in self.model.keys():
                self.model[k] = self.model[k]()
                if Xs is None or ys is None:
                    self.model[k].fit(self.Xs[k], self.ys[k])
                else:
                    self.model[k].fit(Xs[k], ys[k])
        else:
            self.model = self.model()
            if X is None or y is None:
                self.model.fit(self.X, self.y)
            else:
                self.model.fit(X, y)

    def initializeModel(self):
        """
        Initialize the model (based on whether or not clusters are enabled).
        """
        if self.clusters:
            for k in self.model.keys():
                self.model[k] = self.model[k]()
        else:
            self.model = self.model()

    def predictGivenConstituency(self, constituency, year):
        """
        Given the name of a valid constituency, and the year for that constituency,
        get the prediction of the response variable for this constituency.
        :param constituency: name
        :param year: should be year of data you have
        :returns: array either 0 or 1 for stronghold or marginal respectively
        """
        if year not in set([1974, 1983, 2024]):
            print("wrong year constituency given, try again")
        else:
            data = self._fetchDataGivenConstituency(constituency, year)
            data = assess.preprocess_census_df_task2(data, year=self.year)
            X = data[self.features]
            if self.clusters:
                # we implicitly use clusters by getting prediction with highest confidence
                predictions = [
                    [k, v.predict(X), max(v.predict_proba(X)[0])] for k, v in self.model.items()
                ]
                return max(predictions, key=lambda x:x[2])[1]
            else:
                return self.model.predict(X)

    def _fetchDataGivenConstituency(self, constituency, year):
        """
        Helper function
        Given a constituency, fetch its data.
        :param constituency: name
        :param year
        :returns: fetched features from the cloud
        """
        if year in set([1974, 1983]):
            query = f"""
            select ce.*
            from constituencies_{year}_with_lookup as co
            right join {self.census_table} as ce
            on co.lookup_census=ce.mnemonic
            where co.geom_name='{constituency}'
            limit 1;
            """
            return access.fetch_query_from_cloud_as_df(query, self.conn)
        elif year == 2024:
            query = f"""
            select * from 2024_census_constituencies
            where geography='{constituency}'
            limit 1;
            """
            return access.fetch_query_from_cloud_as_df(query, self.conn)
    
    def showModelDecisionTreeInterpretation(self):
        """
        Plot all of the decision trees from the Random Forest to interpret model.
        """
        if self.clusters:
            for m in self.model.values():
                for i in range(len(m.estimators_)):
                    plt.figure(figsize=(15, 15))
                    plot_tree(m.estimators_[i], feature_names=self.features, class_names=['stronghold', 'marginal'], filled=True, fontsize=6)
                    plt.show()
        else:
            for i in range(len(self.model.estimators_)):
                plt.figure(figsize=(15, 15))
                plot_tree(self.model.estimators_[i], feature_names=self.features, class_names=['stronghold', 'marginal'], filled=True, fontsize=6)
                plt.show()

    def kFoldTestModel(self, k, return_train_score=False, initializeModel=True, metrics=['accuracy', 'precision', 'recall', 'f1', 'roc_auc']):
        """
        K-Fold test the model nationally.
        :param k: number of folds
        :param return_train_score: boolean whether or not to run metrics on train data too
        :param initializeModel: boolean whether or not to initialize the model before testing
        :param metrics: list of metrics to get
        :returns: DataFrame of metrics for each fold
        """
        scoring = {i:i for i in metrics}

        if initializeModel:
            self.initializeModel()
        
        kf = KFold(n_splits=k, shuffle=True, random_state=12)

        if self.clusters:
            scores_s = [
                pd.DataFrame(cross_validate(
                    self.model[model_idx],
                    self.Xs[model_idx],
                    self.ys[model_idx].astype(int),
                    cv=kf,
                    scoring=scoring,
                    return_train_score=return_train_score
                )) for model_idx in set(self.model.keys())
            ]
            # ignore the index so we can group for each of the models, and avg
            scores = pd.concat(scores_s, ignore_index=True)
            scores = scores.groupby(scores.index).mean()
            return scores
        else:
            scores = cross_validate(
                self.model,
                self.X,
                self.y.astype(int),
                cv=kf,
                scoring=scoring,
                return_train_score=return_train_score
            )
            return pd.DataFrame(scores)
    
    def plotModelPredictionPower(self, k_lower=2, k_upper=10):
        """
        K-Fold test for a range of k values, and plot the chart of metrics.
        Should be run on a brand new class instantiation with fetched data.
        :param k_lower: inclusive
        :param k_upper: inclusive
        """
        metrics = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']
        scores = self.kFoldTestModel(k=k_lower, return_train_score=True, metrics=metrics)
        scores['k'] = k_lower
        for k in range(k_lower+1, k_upper+1):
            to_add = self.kFoldTestModel(k=k, return_train_score=True, initializeModel=False, metrics=metrics)
            to_add['k'] = k
            scores = pd.concat(
                [
                    scores,
                    to_add
                ]
            )
        scores = scores.reset_index(drop=True)
        scores = scores.groupby('k').mean().reset_index(drop=False)
        scores = scores.melt(
            id_vars='k',
            value_vars=["test_"+m for m in metrics]+["train_"+m for m in metrics],
            var_name='metric',
            value_name='score'
        )

        sns.lineplot(data=scores, x='k', y='score', hue='metric')
        plt.title("Metrics for model used in classification of swing vs stronghold")