# This file contains code for suporting addressing questions in the data

"""# Here are some of the imports we might expect 
import sklearn.model_selection  as ms
import sklearn.linear_model as lm
import sklearn.svm as svm
import sklearn.naive_bayes as naive_bayes
import sklearn.tree as tree

import GPy
import torch
import tensorflow as tf

# Or if it's a statistical analysis
import scipy.stats"""

"""Address a particular question that arises from the data"""

"""
PRACTICAL 2 FUNCTIONS
"""

import matplotlib.pyplot as plt
import numpy as np

def examine_price_area_relationship(input_df):
    corr = input_df['price'].corr(input_df["area_sq_m"])
    print(f"The Pearson coefficient of correlation between price and area is: {corr}")

    # Draw graph accordingly
    fig, ax = plt.subplots(figsize=(10,6))

    a, b = np.polyfit(input_df['area_sq_m'], input_df['price'], 1)
    ax.scatter(input_df['area_sq_m'], input_df['price'], color="blue", label="amenity")
    ax.plot(input_df['area_sq_m'], a*input_df['area_sq_m']+b, color="blue")

    plt.xlabel("Area (m^2)")
    plt.ylabel("Price")
    plt.title("Price against area")
    plt.show()