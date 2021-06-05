from flask import Flask, request, jsonify

import os, random, re, json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans


#Init app
app = Flask(__name__)
basedir = os.path.abspath(os.path.dirname(__file__))

#Number of Clusters to be formed
CLUSTER_NUMBERS = 200

def __cluster_helper(json_object):

    df = pd.read_json(json_object)
    df.dropna(axis=0,how='any',subset=['latitude','longitude'], inplace=True)
    # Variable with the Longitude and Latitude


    rand = lambda: random.randint(0,255)
    colordict = {i: '#%02X%02X%02X' % (rand(),rand(),rand()) for i in range(CLUSTER_NUMBERS)}

    kmeans = KMeans(n_clusters = CLUSTER_NUMBERS, init ='k-means++')
    kmeans.fit(df[df.columns[1:3]]) # Compute k-means clustering.
    df['cluster_label'] = kmeans.fit_predict(df[df.columns[1:3]])
    centers = kmeans.cluster_centers_ # Coordinates of cluster centers.

    label_groups = df.groupby("cluster_label")

    store = {}

    for i in range(len(centers)):
        group = label_groups.get_group(i)["ik_number"].to_numpy()    
        group = np.append(group, centers[i])
        new_group = group.tolist()
        store[i] = new_group

    return store
    


@app.route('/clusters_info', methods=["GET"])
def get_clusters():
    json_object = json.dumps(request.get_json())

    res = __cluster_helper(json_object)

    res_json = json.dumps(res)
    return res_json


# Run Server
if __name__ == "__main__":
    app.run(debug=False)

