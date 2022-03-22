import pandas as pd
import numpy as np
import random


def kmeans(X, k, centroids=None, max_iter=30, tolerance=0.01):
    if centroids == 'kmeans++':

        first_centroid = random.sample(list(X), 1)
        curr_centroids = first_centroid
        for j in range(k-1):
            dists_ = np.array([float('inf')]*X.shape[0])
            dists = [dists_] # make list of array, to get array with dimension more than 1 in step *
            for centroid in curr_centroids:
                dist = [np.linalg.norm(X - centroid, axis = 1)]
                dists = np.concatenate((dists, dist), axis = 0) # step *
            dists = dists.T
            min_idx = np.argmin(dists, axis = 1)
            min_dist = [dists[i][min_idx[i]] for i in range(X.shape[0])]
            next_cent_idx = np.argmax(min_dist, axis = 0)
            next_centroid = [X[next_cent_idx]]
            curr_centroids = np.concatenate((curr_centroids, next_centroid), axis = 0)

    else:
        x_unique = np.unique(X, axis = 0)
        curr_centroids = random.sample(list(x_unique), 3)
    dist_centroids = np.array([float('inf') for _ in range(k)])

    while np.any(dist_centroids) >= tolerance:
        clusters = [[] for _ in range(k)]
        for i in range(X.shape[0]):
            cluster = np.argmin(np.linalg.norm(X[i] - curr_centroids, axis=1))
            clusters[cluster].append(i)
        next_centroids = np.array([np.mean(X[clusters[j]], axis=0) for j in range(k)])
        dist_centroids = np.linalg.norm(curr_centroids - next_centroids, axis=1)
        curr_centroids = next_centroids
    clusters_map = [0]*X.shape[0]
    for i in range(len(clusters)):
        for j in range(len(clusters[i])):
            clusters_map[clusters[i][j]] = i
    return next_centroids, clusters_map
