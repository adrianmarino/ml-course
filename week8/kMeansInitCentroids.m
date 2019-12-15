function centroids = kMeansInitCentroids(X, K)
    # Number of examples...
    m = size(X, 1);

    # Get array with m random values bethween "1..m"... 
    random_indexes = randperm(m);

    # Get first K indexes...
    centroid_indexes = random_indexes(1:K);

    # Get centroids by index...
    centroids = X(centroid_indexes, :);
end

