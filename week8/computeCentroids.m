function centroids = computeCentroids(X, idx, K)
    centroids = zeros(K, size(X, 2));
    for k=1:K
        # Get position for all elements equals to k
        k_indexes = idx == k;

        # Get x_rows with k_indexes... 
        x_k = X(k_indexes, :);

        # x_k count...
        c_k = size(x_k, 1);

        # Calculate next k centroid position... 
        centroids(k, :) = sum(x_k) / c_k;
    end
end

