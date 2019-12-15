function idx = findClosestCentroids(X, centroids)
    K = size(centroids, 1);
    M = size(X, 1);
    idx = zeros(M, 1);

    # For each X row ((x,y) point)...
    for x_idx=1:M
        x_row = X(x_idx, :);
 
        # Get matrix with "k" x_rows(repeated)...
        x_rep = repmat(x_row, K, 1);

        # Distance bethween x and each centroid...
        dists = vecnorm(x_rep - centroids, 2, 2).^2;
 
        # Get centroid "k" with min distance to x...
        [_, min_dist_k] = min(dists, [], 1);

        idx(x_idx) = min_dist_k;
    end
end

