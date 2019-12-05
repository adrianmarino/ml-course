function idx = findClosestCentroids(X, centroids)
    K = size(centroids, 1);
    M = size(X, 1);
    idx = zeros(M, 1);

    for x_idx=1:M
        x_row = X(x_idx, :)
        x_row_rep = repmat(x_row, K, 1);
        dists = vecnorm(x_row_rep - centroids, 2, 2).^2;
        [_, min_dist_k] = min(dists, [], 1);
        idx(x_idx) = min_dist_k;
    end
end

