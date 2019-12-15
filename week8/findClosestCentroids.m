function idx = findClosestCentroids(X, centroids)
    K = size(centroids, 1);
    M = size(X, 1);
    idx = zeros(M, 1);

    for x_idx=1:M
        x = X(x_idx, :);
         
        x_rep = repmat(x, K, 1);
        dists = vecnorm(x_rep - centroids, 2, 2).^2;
        [_, min_dist_k] = min(dists, [], 1);

        idx(x_idx) = min_dist_k;
    end
end

