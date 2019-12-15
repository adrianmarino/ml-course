function centroids = computeCentroids(X, idx, K)
    centroids = zeros(K, size(X, 2));
    for k=1:K
        x_k = X(idx == k, :);
        c_k = size(x_k, 1);
        centroids(k, :) = sum(x_k) / c_k;
    end
end

