function [U, S] = pca(X)
    m = size(X, 1); 
    sigma_matrix = (X' * X) / m;
    [U, S, _] = svd(sigma_matrix);
end
