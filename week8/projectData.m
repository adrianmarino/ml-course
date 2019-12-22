function Z = projectData(X, U, K)
    m = size(X, 1);

    U_reduced = U(:, 1:K);
    Z = zeros(m, K);

    for i = 1:m
        #          (1x2)      (2x1)
        Z(i, :) = X(i, :) * U_reduced; # (1x1)
    end
end
