function [X_poly] = polyFeatures(x, p)
    X_poly = [x, zeros(size(x, 1), ifelse(p <= 1, 1, p-1))];
    for i = 2:p
        X_i = x.^i;
        X_poly(:, i) = X_i;
    end
end
