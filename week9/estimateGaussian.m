function [mu sigma2] = estimateGaussian(X)
    [m, n] = size(X);

    mu = sum(X) * 1/m; % (1x2)
    % Note: sum function sum matrix columns.

    dif = (X - mu).^2; % (307x2)
    sigma2 = sum(dif) * 1/m; % (1x2)
end
