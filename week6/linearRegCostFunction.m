function [J, grad] = linearRegCostFunction(X, y, theta, lambda)
    m = length(y); % number of training examples

    %   (12x2)  (2x1)
    Hx =  X  * theta; % (12x1)

    %            (12x1)   (12x1)
    pred_error =   Hx   -   y;   % (12x1)

    J = (1/(2*m)) * sum(pred_error.^2) + (lambda/(2*m)) * sum(theta(2:end).^2);

    grad(1) = (1/m) * sum(pred_error .* X(:, 1));
    grad(2) = (1/m) * sum(pred_error .* X(:, 2)) + (lambda/m) * sum(theta(2:end).^2);
    grad = grad(:);

    G = (lambda/m) .* theta;
    G(1) = 0; % this is always 0

    grad = ((1/m) .* X' * (X*theta - y)) + G;
end
