function [J, grad] = linearRegCostFunction(X, y, theta, lambda)
    m = length(y); % number of training examples

    %   (12x2)  (2x1)
    Hx =  X  * theta; % (12x1)

    %            (12x1)   (12x1)
    pred_error =   Hx   -   y;   % (12x1)

    mse = 1/(2*m) * sum(pred_error.^2);
    reg_term = lambda/(2*m) * sum(theta(2:end).^2)
    J = mse + reg_term;

    grad(1) = 1/m * sum(pred_error .* X(:, 1));
    grad(2) = 1/m * sum(pred_error .* X(:, 2)) + lambda/m * theta(2);
    grad = grad(:);
end
