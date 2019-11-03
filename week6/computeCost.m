function [J] = computeCost(X, y, theta, lambda)
fprintf("PASO 0")

    m = length(y); % number of training examples
    
    
fprintf("PASO 0")

    %   (12x2)  (2x1)
    Hx =  X  * theta; % (12x1)
    

fprintf("PASO 1")

    %            (12x1)   (12x1)
    pred_error =   Hx   -   y;   % (12x1)

fprintf("PASO 2")

    J = (1/(2*m)) * sum(pred_error.^2) + (lambda/(2*m)) * sum(theta(2:end).^2);

    fprintf('J: %f', J)
end
