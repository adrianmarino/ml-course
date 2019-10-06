function [theta, J_history] = gradientDescent(X, y, theta, alpha, num_iters) 
  m = length(y); % number of training examples
  J_history = zeros(num_iters, 1);
  for iter = 1:num_iters
    %           (mx2)    (2x1)
    y_predicted = X   *  theta; % (mx1)
    %            (mx1)    (mx1)
    y_error = y_predicted - y; % (mx1)

    %             (1xm)    (mx2)
    theta_diff = y_error' *  X; % 1X2
    %       (2x1)                   (2x1)
    theta = theta - (alpha / m) * theta_diff'; % (2x1)

    % Save the cost J in every iteration    
    J_history(iter) = computeCost(X, y, theta);
  end
end
