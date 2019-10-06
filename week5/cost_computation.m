function [J] = cost_computation(Theta1, Theta2, num_labels, X, y, lambda)
    % Setup some useful variables
    m = size(X, 1);

    h_theta    = feedforward(Theta1, Theta2, X, y); % (5000x10)

    yv          = to_one_hot_vector(y, num_labels); % (5000x10)

    %         (5000x10)  (5000x10)
    part_1     = -yv  .* log(h_theta); % (5000x10)

    %         (5000x10)  (5000x10)
    part_2     = (1-yv) .* log(1 - h_theta);  % (5000x10)

    error_term = sum(sum(part_1 - part_2)); % (1x1)

    J          = 1/m * error_term + regularization(Theta1, Theta2, lambda, m); % (1x1)
end