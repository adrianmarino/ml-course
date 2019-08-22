function [J, grad] = lrCostFunction(theta, X, y, lambda)
    # n = Number of theta terms.
    # m = Number of training examples.
    #
    m = length(y);
    #
    # =======================================================
    # Prediction:
    # =======================================================
    %        (mxn)  (nx1)
    h_theta =  X  * theta; % (mx1)
    y_predicted = sigmoid(h_theta); % (mx1)
    #
    #
    # =======================================================
    # Cost function:
    # =======================================================
    %           (1xm)        (mx1)
    part_1     = -y'  * log(y_predicted); % (1x1)
    %            (1xm)          (mx1)
    part_2     = (1-y') * log(1 - y_predicted); % (1x1)
    
    parts_diff  = part_1 - part_2; % (1x1)

    # theta_1: without zero component.
    theta_1 = theta(2:end);

    cost_regularization_term = (lambda/(2 * m)) * sum(theta_1.^2); # (1x1)

    J = 1/m * parts_diff + cost_regularization_term; % (1x1)
    # =======================================================
    #
    #
    #
    # =======================================================
    # Gradient of the cost function:
    # =======================================================
    %            (mx1)    (mx1)
    y_error = y_predicted - y; % (mx1)

    # theta_2: Replace first comonent to zero.
    theta_2 = [0; theta(2:end)];

    gradient_regularization_term = (lambda / m) * theta_2; # (nx1)

    %            (nxm)    (mx1)
    grad = 1/m * ( X'  * y_error) + gradient_regularization_term; % (nx1)
    # =======================================================
end
