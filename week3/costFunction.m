function [J, grad] = costFunction(theta, X, y)
    % m = 20
    m = length(y); % number of training examples

    %        (mx3)  (3x1)
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

    J = 1/m * parts_diff; % (1x1)
    # =======================================================
    #
    #
    #
    # =======================================================
    # Gradient of the cost function:
    # =======================================================
    %            (mx1)    (mx1)
    y_error = y_predicted - y; % (mx1)

    %            (3xm)    (mx1)
    grad = 1/m * ( X'  * y_error); % (3x1)
    # =======================================================
end
