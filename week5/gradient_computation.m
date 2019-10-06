function [Theta1_grad, Theta2_grad] = gradient_computation(Theta1, Theta2, num_labels, X, y, lambda)
    % Setup some useful variables
    m = size(X, 1);

    % Part 2

    Theta1_grad = zeros(size(Theta1)); % (25x401)
    Theta2_grad = zeros(size(Theta2)); % (10x26)

    for sample_index = 1:m
        xi = [1; X(sample_index, :)']; % (401x1) Add bias.
        yi = y(sample_index); % (1x1)

        a1 = xi;

        %   (25x401)  (401x1)
        z2 = Theta1  *  a1; % (25x1)

        a2 = sigmoid(z2);  % (25x1)

        a2 = [1; a2]; % (26x1) Add bias.

        %    (10x26)  (26X1) 
        z3 = Theta2  * a2; % (10x1)

        a3 = sigmoid(z3); % (10x1)

        yv = to_one_hot_vector(yi, num_labels); % (1x10)
        yv = yv'; % (10x1)

        delta3 = a3 - yv; % (10x1)

        z2_bias = [1; z2]; % (26x1) 

        %        (26x10)    (10x1)             (1x1)
        delta2 = (Theta2' * delta3) .* sigmoidGradient(z2_bias); % (26x1)
        delta2 = delta2(2:end); % Remove a2_0 (Layer two bias)  % (25x1)

        %                           (25x1)  (1x401)
        Theta1_grad = Theta1_grad + delta2 * a1'; % (25x401)
        %                           (10x1) (1x26)
        Theta2_grad = Theta2_grad + delta3 * a2'; % (10x26)
    endfor

    % Part 3

    Theta1_grad = (1/m) * Theta1_grad + (lambda/m) * [zeros(size(Theta1, 1), 1), Theta1(:, 2:end)];
    Theta2_grad = (1/m) * Theta2_grad + (lambda/m) * [zeros(size(Theta2, 1), 1), Theta2(:, 2:end)];
end
