function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)

Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1)); % (25x401)

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1)); % (10x26)

% Setup some useful variables
m = size(X, 1);

% Part 1

h_theta    = feedforward(Theta1, Theta2, X, y); % (5000x10)

y          = to_one_hot_vector(y, num_labels); % (5000x10)

%         (5000x10)  (5000x10)
part_1     = -y  .* log(h_theta); % (5000x10)

%         (5000x10)  (5000x10)
part_2     = (1-y) .* log(1 - h_theta);  % (5000x10)

error_term = sum(sum(part_1 - part_2)); % (1x1)

J          = 1/m * error_term + regularization(Theta1, Theta2, lambda, m); % (1x1)

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

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];

end
