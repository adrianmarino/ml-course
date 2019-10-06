function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)

    Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                    hidden_layer_size, (input_layer_size + 1)); % (25x401)

    Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                    num_labels, (hidden_layer_size + 1)); % (10x26)

    % Part 1

    J = cost_computation(Theta1, Theta2, num_labels, X, y, lambda);

    % Part 2 and 3

    [Theta1_grad, Theta2_grad] = gradient_computation(Theta1, Theta2, num_labels, X, y, lambda);

    % Unroll gradients
    grad = [Theta1_grad(:); Theta2_grad(:)];
end