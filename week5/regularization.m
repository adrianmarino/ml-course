function [regularization_term] = regularization(Theta1, Theta2, lambda, m)
    theta1_without_zero = Theta1(:, 2:end); % (25x400)
    theta2_without_zero = Theta2(:, 2:end); % (10x25)
    regularization_term = (lambda/(2 * m)) * sum(sum(theta1_without_zero.^2)) * sum(sum(theta2_without_zero.^2)); % (1x1)
end