function [a3] = feedforward(Theta1, Theta2, X, y)
    m = size(X, 1);

    % Add ones first column to X data matrix
    X = [ones(m, 1) X]; % 5000x401

    a1 = X;

    %    (5000x401)(401x25)    
    z2 =     a1  *  Theta1'; % (5000x25)

    a2 = sigmoid(z2);  % (5000x25)

    a2_0 = ones(size(a2, 1), 1); % 5000x1

    a2 = [a2_0, a2]; % 5000x26

    %    (5000x26)(26X10)    
    z3 =     a2  *  Theta2'; % (5000x10)

    a3 = sigmoid(z3); % (5000x10)
end
