function p = predict(Theta1, Theta2, X)
    # =======================================
    # Feedforward Propagation and Prediction
    # =======================================

    # Examples number...
    m1 = size(X, 1);

    % Add unit zero to layer 1 (x0=1)
    a1 = [ones(m1, 1) X]; # (5000x401)

    #   (5000x401)  (401x25)
    z2 =     a1    *  Theta1'; # (5000x25)

    a2 = sigmoid(z2); % (5000x25)

    % Add unit zero to layer 2 (x0=1)
    m2 = size(a2, 1);
    a2 = [ones(m2, 1) a2]; # (5000x26)

    #    (5000x26)   (26x10)
    z3 =     a2   *  Theta2'; # (5000x10)

    all_probality_distributions = sigmoid(z3); # (5000x10)

    # Get index of max pobability for each 'm'...
    [_, p] = max(all_probality_distributions, [], 2);  % (5000x1)
end