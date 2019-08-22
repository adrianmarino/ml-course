function p = predictOneVsAll(all_theta, X)
    # Examples number...
    m = size(X, 1);

    % Add theta-zero column...
    X = [ones(m, 1) X]; # (5000x401)

    %             (5000x401)   (401x10)
    all_h_theta =     X     * all_theta'; % (5000x10)

    all_y_probabilities = sigmoid(all_h_theta); % (5000x10)

    # Get index of max pobability for each 'm'...
    [_, p] = max(all_y_probabilities, [], 2);  % (5000x1)
end
