function x = emailFeatures(word_indices)
    % Total number of words in the dictionary
    n = 1899;

    % You need to return the following variables correctly.
    x = zeros(n, 1);

    for word_index = word_indices
        x(word_index) = 1;
    end
end
