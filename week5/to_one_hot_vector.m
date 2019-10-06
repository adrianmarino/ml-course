function [yr] = to_one_hot_vector(y, num_labels)
    yr = zeros(size(y), num_labels);

    for i = 1:size(y)
        yr(i, y(i)) = 1;
    end
end