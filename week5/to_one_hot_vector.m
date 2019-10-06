function [y] = to_one_hot_vector(labels, column_size)
    y = zeros(size(labels), column_size);

    for label_index = 1:size(labels)
        one_hot_vector = [1:column_size] == labels(label_index);
        y(label_index, :) = one_hot_vector;
    endfor
end