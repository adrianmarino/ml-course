function [y] = to_one_hot_vector(labels, column_size)
    y = zeros(size(labels), column_size);

    for label_index = 1:size(labels)
        y(label_index, labels(label_index)) = 1;
    endfor
end