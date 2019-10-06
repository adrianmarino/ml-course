function [all_theta] = oneVsAll(X, y, classes_size, lambda)
    m = size(X, 1);
    n = size(X, 2);
 
    all_theta = zeros(classes_size, n + 1);

    % Add ones first column to X data matrix
    X = [ones(m, 1) X];

    options = optimset('GradObj', 'on', 'MaxIter', 50);

    for class_num = 1:classes_size
        class_one_hot_vector = y == class_num;

        theta_minimization_callback = @(current_theta)(
            lrCostFunction(
                current_theta,
                X, 
                class_one_hot_vector, 
                lambda
            )
        );

        initial_theta = zeros(n + 1, 1);

        [class_min_theta] = fminunc(
            theta_minimization_callback, 
            initial_theta, 
            options
        );
    
        all_theta(class_num, :) = [class_min_theta];
end
