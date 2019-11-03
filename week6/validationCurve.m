function [lambda_vec, error_train, error_val] = validationCurve(X, y, Xval, yval)
    % Selected values of lambda (you should not change this)
    lambda_vec = [0 0.001 0.003 0.01 0.03 0.1 0.3 1 3 10]';

    % You need to return these variables correctly.
    error_train = zeros(length(lambda_vec), 1);
    error_val = zeros(length(lambda_vec), 1);

    for i = 1:length(lambda_vec)
        lambda = lambda_vec(i);

        # Train linear regresion with selected lambda reg parameter and get trainer theta vector...
        theta = trainLinearReg(X, y, lambda);
 
        # Compute cost over train set for theta vector with "lamdba == zero"...
        error_train(i, :) = linearRegCostFunction(X, y, theta, 0);

        # Compute cost over calisation set for theta vector with "lamdba == zero"...
        error_val(i, :) = linearRegCostFunction(Xval, yval, theta, 0);

        # Why lambda == zero?
        # Because, lambda is hiper-paramter used to limit theta's max values over training process only. 
    end
end
