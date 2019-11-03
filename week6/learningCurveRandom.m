function [error_train, error_val] = learningCurveRandom(X, y, Xval, yval, lambda)
    m = size(X, 1)
    m_val = size(Xval, 1)
  
    process_times = 100
    batc_size = 4
    error_train = zeros(m, 1);
    error_val   = zeros(m, 1);

    for i = 1:m
        t_error_train = zeros(m, 1);
        t_error_val   = zeros(m, 1);

        for t = 1:process_times
            train_batch_size = batc_size;

            train_indexes = randi([1, m], [1, train_batch_size]);
            x_train = X(train_indexes,:);
            y_train = y(train_indexes);

            val_batch_size = batc_size;
            val_indexes = randi([1, m_val], [1, val_batch_size]);
            x_val = Xval(val_indexes,:);
            y_val = yval(val_indexes);

            theta = trainLinearReg(x_train, y_train, lambda);

            t_error_train(i, :) = linearRegCostFunction(x_train, y_train, theta, 0);
            t_error_val(i, :) = linearRegCostFunction(x_val, y_val, theta, 0);
        end

        error_train(i, :) = mean(t_error_train);
        error_val(i, :) = mean(t_error_val);
    end
end
