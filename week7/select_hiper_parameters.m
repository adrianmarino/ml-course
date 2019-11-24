function [C, sigma] = select_hiper_parameters(X, y, Xval, yval, c_values, sigma_values)
    min_error = 1000000;
    for current_c = c_values
        for current_sigma = sigma_values
            model = svmTrain(X, y, current_c, @(x1, x2) gaussianKernel(x1, x2, current_sigma));
            predictions = svmPredict(model, Xval);
    
            error = mean(double(predictions ~= yval));
            if error < min_error
                min_error = error;
                C = current_c;
                sigma = current_sigma;
            end
        end
    end
    fprintf(['Best hiperparameters:  C = %f, sigma = %f :'], C, sigma);
end
