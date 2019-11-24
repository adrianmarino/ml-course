function [C, sigma] = dataset3Params(X, y, Xval, yval)
    # Select best values for C and Sigma hiper párameters...
    # c_values = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30];
    # sigma_values = [0.05 : 0.1 : 1.3];
    # [C, sigma] = select_hiper_parameters(X, y, Xval, yval, c_values, sigma_values);

    # Use best hiper parameters discovered with above select_hiper_parameters invocation:
    C = 1;
    sigma = 0.15; 
end
