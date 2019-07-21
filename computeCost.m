function J = computeCost(X, y, theta)
m = length(y); % number of training examples
%1.
preffix = 1 / (2 * m);
% 2.        (mx2) (2x1)
y_predicted = X * theta; % (mx1)
% 3.          (mx1)    (mx1) 
y_error = (y_predicted - y).^2; % (mx1)
% 4.
J = preffix * sum(y_error);
end
