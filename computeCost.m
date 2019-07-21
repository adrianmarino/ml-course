function J = computeCost(X, y, theta)
%COMPUTECOST Compute cost for linear regression
%   J = COMPUTECOST(X, y, theta) computes the cost of using theta as the
%   parameter for linear regression to fit the data points in X and y

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta
%               You should set J to the cost.

%1.
 preffix = 1 / (2 * m)

% 2. 
%                        (1x2)       (2xm)
y_predicted = theta'  *     X'  ;  %   (1xm)

% 3.
 %                 (mx1)          (mx1) 
y_diff = (y_predicted'   -   y    ).^2; %  (mx1)

% 4.
J = preffix * sum(y_diff)

% =========================================================================

end
