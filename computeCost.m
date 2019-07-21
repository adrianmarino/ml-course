function J = computeCost(X, y, theta)
  m = length(y); % number of training examples
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
end
