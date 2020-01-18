function [J, grad] = cofiCostFunc(params, Y, R, num_users, num_movies, ...
                                  num_features, lambda)

    % Y: num_movies x num_users matrix of user ratings of movies.
    % R: num_movies x num_users matrix, where R(i, j) = 1 if the 
    %    i-th movie was rated by the j-th user.

    % X: num_movies  x num_features matrix of movie features.
    X = reshape(params(1:num_movies*num_features), num_movies, num_features); % (5x3)

    % Theta: num_users x num_features matrix of user features.
    Theta = reshape(params(num_movies*num_features+1:end), num_users, num_features); % (4x3)

    % Calculate J Cost

    % Error value for each movie vs. user. ERROR(i,j).
    %      (5x3)  (3x4)  (5x4)  (5x4)    
    Error = (X  * Theta' - Y)  .* R; % (5x4)

    j_reg = lambda/2 * (sum(sum(Theta.^2)) + sum(sum(X.^2)));

    J = 1/2 * sum(sum(Error.^2)) + j_reg;

    % X_grad: num_movies x num_features matrix, containing the partial derivatives w.r.t. to each element of X.
    % Theta_grad: num_users x num_features matrix, containing the partial derivatives w.r.t. to each element of Theta.

    X_grad_reg = lambda * X

    %         (5x4)   (4x3)
    X_grad = Error * Theta + X_grad_reg; % (5x3)

    Theta_gra_reg = lambda * Theta;

    %           (4x5)   (5x3)
    Theta_grad = Error' * X + Theta_gra_reg; % (4x3)

    grad = [X_grad(:); Theta_grad(:)];
end
