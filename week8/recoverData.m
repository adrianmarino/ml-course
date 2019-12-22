function X_rec = recoverData(Z, U, K)
    m_examples = size(Z, 1);
    n_features = size(U, 1);

    # Z (15x5)
    X_rec = zeros(m_examples, n_features); # 15x11
    
    U_reduced = U(:, 1:K); # 11x5

    for i = 1:m_examples
        #              (11x5)      (5x1)
        X_rec(i, :) = U_reduced * Z(i, :)'; # (11x1)
    end
end
