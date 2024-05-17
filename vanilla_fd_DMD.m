function sys = vanilla_fd_DMD(X1, X2)
    %% structured DMD implementing a finite-difference model
    N = size(X1, 1);
    I = eye(N);
    S(:, :, 1) = I;
    S(:, :, 2) = toeplitz(I(:, 2), 0*I(:, 2));
    S(:, :, 3) = toeplitz(0*I(:, 2), I(:, 2));
    S(1, 2, 4) = 1;
    S(N, N-1, 5) = 1;
    Q = size(S, 3);
    Z = zeros(Q);
    c = zeros([Q, 1]);
    for k = 1:Q
        for i = k:Q
            Z(k, i) = trace(X1'*S(:, :, i)'*S(:, :, k)*X1);
        end
        c(k) = trace(X1'*S(:, :, k)'*X2);
    end
    Z = Z + Z' - diag(diag(Z));
    a = pinv(Z)*c;
    sys = 0;
    for q = 1:Q
        sys = sys + a(q)*S(:,:,q);
    end
end