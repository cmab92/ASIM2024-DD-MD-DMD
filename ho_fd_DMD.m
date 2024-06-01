function sys = ho_fd_DMD(X1, X2)
    %% structured DMD implementing a higher-order finite-difference model
    N = size(X1, 1);
    I = eye(N);
    S(:, :, 1) = I;
    S(:, :, 2) = toeplitz(I(:, 2), 0*I(:, 2));
    S(:, :, 3) = toeplitz(0*I(:, 2), I(:, 2));
    S(:, :, 4) = toeplitz(I(:, 3), 0*I(:, 3));
    S(:, :, 5) = toeplitz(0*I(:, 3), I(:, 3));
    S(1, 2, 6) = 1;
    S(1, 3, 7) = 1;
    S(2, 3, 8) = 1;
    S(N, N-1, 9) = 1;
    S(N, N-2, 10) = 1;
    S(N-1, N-2, 11) = 1;
    S(1, 1, 12) = 1;
    S(2, 2, 13) = 1;
    S(N, N, 14) = 1;
    S(N-1, N-1, 15) = 1;
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