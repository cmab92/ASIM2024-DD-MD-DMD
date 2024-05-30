function sys = vanilla_pe_DMD(X1, X2)
    %% structured DMD implementing parameter estimation
    N = size(X1, 1);
    I = eye(N);
    S = -2*I;
    S = S + toeplitz(I(:, 2), 0*I(:, 2));
    S = S + toeplitz(0*I(:, 2), I(:, 2));
    S(1, 2) = 2;
    S(N, N-1) = 2;
    %%
    Z = trace(X1'*(S'*S)*X1);
    c = trace(X1'*S'*(X2-X1));
    a = c/Z;
    sys = a*S + I;
end