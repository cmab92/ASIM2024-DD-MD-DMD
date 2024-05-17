function sys = fd_DMD(X1, X2)
    %% structured DMD implementing a finite-difference model
    N = size(X1, 1);
    Q = 5;
    Z = zeros(Q);
    c = zeros([Q, 1]);
    %%
    X11 = X1;
    X12 = circshift(X1, 1, 1);
    X12(1, :) = 0;
    X13 = circshift(X1, -1, 1);
    X13(end, :) = 0;
    X14 = zeros(size(X1));
    X14(1, :) = X1(2, :);
    X15 = zeros(size(X1));
    X15(end,:) = X1(end-1, :);
    % row 1
    Z(1,1) = trace(X11'*X11);
    Z(1,2) = trace(X12'*X11);
    Z(1,3) = Z(1,2); % Z(1,3) = trace(X13'*X11);
    Z(1,4) = trace(X14'*X11);
    Z(1,5) = trace(X15'*X11);
    c(1) = trace(X11'*X2);
    % row 2
    Z(2,2) = trace(X12'*X12);
    Z(2,3) = trace(X13'*X12);
    Z(2,4) = 0; % Z(2,4) = trace(X14'*X12);
    Z(2,5) = trace(X15'*X12);
    c(2) = trace(X12'*X2);
    % row 3
    Z(3,3) = trace(X13'*X13);
    Z(3,4) = trace(X14'*X13);
    Z(3,5) = 0; % Z(3,5) = trace(X15'*X13);
    c(3) = trace(X13'*X2);
    % row 4
    Z(4,4) = Z(3,4); % Z(4,4) = trace(X14'*X14);
    Z(4,5) = 0; % Z(4,5) = trace(X15'*X14);
    c(4) = trace(X14'*X2);
    % row 5
    Z(5,5) = Z(2, 5); % Z(5,5) = trace(X15'*X15);
    c(5) = trace(X15'*X2);
    %%
    Z = Z + Z' - diag(diag(Z));
    a = pinv(Z)*c;
    I = eye(N);
    %%
    sys = a(1)*I + a(2)*toeplitz(I(:, 2), 0*I(:, 2)) + a(3)*toeplitz(0*I(:, 2), I(:, 2));
    sys(1, 2) = sys(1, 2) + a(4);
    sys(N, N-1) = sys(N, N-1) + a(5);
end