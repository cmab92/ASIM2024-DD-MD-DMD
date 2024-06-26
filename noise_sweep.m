close all; clear all; clc; 
%%
rng(1);
%% gen. data
n_samples_x = 21;
k = 401;  % W/(m*K) thermal conductivity (copper)
c = 385;  % J/(kg*K) specific thermal capacity (copper)
r = 8.96; % kg/(m^3), mass density (copper)    
a = k/(r*c); % diffusivity constant
%%
f = 1;  % control sampling (stability condition)
train_test_ratio = 0.25;
%%
n_iter = 100;
noise_lvl = 0.005;
num_methods = 6;
sweep_var = [0.0, 0.0005, 0.001, 0.002, 0.004, 0.006, 0.008, 0.01, 0.02, 0.03, 0.04, 0.05]; % noise
sweep_len = length(sweep_var);
pred_error = zeros([n_iter, sweep_len, num_methods]);
for iter = 1:n_iter
    disp(iter/n_iter)
    %% num of train-samples??
    for idx = 1:sweep_len
        noise_lvl = sweep_var(idx);
        %%
        p = randn([3, 1]);
        init_func = @(x) (p(1) + p(2)*x + p(3)*x.^2);
        clean_data = analytical_solution(n_samples_x, a, f, init_func);
        data = clean_data + noise_lvl*randn(size(clean_data));
        train_size = min(ceil(train_test_ratio*size(data, 2))+1, size(data, 2)-1);
        %% train-test split:
        train_data = data(:, 1:train_size);
        test_len = size(data(:, train_size+1:end), 2);
        X1 = train_data(:, 1:end-1);
        X2 = train_data(:, 2:end);
        %% sDMD (fin. diff.)
        A_sdmdfd = vanilla_fd_DMD(X1, X2);
        %% sDMD (higher order)
        A_sdmdho = ho_fd_DMD(X1, X2);
        %% sDMD (param. est.)
        A_sdmdpe = vanilla_pe_DMD(X1, X2);
        %% piDMD
        % piDMD is numerically instable
        [A_pidmd, varargout] = piDMD(X1, X2, "diagonalpinv", 2);
        A_pidmd = A_pidmd(eye(size(train_data, 1)));
        %% basic DMD
        [U, S, V] = svd(X1);
        threshold = optimal_SVHT_coef(min(size(X1, 1)/size(X1, 2), 1), 0)*median(diag(S));
        r = sum(diag(S)>threshold);
        A_dmd = X2*V(:, 1:r)*pinv(S(1:r, 1:r))*U(:, 1:r)';
        %% simulation
        fd_system = md_fd_system(n_samples_x, a, f);
        %% predict
        N = size(data, 2);
        pred_sdmdfd = data;
        pred_sdmdho = data;
        pred_sdmdpe = data;
        pred_pidmd = data;
        pred_dmd = data;
        pred_simulation = data;
        for i = train_size:N
            pred_sdmdfd(:, i) = A_sdmdfd*pred_sdmdfd(:, i-1);
            pred_sdmdho(:, i) = A_sdmdho*pred_sdmdho(:, i-1);
            pred_sdmdpe(:, i) = A_sdmdpe*pred_sdmdpe(:, i-1);
            pred_pidmd(:, i) = A_pidmd*pred_pidmd(:, i-1);
            pred_dmd(:, i) = A_dmd*pred_dmd(:, i-1);
            pred_simulation(:, i) = fd_system*pred_simulation(:, i-1);
        end
        pred_sdmdfd = pred_sdmdfd(train_size+1:end);
        pred_sdmdho = pred_sdmdho(train_size+1:end);
        pred_sdmdpe = pred_sdmdpe(train_size+1:end);
        pred_pidmd = pred_pidmd(train_size+1:end);
        pred_dmd = pred_dmd(train_size+1:end);
        pred_simulation = pred_simulation(train_size+1:end);
        clean_data = clean_data(train_size+1:end);
        %% error
        eps = 1e-16;
        pred_error(iter, idx, 1) = mean(mean(abs(pred_sdmdfd - clean_data)./(abs(clean_data)+eps)));
        pred_error(iter, idx, 2) = mean(mean(abs(pred_sdmdho - clean_data)./(abs(clean_data)+eps)));
        pred_error(iter, idx, 3) = mean(mean(abs(pred_sdmdpe - clean_data)./(abs(clean_data)+eps)));
        pred_error(iter, idx, 4) = mean(mean(abs(pred_pidmd - clean_data)./(abs(clean_data)+eps)));
        pred_error(iter, idx, 5) = mean(mean(abs(pred_dmd - clean_data)./(abs(clean_data)+eps)));
        pred_error(iter, idx, 6) = mean(mean(abs(pred_simulation - clean_data)./(abs(clean_data)+eps)));
    end
end
pred_median = zeros([sweep_len, num_methods]);
pred_min = zeros([sweep_len, num_methods]);
pred_max = zeros([sweep_len, num_methods]);
for idx = 1:sweep_len
    for i = 1:num_methods
        pred_median(idx, i) = median(pred_error(:, idx, i));
        pred_min(idx, i) = min(pred_error(:, idx, i));
        pred_max(idx, i) = max(pred_error(:, idx, i));
    end
end
%%
figure()
plot(sweep_var, pred_median(:, 1), 'Displayname', 'sDMD', 'color', 'b'), hold on;
plot(sweep_var, pred_min(:, 1), 'b', 'linestyle', ':', 'Displayname', '...')
plot(sweep_var, pred_max(:, 1), 'b', 'linestyle', ':', 'Displayname', '...')
plot(sweep_var, pred_median(:, 2), 'Displayname', 'sDMD (ho)', 'color', 'c'), hold on;
plot(sweep_var, pred_min(:, 2), 'c', 'linestyle', ':', 'Displayname', '...')
plot(sweep_var, pred_max(:, 2), 'c', 'linestyle', ':', 'Displayname', '...')
plot(sweep_var, pred_median(:, 3), 'Displayname', 'sDMD (pe)', 'color', 'k'), hold on;
plot(sweep_var, pred_min(:, 3), 'k', 'linestyle', ':', 'Displayname', '...')
plot(sweep_var, pred_max(:, 3), 'k', 'linestyle', ':', 'Displayname', '...')
plot(sweep_var, pred_median(:, 4), 'Displayname', 'piDMD', 'color', 'g');
plot(sweep_var, pred_min(:, 4), 'g', 'linestyle', ':', 'Displayname', '...')
plot(sweep_var, pred_max(:, 4), 'g', 'linestyle', ':', 'Displayname', '...')
plot(sweep_var, pred_median(:, 5), 'Displayname', 'DMD', 'color', 'r');
plot(sweep_var, pred_min(:, 5), 'r', 'linestyle', ':', 'Displayname', '...')
plot(sweep_var, pred_max(:, 5), 'r', 'linestyle', ':', 'Displayname', '...')
plot(sweep_var, pred_median(:, 6), 'Displayname', 'Simulation', 'color', 'y');
plot(sweep_var, pred_min(:, 6), 'r', 'linestyle', ':', 'Displayname', '...')
plot(sweep_var, pred_max(:, 6), 'r', 'linestyle', ':', 'Displayname', '...')
title("prediction error")
ylim([0, 1])
legend()
%%
function system = md_fd_system(N, a, f)
    L = 1; % length
    % dx = L/(N-1); % spatial sampling
    xgrid = linspace(0, L, N);
    dx = xgrid(2); % spatial sampling
    dt = f*(dx^2/(2*a)); % =>  1/2 > (dt*a)/dx^2 (see Colaco, p.180, eq. 5.138b) % same (!) as in analytical_solution.m
    D = full(gallery('tridiag',N,1,-2,1));
    D(1,2) = 2;
    D(N,N-1) = 2;    
    system = dt*a*D/dx^2 + eye(N);       % p. 179, eq. 5.132
end