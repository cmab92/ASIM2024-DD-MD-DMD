close all; clear all; clc; 
%%
rng(1);
%% gen. data
n_samples_x = 21;
k = 401; % thermal conductivity
c = 385; % specific heat capacity
r = 8.96; % mass density    
f = 0.5;            % fummelkonstante space-time-discretization
%%
ana_data = analytical_solution(n_samples_x, k, r, c, f);
n_iter = 100;
noise_lvl = 0.01;
max_train_size = size(ana_data, 2) - 2;
pred_error = zeros([max_train_size, 3]);
sys_error = zeros([max_train_size, 3]);
for iter = 1:n_iter
    disp(iter/n_iter)
    data = ana_data + noise_lvl*randn(size(ana_data));
    fd_system = md_fd_system(n_samples_x, k, r, c, f);
    %% num of train-samples??
    for idx = 1:max_train_size
        train_size = idx+1;
        %% train-test split:
        train_data = data(:, 1:train_size);
        % test_size = size(data, 2) - train_size;
        X1 = train_data(:, 1:end-1);
        X2 = train_data(:, 2:end);
        %% sDMD
        A_sdmd = vanilla_fd_DMD(X1, X2);
        % more degrees of freedom (ho_fd_DMD) allow for better convergence, yet
        % this deteriorates convergence for very few samples
        % A_sdmd = ho_fd_DMD(X1, X2);
        %% piDMD
        % piDMD is numerically instable
        [A_pidmd, varargout] = piDMD(X1, X2, "diagonalpinv", 3);
        A_pidmd = A_pidmd(eye(size(train_data, 1)));
        %% basic DMD
        [U, S, V] = svd(X1);
        r = min(size(X1));
        A_dmd = X2*V(:, 1:r)*pinv(S(1:r, 1:r))*U(:, 1:r)';
        %% predict
        N = size(data, 2);
        pred_data1 = data;
        pred_data2 = data;
        pred_data3 = data;
        for i = train_size+1:N
            pred_data1(:, i) = A_sdmd*pred_data1(:, i-1);
            pred_data2(:, i) = A_pidmd*pred_data2(:, i-1);
            pred_data3(:, i) = A_dmd*pred_data3(:, i-1);
        end
        %% error
        pred_error(idx, 1) = pred_error(idx, 1) + norm(pred_data1 - data, 'fro');
        pred_error(idx, 2) = pred_error(idx, 2) + norm(pred_data2 - data, 'fro');
        pred_error(idx, 3) = pred_error(idx, 3) + norm(pred_data3 - data, 'fro');
        sys_error(idx, 1) = sys_error(idx, 1) + norm(fd_system - A_sdmd, 'fro');
        sys_error(idx, 2) = sys_error(idx, 2) + norm(fd_system - A_pidmd, 'fro');
        sys_error(idx, 3) = sys_error(idx, 3) + norm(fd_system - A_dmd, 'fro');
    end
end
% pred_error = log(1+pred_error)/n_iter;
% sys_error = log(1+sys_error)/n_iter;
pred_error = pred_error/n_iter;
sys_error = sys_error/n_iter;
%%
figure()
plot(sys_error(:, 1), 'Displayname', 'sDMD'), hold on;
plot(sys_error(:, 2), 'Displayname', 'piDMD');
plot(sys_error(:, 3), 'Displayname', 'DMD');
title("FD-system approximation")
legend()
ylim([0,10])
figure()
plot(pred_error(:, 1), 'Displayname', 'sDMD'), hold on;
plot(pred_error(:, 2), 'Displayname', 'piDMD');
plot(pred_error(:, 3), 'Displayname', 'DMD');
title("prediction error")
ylim([0, 20])
legend()
%%
function system = md_fd_system(N, k, rho, c, f)
    L = 1; % length
    dx = L/(N-1); % spatial sampling
    a = k/(rho*c);
    dt = f*(dx^2/(2*a)); % => dt < dx^2/(2*a) % same (!) as in analytical_solution.m
    D = full(gallery('tridiag',N,1,-2,1));
    D(1,2) = 2;
    D(N,N-1) = 2;    
    system = dt*(a/dx * D)/dx + eye(N);
end