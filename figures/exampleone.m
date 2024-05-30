close all; clear all; clc; 
addpath '/home/bonenberger/matlab/DMD/ASIM'
%%
rng(1);
%% gen. data
n_samples_x = 21;
k = 401;  % W/(m*K) thermal conductivity (copper)
c = 385;  % J/(kg*K) specific thermal capacity (copper)
r = 8.96; % kg/(m^3), mass density (copper)    
a = k/(r*c); % diffusivity constant
f = 0.5;  % control sampling (Courant condition)
%%
init_func = @(x) x;
init_func = @(x) (x.^2);
init_func = @(x) (x.^3);
% plot(ana_data(:, 1))
% asdas
n_iter = 25;
noise_lvl = 0.0;
num_methods = 5;
sweep_var = 0.25:0.25:4;
sweep_len = length(sweep_var);
pred_error = zeros([n_iter, sweep_len, num_methods]);
sys_error = zeros([n_iter, sweep_len, num_methods]);
for iter = 1:n_iter
    disp(iter/n_iter)
    %% num of train-samples??
    for idx = 1:sweep_len
        f = sweep_var(idx);
        ana_data = analytical_solution(n_samples_x, a, f, init_func);
        fd_system = md_fd_system(n_samples_x, a, f);
        data = ana_data + noise_lvl*randn(size(ana_data));
        train_size = ceil(0.1*size(data, 2))+1;
        %% train-test split:
        train_data = data(:, 1:train_size);
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
        %% predict
        N = size(data, 2);
        pred_sdmdfd = data;
        pred_sdmdho = data;
        pred_sdmdpe = data;
        pred_pidmd = data;
        pred_dmd = data;
        % pred_data4 = data;
        for i = train_size:N
            pred_sdmdfd(:, i) = A_sdmdfd*pred_sdmdfd(:, i-1);
            pred_sdmdho(:, i) = A_sdmdho*pred_sdmdho(:, i-1);
            pred_sdmdpe(:, i) = A_sdmdpe*pred_sdmdpe(:, i-1);
            pred_pidmd(:, i) = A_pidmd*pred_pidmd(:, i-1);
            pred_dmd(:, i) = A_dmd*pred_dmd(:, i-1);
        end
        %% error
        pred_error(iter, idx, 1) = norm(pred_sdmdfd - ana_data, 'fro');
        pred_error(iter, idx, 2) = norm(pred_sdmdho - ana_data, 'fro');
        pred_error(iter, idx, 3) = norm(pred_sdmdpe - ana_data, 'fro');
        pred_error(iter, idx, 4) = norm(pred_pidmd - ana_data, 'fro');
        pred_error(iter, idx, 5) = norm(pred_dmd - ana_data, 'fro');
        sys_error(iter, idx, 1) = norm(fd_system - A_sdmdfd, 'fro');
        sys_error(iter, idx, 2) = norm(fd_system - A_sdmdho, 'fro');
        sys_error(iter, idx, 3) = norm(fd_system - A_sdmdpe, 'fro');
        sys_error(iter, idx, 4) = norm(fd_system - A_pidmd, 'fro');
        sys_error(iter, idx, 5) = norm(fd_system - A_dmd, 'fro');
    end
end
% pred_error = log(1+pred_error)/n_iter;
% sys_error = log(1+sys_error)/n_iter;

pred_var = zeros([sweep_len, num_methods]);
pred_mean = zeros([sweep_len, num_methods]);
pred_min = zeros([sweep_len, num_methods]);
pred_max = zeros([sweep_len, num_methods]);
sys_var = zeros([sweep_len, num_methods]);
sys_mean = zeros([sweep_len, num_methods]);
sys_min = zeros([sweep_len, num_methods]);
sys_max = zeros([sweep_len, num_methods]);
for idx = 1:sweep_len
    for i = 1:num_methods
        pred_var(idx, i) = var(pred_error(:, idx, i));
        pred_mean(idx, i) = mean(pred_error(:, idx, i));
        pred_min(idx, i) = min(pred_error(:, idx, i));
        pred_max(idx, i) = max(pred_error(:, idx, i));
        sys_var(idx, i) = var(sys_error(:, idx, i));
        sys_mean(idx, i) = mean(sys_error(:, idx, i));
        sys_min(idx, i) = min(sys_error(:, idx, i));
        sys_max(idx, i) = max(sys_error(:, idx, i));
    end
end
%%
% pred_var = log(1+pred_var);
% pred_mean = log(1+pred_mean);
% pred_min = log(1+pred_min);
% pred_max = log(1+pred_max);
%%
figure()
plot(sys_mean(:, 1), 'Displayname', 'sDMD (fd)', 'color', 'b'), hold on;
plot(sys_min(:, 1), 'b', 'linestyle', ':', 'Displayname', '...')
plot(sys_max(:, 1), 'b', 'linestyle', ':', 'Displayname', '...')
plot(sys_mean(:, 2), 'Displayname', 'sDMD (ho)', 'color', 'c'), hold on;
plot(sys_min(:, 2), 'c', 'linestyle', ':', 'Displayname', '...')
plot(sys_max(:, 2), 'c', 'linestyle', ':', 'Displayname', '...')
plot(sys_mean(:, 3), 'Displayname', 'sDMD (pe)', 'color', 'k'), hold on;
plot(sys_min(:, 3), 'k', 'linestyle', ':', 'Displayname', '...')
plot(sys_max(:, 3), 'k', 'linestyle', ':', 'Displayname', '...')
plot(sys_mean(:, 4), 'Displayname', 'piDMD', 'color', 'g');
plot(sys_min(:, 4), 'g', 'linestyle', ':', 'Displayname', '...')
plot(sys_max(:, 4), 'g', 'linestyle', ':', 'Displayname', '...')
plot(sys_mean(:, 5), 'Displayname', 'DMD', 'color', 'r');
plot(sys_min(:, 5), 'r', 'linestyle', ':', 'Displayname', '...')
plot(sys_max(:, 5), 'r', 'linestyle', ':', 'Displayname', '...')
title("FD-system approximation")
legend()
ylim([0,10])
figure()
plot(pred_mean(:, 1), 'Displayname', 'sDMD', 'color', 'b'), hold on;
plot(pred_min(:, 1), 'b', 'linestyle', ':', 'Displayname', '...')
plot(pred_max(:, 1), 'b', 'linestyle', ':', 'Displayname', '...')
plot(pred_mean(:, 2), 'Displayname', 'sDMD (ho)', 'color', 'c'), hold on;
plot(pred_min(:, 2), 'c', 'linestyle', ':', 'Displayname', '...')
plot(pred_max(:, 2), 'c', 'linestyle', ':', 'Displayname', '...')
plot(pred_mean(:, 3), 'Displayname', 'sDMD (pe)', 'color', 'k'), hold on;
plot(pred_min(:, 3), 'k', 'linestyle', ':', 'Displayname', '...')
plot(pred_max(:, 3), 'k', 'linestyle', ':', 'Displayname', '...')
plot(pred_mean(:, 4), 'Displayname', 'piDMD', 'color', 'g');
plot(pred_min(:, 4), 'g', 'linestyle', ':', 'Displayname', '...')
plot(pred_max(:, 4), 'g', 'linestyle', ':', 'Displayname', '...')
plot(pred_mean(:, 5), 'Displayname', 'DMD', 'color', 'r');
plot(pred_min(:, 5), 'r', 'linestyle', ':', 'Displayname', '...')
plot(pred_max(:, 5), 'r', 'linestyle', ':', 'Displayname', '...')
title("prediction error")
ylim([0, 5])
legend()
path = "/home/bonenberger/Dokumente/eigenePaper/ASIM/data/";
save_file(ana_data, strcat("raw_", num2str(noise_lvl), ".csv"), path)
save_file(data, strcat("noisy_", num2str(noise_lvl), ".csv"), path)
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
function save_file(X, filename, path)
    X(isinf(X)) = 0;
    X_ = X(:, 1:end);
    X_ = X_(1:end, :);
    [xx_, yy_] = meshgrid((0:size(X_, 2)-1), (0:size(X_, 1)-1));
    x = round([xx_(:), yy_(:), X_(:)], 8);
    
    fid = fopen(strcat(path, filename),'w');   
    fprintf(fid,' ');
    fclose(fid);
    fid = fopen(strcat(path, filename),'a'); 
    for i = 1:size(x, 1)
        if x(i, 2) == 0
            fprintf(fid, '\n');
        end
        fprintf(fid,'%i ',x(i, 1:2));
        fprintf(fid,'%.4f ',x(i, 3));
        fprintf(fid, '\n');
    end
    fclose(fid);
end
