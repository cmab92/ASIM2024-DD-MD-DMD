close all; clear all; clc; 
%%
rng(1);
%% gen. data
n_samples_x = 21;
k = 401; % thermal conductivity
c = 385; % specific heat capacity
r = 8.96; % mass density    
f = 1/2;            % fummelkonstante space-time-discretization
data = analytical_solution(n_samples_x, k, r, c, f);
%% num of train-samples??
% sDMD is reasonable for very few samples. Otherwise basic DMD seems to
% converge faster and is less complicated
train_size = floor(0.9*size(data, 2));
% train_size = 2;
%% train-test split:
train_data = data(:, 1:train_size);
test_data = data(:, train_size+1:end);
X1 = train_data(:, 1:end-1);
X2 = train_data(:, 2:end);
disp(['spatial samples: ', int2str(size(data, 1))]);
disp(['train-data samples: ', int2str(size(train_data, 2))]);
disp(['test-data samples: ', int2str(size(test_data, 2))]);
disp(" ")
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
%%
disp(['sDMD overall error: ', num2str(norm((data-pred_data1)./data))]);
disp(['piDMD overall error: ', num2str(norm((data-pred_data2)./data))]);
disp(['basic DMD overall error: ', num2str(norm((data-pred_data3)./data))]);
%% BBBB
figure()
maxval = max(max(data));
minval = min(min(data));
subplot(2,4,1)
imagesc(data)
title("orig")
clim([minval, maxval])
subplot(2,4,2)
imagesc(pred_data1)
title("sDMD")
clim([minval, maxval])
subplot(2,4,3)
imagesc(pred_data2)
title("piDMD (tridiag)")
clim([minval, maxval])
subplot(2,4,4)
imagesc(pred_data3)
title("basic DMD")
clim([minval, maxval])
%%
err1 = pred_error_visu(data, pred_data1);
err2 = pred_error_visu(data, pred_data2);
err3 = pred_error_visu(data, pred_data3);
maxval = max(max([err1, err2, err3]));
minval = min(min([err1, err2, err3]));
subplot(2,4,5)
imagesc(pred_error_visu(data, data))
title("orig")
subplot(2,4,6)
imagesc(err1)
clim([minval, maxval])
title("sDMD")
subplot(2,4,7)
imagesc(err2)
clim([minval, maxval])
title("piDMD (tridiag)")
subplot(2,4,8)
imagesc(err3)
clim([minval, maxval])
title("basic DMD")
%%
figure()
fd_system = md_fd_system(n_samples_x, k, r, c, f);
maxval = max(max([fd_system, A_sdmd]));
minval = min(min([fd_system, A_sdmd]));
subplot(1,3,1)
imagesc(fd_system)
clim([minval, maxval])
title("fin.-diff. system")
subplot(1,3,2)
imagesc(A_sdmd)
clim([minval, maxval])
title("sDMD system")
subplot(1,3,3)
imagesc(A_sdmd-fd_system)
clim([minval, maxval])
title("difference")
disp(" ")
disp(['"error" (or rather difference) between modelled system and "learned" system: ', num2str(norm((fd_system-A_sdmd)./(fd_system + 1e-16)))]);
%%
function Z = pred_error_visu(true, pred)
    Z = log(abs((true-pred)./(true+1e-16)));
end
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