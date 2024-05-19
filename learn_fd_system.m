close all; clear all; clc; 
%%
rng(1);
%% gen. data
n_samples_x = 21;
k = 401; % thermal conductivity
c = 385; % specific heat capacity
r = 8.96; % mass density    
f = 5.9;            % fummelkonstante space-time-discretization
data = analytical_solution(n_samples_x, k, r, c, f);
%% num of train-samples??
% sDMD is reasonable for very few samples. Otherwise basic DMD seems to
% converge faster and is less complicated
fd_system = md_fd_system(n_samples_x, k, r, c, f);

for train_size = 1:size(data, 2)-1
    %% train-test split:
    train_data = data(:, 1:train_size+1);
    test_data = data(:, train_size+2:end);
    X1 = train_data(:, 1:end-1);
    X2 = train_data(:, 2:end);
    %% sDMD
    A_sdmd = vanilla_fd_DMD(X1, X2);
    %% basic DMD
    [U, S, V] = svd(X1);
    r = min(size(X1));
    A_dmd = X2*V(:, 1:r)*pinv(S(1:r, 1:r))*U(:, 1:r)';
    %%
    [A_pidmd, varargout] = piDMD(X1, X2, "diagonalpinv", 2);
    A_pidmd = A_pidmd(eye(size(train_data, 1)));
    %%
    error_sdmd(train_size) = norm(fd_system - A_sdmd, 'fro');
    error_pidmd(train_size) = norm(fd_system - A_pidmd, 'fro');
    error_dmd(train_size) = norm(fd_system - A_dmd, 'fro');
end
%%
figure()
plot(error_dmd, 'Displayname', 'DMD'), hold on;
plot(error_pidmd, 'Displayname', 'piDMD');
plot(error_sdmd, 'Displayname', 'sDMD');
legend()
%% 
figure()
maxval = max(max([fd_system, A_sdmd]));
minval = min(min([fd_system, A_sdmd]));
subplot(1,7,1)
imagesc(fd_system)
clim([minval, maxval])
title("fin.-diff. system")
subplot(1,7,2)
imagesc(A_sdmd)
clim([minval, maxval])
title("sDMD system")
subplot(1,7,3)
imagesc(A_sdmd-fd_system)
clim([minval, maxval])
title("difference")
subplot(1,7,4)
imagesc(A_dmd)
clim([minval, maxval])
title("sDMD system")
subplot(1,7,5)
imagesc(A_dmd-fd_system)
clim([minval, maxval])
title("difference")
subplot(1,7,6)
imagesc(A_pidmd)
clim([minval, maxval])
title("piDMD system")
subplot(1,7,7)
imagesc(A_pidmd-fd_system)
clim([minval, maxval])
title("difference")
disp(" ")
disp(['"error" (or rather difference) between modelled system and "learned" sDMD system: ', num2str(norm((fd_system-A_sdmd)./(fd_system + 1e-16)))]);
disp(['"error" (or rather difference) between modelled system and "learned" DMD system: ', num2str(norm((fd_system-A_dmd)./(fd_system + 1e-16)))]);
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