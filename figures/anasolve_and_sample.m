close all; clear all; clc; 
addpath '/home/bonenberger/matlab/DMD/ASIM'
%%
rng(1);
%% gen. data
n_samples_x = 11;
k = 401;  % W/(m*K) thermal conductivity (copper)
c = 385;  % J/(kg*K) specific thermal capacity (copper)
r = 8.96; % kg/(m^3), mass density (copper)    
a = k/(r*c); % diffusivity constant
%%
f = 1;  % control sampling (stability condition)
noise_lvl = 0.005;
p = randn([3, 1]);
init_func = @(x) (p(1) + p(2)*x + p(3)*x.^2);
clean_data = analytical_solution(n_samples_x, a, f, init_func);
sampled_data = clean_data + noise_lvl*randn(size(clean_data));
%%
n_samples_x = 31;
init_func = @(x) (p(1) + p(2)*x + p(3)*x.^2);
clean_data = analytical_solution(n_samples_x, a, f, init_func);


path = "/home/bonenberger/Dokumente/eigenePaper/ASIM/data/ttr_sweep/";
save_file(sampled_data, strcat("noisy_", num2str(noise_lvl), ".csv"), path)
save_file(clean_data, strcat("raw_", num2str(noise_lvl), ".csv"), path)
subplot(1,2,1)
imagesc(clean_data)
subplot(1,2,2)
imagesc(sampled_data)
%%
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
