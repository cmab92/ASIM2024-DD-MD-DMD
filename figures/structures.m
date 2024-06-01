clc; close all; clear all;
rng(1)
path = "/home/bonenberger/Dokumente/eigenePaper/ASIM/data/structures/";

%% DMD
N = 11;
A = rand(N);
% A = A - mean(mean(A));
subplot(5,1,1)
imagesc(A)
filename = "dmd.dat";
save_file(A, filename, path)
%% piDMD
N = 11;
A = zeros(N);
for i = 1:N
    A(i,i) = rand(1);
    if i>1
        A(i,i-1) = 0.5*rand(1)+0.5;
    end
    if i<N
        A(i,i+1) = 0.5*rand(1)+0.5;
    end
end
% A = A - mean(mean(A));
subplot(5,1,2)
imagesc(A)
filename = "pidmd.dat";
save_file(A, filename, path)
%% FDsDMD
I = eye(N);
S(:, :, 1) = I;
S(:, :, 2) = toeplitz(I(:, 2), 0*I(:, 2));
S(:, :, 3) = toeplitz(0*I(:, 2), I(:, 2));
S(1, 2, 4) = 1;
S(N, N-1, 5) = 1;
A = zeros(N);
r = rand([5,1]);
for i = 1:5
    A = A + (0.5+0.5*r(i))*S(:,:,i);
end
% A = A - mean(mean(A));
subplot(5,1,3)
imagesc(A)
filename = "fddmd.dat";
save_file(A, filename, path)
for i = 1:5
    filename = strcat("struct_fddmd", int2str(i) , ".dat");
    save_file(S(:,:,i), filename, path)
end
%% HOsDMD
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
A = zeros(N);
for i = 1:size(S, 3)
    A = A + (0.5+0.5*rand(1))*S(:,:,i);
end
% A = A - mean(mean(A));
subplot(5,1,4)
imagesc(A)
filename = "hodmd.dat";
save_file(A, filename, path)

function save_file(X, filename, path)
    X(isinf(X)) = 0;
    X = [X zeros([size(X, 1), 1])];
    X = [X; zeros([1, size(X, 2)])];
    [xx_, yy_] = meshgrid((0:size(X, 2)-1), (0:size(X, 1)-1));
    x = round([xx_(:), yy_(:), X(:)], 8);
    % save(strcat(path, filename), 'x', '-ascii');
    
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