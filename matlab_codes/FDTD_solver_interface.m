clc
clear

eps_heads = importdata('../data/eps_rand_heads.mat');
sigma_heads = importdata('../data/sigma_rand_heads.mat');

case_num = size(eps_heads, 3);
S_Ez_mat = zeros(16, 16, case_num);


for kk = 1 : case_num
    eps_gama = eps_heads(:, :, kk);
    sigma_gama = sigma_heads(:, :, kk);
    
    S_Ez_mat(:, :, kk) = func_FDTD(eps_gama, sigma_gama, 0);
    fprintf('The %ith case is computed...\n', kk);
    fprintf('----------------------------\n');
end



