% Copyright (c) 2026 Lei Guo
% email: l.guo3@uq.edu.au
% Institution: The University of Queensland

% Licensed under the MIT License.
% See LICENSE file for details.

clc
clear

eps_heads = importdata('../data/eps_rand_heads.mat');
sigma_heads = importdata('../data/sigma_rand_heads.mat');

eps_b = 30;
sigma_b = 0.5;

case_num = size(eps_heads, 3);

Ez_tot = zeros(128, 128, case_num);
S_Ez_mat = zeros(16, 16, case_num);


for kk = 1 : case_num
    eps_gama = eps_heads(:, :, kk);
    sigma_gama = sigma_heads(:, :, kk);

    [Ess, Ez_tot] = func_MoM(eps_gama, sigma_gama, eps_b, sigma_b, 0);
    fprintf('The %ith case is computed...\n', eps_heads);
end
