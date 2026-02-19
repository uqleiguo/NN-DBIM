% Copyright (c) 2026 Lei Guo
% Licensed under the MIT License.
% See LICENSE file for details.

function [Ess, Ez_tot] = func_MoM(gt_eps, gt_sigma, eps_r_b, sigma_b, verify_flag)


%%% --------------------- Define EM parameters ---------------------

freq = 1000e6;
w = 2 * pi * freq;
eps_o = 8.854187817e-12;
u_o = 4 * pi * 1e-7;
eps_bx = eps_r_b * eps_o - 1i * sigma_b / w;
kb = w * sqrt(u_o * eps_bx);
c = 1 / sqrt(u_o * eps_o);

src_Rx_N = 16;
src_Tx_N = 16;

dx = 2e-3;
dy = 2e-3;
al = 2e-3;
a = al / (sqrt(pi));

x_dash = -250e-3 + dx : dx : 250e-3;
y_dash = -250e-3 + dx : dy : 250e-3;

x_dash = x_dash(63 : 190);
y_dash = y_dash(63 : 190);


Nx = length(x_dash);
Ny = length(y_dash);
total_n = Nx * Ny;

[Axis_x, Axis_y] = ndgrid(x_dash, y_dash);
Axis_x = Axis_x(:);
Axis_y = Axis_y(:);


X = (squeeze(gt_eps) * eps_o ./ eps_bx) - 1i * (squeeze(gt_sigma) ./ (w * eps_bx)) - 1; 

%%% ---------------------------- Find the Tx points ------------------------------

phi_Tx_1 = (0 + pi / 16 : pi / 8 : (2 * pi + pi / 16 - pi / 8));
phi_Tx_1 = circshift(phi_Tx_1, -4);

source_contour_x = -6e-3;
source_contour_y = 5e-3;

Tx_Xradius_1 = 200e-3;
Tx_Yradius_1 = 200e-3;
Probes_Tx = zeros(src_Tx_N, 2);

Ez_inc_MoM = zeros(Nx * Ny, src_Tx_N);

for src_dash = 1 : src_Tx_N

    Probes_Tx(src_dash, 1) = source_contour_x - cos(phi_Tx_1(src_dash)) * Tx_Xradius_1;
    Probes_Tx(src_dash, 2) = source_contour_y - sin(phi_Tx_1(src_dash)) * Tx_Yradius_1;
    Probes_pho = sqrt((Axis_x(:) - Probes_Tx(src_dash, 2)) .^ 2 + (Axis_y(:) - Probes_Tx(src_dash, 1)) .^ 2);

    Ez_inc_MoM(:, src_dash) = -1 * w * u_o * 0.25 * besselh(0, 2, kb .* Probes_pho');

end


%%% ---------------------------- Find the Rx points ------------------------------

phi_Rx_1 = (0 + pi / 16 : pi / 8 : (2 * pi + pi / 16 - pi / 8));
phi_Rx_1 = circshift(phi_Rx_1, -4);

source_contour_x = 3e-3;
source_contour_y = 3e-3;

Rx_Xradius_1 = 100e-3;
Rx_Yradius_1 = 120e-3;
Probes_Rx = zeros(src_Rx_N, 2);

source_X_Rx = zeros(16, 1);
source_Y_Rx = zeros(16, 1);

for src_dash = 1 : src_Rx_N
    Probes_Rx(src_dash, 1) = source_contour_x - cos(phi_Rx_1(src_dash)) * Rx_Xradius_1;
    Probes_Rx(src_dash, 2) = source_contour_y - sin(phi_Rx_1(src_dash)) * Rx_Yradius_1;
    
    dis_source_x = abs(Probes_Rx(src_dash, 1) - x_dash);
    [~, source_X_Rx(src_dash)] = min(dis_source_x);
    dis_source_y = abs(Probes_Rx(src_dash, 2) - y_dash);
    [~, source_Y_Rx(src_dash)] = min(dis_source_y);
end

Probes_Rx(:, 1) = x_dash(source_X_Rx);
Probes_Rx(:, 2) = y_dash(source_Y_Rx);

%%% --------------------- Build the Green function matrix ---------------------

Gezz = zeros(Nx * Ny, Nx * Ny);

for nn = 1 : Nx * Ny
    x = Axis_x(nn);
    y = Axis_y(nn);

    p = sqrt((x - Axis_x(:)).^2 + (y - Axis_y(:)).^2);

    Gezz(nn, :) = (kb ^ 2) .* ((1i * pi .* a) / (2 * kb)) .* besselj(1, kb .* a) .* besselh(0, 2, kb .* p);
    Gezz(nn, nn) = (kb ^ 2) * ((1 / (kb ^ 2)) + (1i * pi * a * besselh(1, 2, kb * a)) / (2 * kb));
end


Gezz_source = zeros(src_Rx_N, total_n);
for m = 1 : src_Rx_N
    x = Probes_Rx(m, 1);
    y = Probes_Rx(m, 2);

    p = sqrt((x - Axis_x).^2 + (y - Axis_y).^2);       

     %%% Green function for Ez field  
    Gezz_source(m, :) = (kb ^ 2) .* ((-1i * pi .* a) / (2 * kb)) .* besselj(1, kb .* a) .* besselh(0, 2, kb .* p);
    Gezz_source(m, p == 0) = (kb ^ 2) * ((1 / (kb ^ 2)) + (-1i * pi * a * besselh(1, 2, kb * a)) / (2 * kb));
end

I = eye(total_n);
GG_Ez = I + Gezz .* repmat(X(:).', total_n, 1);

Ez_tot = zeros(total_n, src_Tx_N);
Ez_scat = zeros(total_n, src_Tx_N);

for src_num = 1 : src_Tx_N
    tic
    Ez_tot(:, src_num) = GG_Ez \ Ez_inc_MoM(:, src_num);
    toc
    Ez_scat(:, src_num) = Ez_tot(:, src_num) - Ez_inc_MoM(:, src_num);
    fprintf('The %ith source is calculated...\n', src_num);
end

wr_Ez = repmat(X(:), 1, src_Tx_N) .* Ez_tot;
Ess = Gezz_source * wr_Ez;



%%% Calculate the total field and verify the calculated total field

if verify_flag
    for ii = 1 : src_Tx_N
        Ez_tot_mom_mat = reshape(Ez_tot(:, ii), Nx, Ny);
        
        [Ez_gy, Ez_gx] = gradient(Ez_tot_mom_mat, dx, dy);
        [Ez_gxy, Ez_gxx] = gradient(Ez_gx, dx, dy);
        [Ez_gyy, Ez_gyx] = gradient(Ez_gy, dx, dy);
        
        curl_curl_Ez = -1 .* Ez_gxx - Ez_gyy;
        gama_k = curl_curl_Ez ./ Ez_tot_mom_mat;
        
        eps_gama_k(:, :, ii) = real(gama_k) ./ (w ^ 2 * u_o);
        sigma_gama_k(:, :, ii) = -1 * ((imag(gama_k)) ./ (w * u_o));
    end

    eps_gama_k = sum(eps_gama_k, 3) ./ src_Tx_N;
    sigma_gama_k = sum(sigma_gama_k, 3) ./ src_Tx_N;
    
    figure; imagesc(Axis_x(:), Axis_y(:), eps_gama_k ./ eps_o); axis image; clim([0 70])
    figure; imagesc(Axis_x(:), Axis_y(:), sigma_gama_k); axis image; clim([0 2.4])
end





