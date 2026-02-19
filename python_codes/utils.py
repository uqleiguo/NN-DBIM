"""
Copyright (c) 2026 Lei Guo

This file is part of the NN-DBIM project.
Licensed under the MIT License.
See LICENSE file in the project root for full license information.
"""

import numpy as np
import torch


if torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")


### Calculate the final output size from convolution layers
def conv_output_size(in_size, kernels, paddings, strides, dilations, max_pool, max_pool_kernels,
                     max_pool_stride=2, max_pool_padding=0, max_pool_dilation=1):
    for kernel_size, padding_size, stride_size, dilation_size in zip(kernels, paddings, strides, dilations):
        in_size = (in_size + 2 * padding_size - dilation_size * (kernel_size - 1) - 1) / stride_size + 1
        if max_pool:
            in_size = (in_size + 2 * max_pool_padding - max_pool_dilation * (max_pool_kernels - 1) - 1) / max_pool_stride + 1

    return in_size


### Calculate the final output size from deconvolution layers
def deconv_output_size(in_size, kernels, paddings, strides, dilations, up_sample):
    for kernel_size, padding_size, stride_size, dilation_size in zip(kernels, paddings, strides, dilations):
        if up_sample:
            in_size = in_size * 2
        in_size = (in_size + 2 * padding_size - dilation_size * (kernel_size - 1) - 1) / stride_size + 1

    return in_size


### Equivalent to the matlab function sub2ind
def sub2ind(array_shape, rows, cols):
    ind = rows*array_shape[1] + cols
    ind[ind < 0] = -1
    ind[ind >= array_shape[0]*array_shape[1]] = -1
    return ind


xx = torch.arange(-248e-3, 252e-3, 2e-3)
yy = torch.arange(-248e-3, 252e-3, 2e-3)
xx = xx[62 : 190]
yy = yy[62 : 190]

eps_o = 8.854187817e-12
freq = 1000e6
w = 2 * np.pi * freq
eps_r_b = 30
sigma_b = 0.5
eps_b = eps_r_b * eps_o - 1j * sigma_b / w
u_o = 4 * np.pi * 1e-7
gama_k_inc = eps_r_b * eps_o * (w ** 2 * u_o) + 1j * (-1 * sigma_b * w * u_o)


def curl_curl_operator_2d_ep(Az):

    Az_cmp = Az[:, 0, :, :] + 1j * Az[:, 1, :, :]

    batch_size = Az.shape[0]
    ep = torch.zeros((batch_size, 2, Az.shape[2], Az.shape[3])).to(device, torch.float)

    for nn in range(batch_size):
        Az_g = torch.gradient(Az_cmp[nn, :, :], spacing=2e-3, edge_order=2)
        Az_gx = Az_g[0]
        Az_gy = Az_g[1]

        Az_gxx = torch.gradient(Az_gx, spacing=2e-3, edge_order=2)[0]
        Az_gyy = torch.gradient(Az_gy, spacing=2e-3, edge_order=2)[1]

        curl_curl_Az = -1 * Az_gxx - Az_gyy

        gama = curl_curl_Az / Az_cmp[nn, :, :]

        X_gama = (torch.real(gama) / (w ** 2 * u_o)) / eps_b - 1j * (-1 * (torch.imag(gama) / (w * u_o))) / (w * eps_b) - 1

        ep[nn, 0, :, :] = torch.real((X_gama + 1) * eps_b) / eps_o
        ep[nn, 1, :, :] = -1 * np.imag((X_gama + 1) * eps_b) * w

    return ep


def curl_curl_operator_2d_v2(Az):

    Az_cmp = Az[:, 0, :, :] + 1j * Az[:, 1, :, :]

    batch_size = Az.shape[0]
    curl_curl_x = torch.zeros((batch_size, 2, Az.shape[2], Az.shape[3])).to(device, torch.float)

    for nn in range(batch_size):
        Az_g = torch.gradient(Az_cmp[nn, :, :], spacing=2e-3, edge_order=2)
        Az_gx = Az_g[0]
        Az_gy = Az_g[1]

        Az_gxx = torch.gradient(Az_gx, spacing=2e-3, edge_order=2)[0]
        Az_gyy = torch.gradient(Az_gy, spacing=2e-3, edge_order=2)[1]

        curl_curl_Az = -1 * Az_gxx - Az_gyy

        curl_curl_x[nn, 0, :, :] = torch.real(curl_curl_Az)
        curl_curl_x[nn, 1, :, :] = torch.imag(curl_curl_Az)

    return curl_curl_x


def tan_sigmoid(x):
    y = 8 / (1 + torch.exp(-2 * x)) - 4

    return y


def image_normalization(A, new_min, new_max):

    ### A is the non-normalized image with the size of [K, 2, M, N]
    ### K is the batch size, 2 contains the real (or permittivity) and imaginary (or conductivity) parts of A, M and N are the size of A

    batch_size = A.shape[0]
    A_norm = torch.zeros((batch_size, 2, A.shape[2], A.shape[3])).to(device, torch.float)

    for nn in range(batch_size):
        A_norm[nn, 0, :, :] = (A[nn, 0, :, :] - torch.amin(A[nn, 0, :, :])) * \
                              ((new_max - new_min) / (torch.amax(A[nn, 0, :, :]) - torch.amin(A[nn, 0, :, :]))) + new_min
        A_norm[nn, 1, :, :] = (A[nn, 1, :, :] - torch.amin(A[nn, 1, :, :])) * \
                              ((new_max - new_min) / (torch.amax(A[nn, 1, :, :]) - torch.amin(A[nn, 1, :, :]))) + new_min

    return A_norm


def es_normalization(A, new_min, new_max, maxNorm_coeff, method):

    ### A is the non-normalized scattered signal with the size of [K, L]
    ### new_min and new_max is used in "Min_Max" normalization method
    ### maxNorm_coeff is used in maxNorm method
    ### method is the normalization method, it has to be "Min_Max", "Mean_Std", or 'maxNorm'

    A = A.view(A.shape[0], 2, int(A.shape[1] / 2))
    A_cmp = A[:, 0, :] + 1j * A[:, 1, :]

    A_cmp_norm = torch.zeros((A_cmp.shape[0], A_cmp.shape[1]), dtype=torch.complex64)
    for kk in range(A_cmp.shape[0]):
        if method == "Min_Max":
            amp = torch.abs(A_cmp[kk, :])
            phase = torch.angle(A_cmp[kk, :])

            amp_norm = (amp - torch.amin(amp)) * ((new_max - new_min) / (torch.amax(amp) - torch.amin(amp))) + new_min

            A_cmp_norm[kk, :] = amp_norm * torch.exp(1j * phase)
        elif method == "Mean_Std":
            real_data = torch.real(A_cmp[kk, :])
            imag_data = torch.imag(A_cmp[kk, :])

            real_norm = (real_data - torch.mean(real_data)) / torch.std(real_data)
            imag_norm = (imag_data - torch.mean(imag_data)) / torch.std(imag_data)

            A_cmp_norm[kk, :] = real_norm + 1j * imag_norm
        elif method == 'maxNorm':
            A_cmp_norm[kk, :] = A_cmp[kk, :] / maxNorm_coeff
        else:
            raise ValueError("The normalization method has to be 'Min_Max', 'Mean_Std', or 'maxNorm'")

    A_norm = torch.zeros((A_cmp_norm.shape[0], 2, A_cmp_norm.shape[1], 1)).to(device, torch.float)
    A_norm[:, 0, :, 0] = torch.real(A_cmp_norm)
    A_norm[:, 1, :, 0] = torch.imag(A_cmp_norm)
    A_norm = A_norm.view(A_norm.shape[0], 2 * A_norm.shape[2])

    return A_norm


def extract_ess_data(e_tot_field, e_inc_field):

    batch_size = e_tot_field.shape[0]

    src_Rx_N = 16
    phi_Rx_1 = np.arange(np.pi / 16, 2 * np.pi + np.pi / 16 - np.pi / 8, np.pi / 8)
    phi_Rx_1 = np.roll(phi_Rx_1, -4)

    source_contour_x = 3e-3
    source_contour_y = 3e-3

    Rx_Xradius_1 = 100e-3
    Rx_Yradius_1 = 120e-3

    es_mat = torch.zeros((batch_size, 2, 16, 1)).to(device, torch.float)

    for src_dash in range(src_Rx_N):
        Probes_Rx_x = source_contour_x - np.cos(phi_Rx_1[src_dash]) * Rx_Xradius_1
        Probes_Rx_y = source_contour_y - np.sin(phi_Rx_1[src_dash]) * Rx_Yradius_1

        dis_source_x = abs(Probes_Rx_x - xx)
        dis_source_y = abs(Probes_Rx_y - yy)

        source_X_Rx = np.argmin(dis_source_x)
        source_Y_Rx = np.argmin(dis_source_y)

        es_mat[:, :, src_dash, 0] = e_tot_field[:, :, source_Y_Rx, source_X_Rx] - e_inc_field[:, :, source_Y_Rx, source_X_Rx]

    return es_mat


def extract_ess_data_rx(e_tot_field, e_inc_field, Rx_N):

    batch_size = e_tot_field.shape[0]

    src_Rx_N = 16
    phi_Rx_1 = np.arange(np.pi / Rx_N, 2 * np.pi + np.pi / Rx_N - np.pi / (Rx_N / 2), np.pi / (Rx_N / 2))
    phi_Rx_1 = np.roll(phi_Rx_1, -4)

    source_contour_x = 3e-3
    source_contour_y = 3e-3

    Rx_Xradius_1 = 100e-3
    Rx_Yradius_1 = 120e-3

    es_mat = torch.zeros((batch_size, 2, Rx_N, 1)).to(device, torch.float)

    for src_dash in range(src_Rx_N):
        Probes_Rx_x = source_contour_x - np.cos(phi_Rx_1[src_dash]) * Rx_Xradius_1
        Probes_Rx_y = source_contour_y - np.sin(phi_Rx_1[src_dash]) * Rx_Yradius_1

        dis_source_x = abs(Probes_Rx_x - xx)
        dis_source_y = abs(Probes_Rx_y - yy)

        source_X_Rx = np.argmin(dis_source_x)
        source_Y_Rx = np.argmin(dis_source_y)

        es_mat[:, :, src_dash, 0] = e_tot_field[:, :, source_Y_Rx, source_X_Rx] - e_inc_field[:, :, source_Y_Rx, source_X_Rx]

    return es_mat


def es_from_gezz(gezz, e_tot, ep):

    es_mom = torch.zeros((e_tot.shape[0], 2, e_tot.shape[2], e_tot.shape[3])).to(device, torch.float)

    x = (ep[:, 0, :, :] * eps_o / eps_b) - 1j * (ep[:, 1, :, :] / (w * eps_b)) - 1
    e_tot_cmp = e_tot[:, 0, :, :] + 1j * e_tot[:, 1, :, :]
    gezz_cmp = (gezz[:, 0, :, :] + 1j * gezz[:, 1, :, :]).squeeze().to(device, torch.complex64)

    W_cmp = e_tot_cmp * x
    W_cmp = W_cmp.view(W_cmp.shape[0], W_cmp.shape[1] * W_cmp.shape[2])

    for mm in range(W_cmp.shape[0]):
        es = gezz_cmp @ W_cmp[mm, :]
        es = es.view(e_tot.shape[2], e_tot.shape[3])

        es_mom[mm, 0, :, :] = torch.real(es)
        es_mom[mm, 1, :, :] = torch.imag(es)

    return es_mom


def es_from_gezz_source(gezz_source, e_tot, ep):

    es_mom = torch.zeros((e_tot.shape[0], 2, 16)).to(device, torch.float)

    x = (ep[:, 0, :, :] * eps_o / eps_b) - 1j * (ep[:, 1, :, :] / (w * eps_b)) - 1
    e_tot_cmp = e_tot[:, 0, :, :] + 1j * e_tot[:, 1, :, :]
    gezz_source_cmp = (gezz_source[:, 0, :, :] + 1j * gezz_source[:, 1, :, :]).squeeze().to(device, torch.complex64)

    W_cmp = e_tot_cmp * x
    W_cmp = W_cmp.view(W_cmp.shape[0], W_cmp.shape[1] * W_cmp.shape[2])

    for mm in range(W_cmp.shape[0]):
        es = gezz_source_cmp @ W_cmp[mm, :]

        es_mom[mm, 0, :] = torch.real(es)
        es_mom[mm, 1, :] = torch.imag(es)

    return es_mom


def total_variation(A):

    batch_size = A.shape[0]

    A_TV = torch.zeros((batch_size, A.shape[1], A.shape[2])).to(device, torch.float)

    for nn in range(batch_size):
        A_g = torch.gradient(A[nn, :, :], spacing=1, edge_order=1)
        A_gx = A_g[0]
        A_gy = A_g[1]

        A_TV[nn, :, :] = torch.real(A_gx * torch.conj(A_gx) + A_gy * torch.conj(A_gy))

    return A_TV

