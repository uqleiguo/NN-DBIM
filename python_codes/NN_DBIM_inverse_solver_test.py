"""
Copyright (c) 2026 Lei Guo

This file is part of the NN-DBIM project.
Licensed under the MIT License.
See LICENSE file in the project root for full license information.
"""

import numpy as np
import matplotlib.pyplot as plt
import torch
import utils
import Unet_forward_model_F
import NN_DBIM_inverse_solver_model


if torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")


def evaluate(inv_model, for_model, test_dataset, Ez_inc_field, plot_flag):
    inv_model.eval()

    for jj in range(len(for_model)):
        for_model[jj].eval()
        for params in for_model[jj].parameters():
            params.requires_grad = False

    Ez_inc_field = Ez_inc_field.expand(1, 2, 128, 128, 16)

    head_ep_recon = np.zeros((len(test_dataset), 2, 128, 128))
    stroke_ep_recon = np.zeros((len(test_dataset), 2, 128, 128))

    head_ep_gt = np.zeros((len(test_dataset), 2, 128, 128))

    with torch.no_grad():
        for ii, data in enumerate(test_dataset):
            src_Es = data[0].to(device, torch.float)
            trg_eps = data[1].to(device, torch.float)
            trg_stroke_eps = data[2].to(device, torch.float)

            head_ep_gt[ii, :, :, :] = trg_eps[0, :, :, :].cpu().numpy()

            X_init = torch.zeros((src_Es.shape[0], 2, 128, 128)).to(device, torch.float)

            X_recon, stroke_x, es_inv = inv_model(for_model, src_Es, Ez_inc_field.clone(), X_init)

            X_cmp = X_recon[:, 0, :, :].cpu().numpy() + 1j * X_recon[:, 1, :, :].cpu().numpy()
            eps_recon = np.real((X_cmp + 1) * utils.eps_b) / utils.eps_o
            sigma_recon = -1 * np.imag((X_cmp + 1) * utils.eps_b) * utils.w
            head_ep_recon[ii, 0, :, :] = eps_recon[0, :, :]
            head_ep_recon[ii, 1, :, :] = sigma_recon[0, :, :]

            stroke_X_cmp = stroke_x[:, 0, :, :].cpu().numpy() + 1j * stroke_x[:, 1, :, :].cpu().numpy()
            stroke_eps_recon = np.real((stroke_X_cmp + 1) * utils.eps_b) / utils.eps_o
            stroke_sigma_recon = -1 * np.imag((stroke_X_cmp + 1) * utils.eps_b) * utils.w
            stroke_ep_recon[ii, 0, :, :] = stroke_eps_recon[0, :, :]
            stroke_ep_recon[ii, 1, :, :] = stroke_sigma_recon[0, :, :]

            if plot_flag:
                plt.close('all')
                fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(nrows=2, ncols=2)
                ax1.imshow(trg_eps[0, 0, :, :].detach().cpu(), cmap='jet', vmin=0, vmax=78)
                ax2.imshow(trg_eps[0, 1, :, :].detach().cpu(), cmap='jet', vmin=0, vmax=2.8)
                ax3.imshow(eps_recon[0, :, :], cmap='jet', vmin=0, vmax=78)
                ax4.imshow(sigma_recon[0, :, :], cmap='jet', vmin=0, vmax=2.8)
                plt.tight_layout(pad=0.1, w_pad=0.1, h_pad=0.1)
                plt.show()

    return head_ep_recon, stroke_ep_recon, head_ep_gt


if __name__ == '__main__':
    test_dataset = torch.load('../data/inverse_solver_test_data.pt', weights_only=False)

    FD_Ez_inc = torch.from_numpy(np.load('../data/ez_inc.npy'))

    ### Load the 16 forward sovler models
    model_names = ['model_forward_Tx_1.pkl', 'model_forward_Tx_2.pkl', 'model_forward_Tx_3.pkl',
                   'model_forward_Tx_4.pkl', 'model_forward_Tx_5.pkl', 'model_forward_Tx_6.pkl',
                   'model_forward_Tx_7.pkl', 'model_forward_Tx_8.pkl', 'model_forward_Tx_9.pkl',
                   'model_forward_Tx_10.pkl', 'model_forward_Tx_11.pkl', 'model_forward_Tx_12.pkl',
                   'model_forward_Tx_13.pkl', 'model_forward_Tx_14.pkl', 'model_forward_Tx_15.pkl',
                   'model_forward_Tx_16.pkl']

    forward_solvers = []

    for ii in range(len(model_names)):
        fd_ez_inc_tx = FD_Ez_inc[:, :, ii]
        ez_inc_real = torch.real(fd_ez_inc_tx).expand(1, 1, 128, 128)
        ez_inc_imag = torch.imag(fd_ez_inc_tx).expand(1, 1, 128, 128)
        ez_inc_tx_new = torch.cat((ez_inc_real, ez_inc_imag), dim=1).to(device, torch.float)

        model = Unet_forward_model_F.ForwardSolver(out_channels=2, channel_num=Unet_forward_model_F.channels,
                                                   e_inc=ez_inc_tx_new).to(device)
        model.load_state_dict(torch.load('../models/' + model_names[ii], map_location=device))
        model.eval()
        for param in model.parameters():
            param.requires_grad = False

        forward_solvers.append(model)

    u_net_channels = np.array([16, 32, 64, 128, 256])
    inverse_model = NN_DBIM_inverse_solver_model.InverseNet(u_net_channel_nums=u_net_channels).to(device)

    Ez_inc_real = torch.real(FD_Ez_inc).expand(1, 1, 128, 128, 16)
    Ez_inc_imag = torch.imag(FD_Ez_inc).expand(1, 1, 128, 128, 16)

    Ez_inc_new = torch.cat((Ez_inc_real, Ez_inc_imag), dim=1).to(device, torch.float)

    inverse_model.load_state_dict(torch.load('../models/NN_DBIM_inverse_solver.pkl', map_location='mps'))

    head_ep_recon, stroke_ep_recon, head_ep_gt = evaluate(inverse_model, forward_solvers, test_dataset, Ez_inc_new, plot_flag=0)


    print('debug')
