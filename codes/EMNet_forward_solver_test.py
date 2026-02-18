import numpy as np
import matplotlib.pyplot as plt
import torch
import utils
import EMNet_forward_solver_train
import time


if torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")


def evaluate(model, test_set, Ez_inc, plot_flag):

    model.eval()

    ep = torch.zeros(len(test_set), 2, 128, 128)
    ep_gt = torch.zeros(len(test_set), 2, 128, 128)
    rmse = torch.zeros(len(test_set), 2)
    e_tot = torch.zeros(len(test_set), 2, 128, 128)
    e_tot_gt = torch.zeros(len(test_set), 2, 128, 128)
    ess_gt = torch.zeros(len(test_set), 2, 16, 1)
    ess_recon = torch.zeros(len(test_set), 2, 16, 1)

    with torch.no_grad():
        for ii, data in enumerate(test_set):
            src_EP = data[0].to(device, torch.float)

            Ez_inc_real = torch.real(Ez_inc).expand(src_EP.shape[0], 1, Ez_inc.shape[0], Ez_inc.shape[1])
            Ez_inc_imag = torch.imag(Ez_inc).expand(src_EP.shape[0], 1, Ez_inc.shape[0], Ez_inc.shape[1])
            Ez_inc_new = torch.cat((Ez_inc_real, Ez_inc_imag), dim=1).to(device, torch.float)

            src = torch.cat((src_EP, Ez_inc_new), dim=1)
            ep_gt[ii, :, :, :] = data[1].to(device, torch.float)
            trg = data[2].to(device, torch.float)

            start_time = time.time()
            y_tot, y_scat, recon_e_tot, recon_e_scat = model(src)
            print('The running time is %s seconds\n' % (time.time() - start_time))

            ep[ii, :, :, :] = utils.curl_curl_operator_2d_ep(recon_e_tot[0, :, :, :].expand(1, 2, 128, 128))

            rmse[ii, 0] = torch.sqrt(torch.sum((ep[ii, 0, :, :] - ep_gt[ii, 0, :, :]) ** 2) /
                                     torch.sum(ep_gt[ii, 0, :, :] ** 2))
            rmse[ii, 1] = torch.sqrt(torch.sum((ep[ii, 1, :, :] - ep_gt[ii, 1, :, :]) ** 2) /
                                     torch.sum(ep_gt[ii, 1, :, :] ** 2))

            e_scat_gt = trg - Ez_inc_new
            e_scat_recon = recon_e_tot - Ez_inc_new

            ess_gt[ii, :, :, :] = utils.extract_ess_data(e_tot_field=trg, e_inc_field=Ez_inc_new)
            ess_recon[ii, :, :, :] = utils.extract_ess_data(e_tot_field=recon_e_tot, e_inc_field=Ez_inc_new)

            if plot_flag:
                plt.close('all')
                fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(nrows=2, ncols=2)
                ax1.matshow(ep[ii, 0, :, :], cmap='jet', vmin=0, vmax=70)
                ax1.set_ylabel('Calculated permittivity', fontweight='bold')
                ax2.matshow(ep[ii, 1, :, :], cmap='jet', vmin=0, vmax=2.5)
                ax2.set_ylabel('Calculated conductivity', fontweight='bold')
                ax3.matshow(ep_gt[ii, 0, :, :], cmap='jet', vmin=0, vmax=70)
                ax3.set_ylabel('Ground truth permittivity', fontweight='bold')
                ax4.matshow(ep_gt[ii, 1, :, :], cmap='jet', vmin=0, vmax=2.5)
                ax4.set_ylabel('Ground truth conductivity', fontweight='bold')
                plt.tight_layout(pad=0.1, w_pad=0.1, h_pad=0.1)
                plt.show()

                plt.close('all')
                fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(nrows=2, ncols=2)
                ax1.matshow(e_scat_recon[0, 0, :, :].detach().cpu(), cmap='jet', vmin=-0.25, vmax=0.25)
                ax1.set_ylabel('EMNet Escat (Real)', fontweight='bold')
                ax2.matshow(e_scat_recon[0, 1, :, :].detach().cpu(), cmap='jet', vmin=-0.25, vmax=0.25)
                ax2.set_ylabel('EMNet Escat (Imag)', fontweight='bold')
                ax3.matshow(e_scat_gt[0, 0, :, :].detach().cpu(), cmap='jet', vmin=-0.25, vmax=0.25)
                ax3.set_ylabel('FDTD Escat (Real)', fontweight='bold')
                ax4.matshow(e_scat_gt[0, 1, :, :].detach().cpu(), cmap='jet', vmin=-0.25, vmax=0.25)
                ax4.set_ylabel('FDTD Escat (Imag)', fontweight='bold')
                plt.tight_layout(pad=0.1, w_pad=0.1, h_pad=0.1)
                plt.show()

                plt.figure()
                plt.plot(ess_gt[ii, 0, :])
                plt.plot(ess_recon[ii, 0, :])

                plt.figure()
                plt.plot(ess_gt[ii, 1, :])
                plt.plot(ess_recon[ii, 1, :])

            e_tot[ii, :, :, :] = recon_e_tot[0, :, :, :]
            e_tot_gt[ii, :, :, :] = trg[0, :, :, :]

        # rmse = torch.mean(rmse, 0)

    return ep, ep_gt, e_tot, e_tot_gt, ess_gt, ess_recon


if __name__ == '__main__':
    test_dataset = torch.load('../data/forward_solver_test_data.pt', weights_only=False)
    FD_Ez_inc = torch.from_numpy(np.load('../data/ez_inc.npy'))

    FD_Ez_inc = torch.mean(FD_Ez_inc, dim=2)
    Ez_inc_real = torch.real(FD_Ez_inc).expand(1, 1, 128, 128)
    Ez_inc_imag = torch.imag(FD_Ez_inc).expand(1, 1, 128, 128)
    Ez_inc_new = torch.cat((Ez_inc_real, Ez_inc_imag), dim=1).to(device, torch.float)

    channels = np.array([64, 128, 256, 512, 1024])

    model = EMNet_forward_solver_train.UNet(out_channels=2, channel_num=channels, e_inc=Ez_inc_new).to(device)

    model_idx = np.arange(5000, 5050, 50)

    model_name = '../models/EMNet_forward_solver.pkl'

    model.load_state_dict(torch.load(model_name, map_location='cpu'))
    ep, ep_gt, e_tot, e_tot_gt, ess_gt, ess_recon = evaluate(model, test_dataset, FD_Ez_inc, plot_flag=1)

    print('debug')
