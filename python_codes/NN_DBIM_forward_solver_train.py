import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from NN_DBIM_forward_solver_model import UNet
import utils


if torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")


def train(model, train_set, Ez_inc, Gezz, loss_weight_1, loss_weight_2, loss_weight_3, optimizer, criterion, clip):
    model.train()

    epoch_loss = 0

    for ii, data in enumerate(train_set):
        src_EP = data[0].to(device, torch.float)

        mask = data[2].to(device, torch.float)
        mask_inv = torch.abs(mask - 1)

        Ez_inc_real = torch.real(Ez_inc).expand(src_EP.shape[0], 1, Ez_inc.shape[0], Ez_inc.shape[1])
        Ez_inc_imag = torch.imag(Ez_inc).expand(src_EP.shape[0], 1, Ez_inc.shape[0], Ez_inc.shape[1])
        Ez_inc_new = torch.cat((Ez_inc_real, Ez_inc_imag), dim=1).to(device, torch.float)

        src = torch.cat((src_EP, Ez_inc_new), dim=1)

        trg = data[1].to(device, torch.float)

        optimizer.zero_grad()

        y_tot, y_scat, recon_e_tot, recon_e_scat = model(src)

        recon_e_tot_complex = recon_e_tot[:, 0, :, :] + 1j * recon_e_tot[:, 1, :, :]
        recon_e_scat_complex = recon_e_scat[:, 0, :, :] + 1j * recon_e_scat[:, 1, :, :]

        trg_tot_new = torch.zeros((trg.shape[0], trg.shape[1], trg.shape[2], trg.shape[3])).to(device, torch.float)
        trg_scat_new = torch.zeros((trg.shape[0], trg.shape[1], trg.shape[2], trg.shape[3])).to(device, torch.float)

        gama_k = trg[:, 0, :, :] * utils.eps_o * (utils.w ** 2 * utils.u_o) + 1j * (
                    -1 * trg[:, 1, :, :] * utils.w * utils.u_o)

        trg_tot_new[:, 0, :, :] = torch.real(gama_k * recon_e_tot_complex)
        trg_tot_new[:, 1, :, :] = torch.imag(gama_k * recon_e_tot_complex)

        trg_scat_new[:, 0, :, :] = torch.real(gama_k * recon_e_scat_complex)
        trg_scat_new[:, 1, :, :] = torch.imag(gama_k * recon_e_scat_complex)

        e_scat_mom = utils.es_from_gezz(Gezz, recon_e_tot, trg)

        e_scat_mom_new = torch.zeros(
            (e_scat_mom.shape[0], e_scat_mom.shape[1], e_scat_mom.shape[2], e_scat_mom.shape[3])).to(device,
                                                                                                     torch.float)
        e_scat_mom_complex = e_scat_mom[:, 0, :, :] + 1j * e_scat_mom[:, 1, :, :]

        e_scat_mom_new[:, 0, :, :] = torch.real(gama_k * e_scat_mom_complex)
        e_scat_mom_new[:, 1, :, :] = torch.imag(gama_k * e_scat_mom_complex)

        loss_tot = criterion(y_tot * mask, trg_tot_new * mask)
        loss_scat_wave_equation = criterion(y_scat * mask_inv, trg_scat_new * mask_inv)
        loss_scat_mom = criterion(e_scat_mom_new * mask_inv, trg_scat_new * mask_inv)

        loss = loss_weight_1 * loss_scat_wave_equation + loss_weight_2 * loss_tot + loss_weight_3 * loss_scat_mom

        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)

        optimizer.step()

        epoch_loss += loss.item()

    return epoch_loss / len(train_set)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == '__main__':

    train_dataset = torch.load('../data/forward_solver_train_data.pt', weights_only=False)

    FD_Ez_inc = torch.from_numpy(np.load('../data/ez_inc.npy'))
    Gezz = torch.from_numpy(np.load('../data/Gezz.npy'))

    FD_Ez_inc = torch.mean(FD_Ez_inc, dim=2)
    Ez_inc_real = torch.real(FD_Ez_inc).expand(1, 1, 128, 128)
    Ez_inc_imag = torch.imag(FD_Ez_inc).expand(1, 1, 128, 128)
    Ez_inc_new = torch.cat((Ez_inc_real, Ez_inc_imag), dim=1).to(device, torch.float)

    GT_Nx = 128
    GT_Ny = 128

    channels = np.array([64, 128, 256, 512, 1024])

    model = UNet(out_channels=2, channel_num=channels, e_inc=Ez_inc_new).to(device)
    criterion = nn.MSELoss()
    CLIP = 1
    learning_rate = 1e-5
    loss_weight_1 = 0.2
    loss_weight_2 = 0.1
    loss_weight_3 = 0.2
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    print(f'The model has {count_parameters(model):,} trainable parameters')

    for epoch in range(1, 10001):

        train_loss = train(model, train_dataset, FD_Ez_inc, Gezz, loss_weight_1, loss_weight_2, loss_weight_3, optimizer, criterion, CLIP)
        print(f"Train Loss at the {epoch}th epoch is {train_loss}\n")

    print('debug')
