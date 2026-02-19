import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import NN_DBIM_inverse_solver_model
import Unet_forward_model_F


if torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def train(inv_model, for_model, train_dataset, Ez_inc_field, optimizer, criterion, clip):
    inv_model.train()

    for jj in range(len(for_model)):
        for_model[jj].eval()
        for params in for_model[jj].parameters():
            params.requires_grad = False

    epoch_loss = 0
    Ez_inc_field = Ez_inc_field.expand(32, 2, 128, 128, 16)

    for ii, data in enumerate(train_dataset):
        src_Es = data[0].to(device, torch.float)
        trg_head_X = data[1].to(device, torch.float)
        trg_stroke_X = data[2].to(device, torch.float)

        X_init = torch.zeros((src_Es.shape[0], 2, 128, 128)).to(device, torch.float)

        es_gt = torch.zeros((src_Es.shape[0], 3, src_Es.shape[1], src_Es.shape[2], src_Es.shape[3])).to(device, torch.float)
        for jj in range(3):
            es_gt[:, jj, :, :, :] = src_Es

        optimizer.zero_grad()

        X_recon, stroke_x, es_inv = inv_model(for_model, src_Es, Ez_inc_field.clone(), X_init)

        ep_head_loss = criterion(X_recon, trg_head_X)
        ep_stroke_loss = criterion(stroke_x, trg_stroke_X)
        es_loss = criterion(es_inv, es_gt)

        loss = ep_head_loss + ep_stroke_loss + es_loss

        loss.backward()

        torch.nn.utils.clip_grad_norm_(inv_model.parameters(), clip)

        optimizer.step()

        epoch_loss += loss.item()

    return epoch_loss / len(train_dataset)


if __name__ == '__main__':
    train_dataset = torch.load('../data/inverse_solver_train_data.pt', weights_only=False)
    FD_Ez_inc = torch.from_numpy(np.load('../data/ez_inc.npy'))

    ### Load the 16 forward sovler models
    model_names = ['model_forward_Tx_1.pkl', 'model_forward_Tx_2.pkl', 'model_forward_Tx_3.pkl',
                   'model_forward_Tx_4.pkl', 'model_forward_Tx_5.pkl', 'model_forward_Tx_6.pkl',
                   'model_forward_Tx_7.pkl', 'model_forward_Tx_8.pkl', 'model_forward_Tx_9.pkl',
                   'model_forward_Tx_10.pkl', 'model_forward_Tx_11.pkl', 'model_forward_Tx_12.pkl',
                   'model_forward_Tx_13.pkl', 'model_forward_Tx_14.pkl', 'model_forward_Tx_15.pkl',
                   'model_forward_Tx_16.pkl']

    forward_models = []

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

        forward_models.append(model)

    Ez_inc_real = torch.real(FD_Ez_inc).expand(1, 1, 128, 128, 16)
    Ez_inc_imag = torch.imag(FD_Ez_inc).expand(1, 1, 128, 128, 16)

    Ez_inc_new = torch.cat((Ez_inc_real, Ez_inc_imag), dim=1).to(device, torch.float)

    GT_Nx = 128
    GT_Ny = 128

    u_net_channels = np.array([16, 32, 64, 128, 256])

    inverse_model = NN_DBIM_inverse_solver_model.InverseNet(u_net_channel_nums=u_net_channels).to(device)
    criterion = nn.MSELoss()

    print(f'The model has {count_parameters(inverse_model):,} trainable parameters')

    CLIP = 1
    learning_rate = 1e-5
    optimizer = optim.Adam(inverse_model.parameters(), lr=learning_rate)
    train_loss = torch.zeros(1001)

    for epoch in range(1, 1001):

        train_loss[epoch] = train(inverse_model, forward_models, train_dataset, Ez_inc_new, optimizer, criterion, CLIP)
        print(f"Train Loss at the {epoch}th epoch is {train_loss[epoch]}\n")

    print('debug')

