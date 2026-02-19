import numpy as np
import torch
import torch.nn as nn
import utils


if torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")


maxNorm_coefficent = torch.from_numpy(np.load('../data/Inverse_solver_maxNorm_normalization_coeff_tumor_heads_16_forwardSolver.npy')).to(device, torch.float)


class SEncodeNet(nn.Module):
    def __init__(self):
        super(SEncodeNet, self).__init__()

        self.ch_in = 2
        self.ch_stage_2 = 16
        self.ch_stage_3 = 32
        self.ch_stage_4 = 64
        self.ch_out = 2

        ############################## Define the structure of encoders ##############################
        ### This layer WILL NOT change the size of the features (16 x 16)
        self.cnn_encoder_stage1_layer1 = nn.Sequential(
                nn.Conv2d(self.ch_in, self.ch_stage_2, kernel_size=(3, 3), stride=1, padding=1),
                nn.BatchNorm2d(self.ch_stage_2),
                nn.PReLU()
        )

        ### This layer WILL HALF the size of the features (8 x 8)
        self.cnn_encoder_stage2_layer1 = nn.Sequential(
                nn.Conv2d(self.ch_stage_2, self.ch_stage_3, kernel_size=(2, 2), stride=2, padding=0),
                nn.BatchNorm2d(self.ch_stage_3),
                nn.PReLU()
        )

        ### This layer WILL NOT change the size of the features (8 x 8)
        self.cnn_encoder_stage2_layer2 = nn.Sequential(
                nn.Conv2d(self.ch_stage_3, self.ch_stage_3, kernel_size=(3, 3), stride=1, padding=1),
                nn.BatchNorm2d(self.ch_stage_3),
                nn.PReLU()
        )

        ### This layer WILL HALF the size of the features (4 x 4)
        self.cnn_encoder_stage3_layer1 = nn.Sequential(
            nn.Conv2d(self.ch_stage_3, self.ch_stage_4, kernel_size=(2, 2), stride=2, padding=0),
            nn.BatchNorm2d(self.ch_stage_4),
            nn.PReLU()
        )

        ### This layer WILL NOT change the size of the features (4 x 4)
        self.cnn_encoder_stage3_layer2 = nn.Sequential(
            nn.Conv2d(self.ch_stage_4, self.ch_stage_4, kernel_size=(3, 3), stride=1, padding=1),
            nn.BatchNorm2d(self.ch_stage_4),
            nn.PReLU()
        )

        ############################## Define the structure of fully-connected layers ##############################
        self.full_layer1 = nn.Sequential(
            nn.Linear(4 * 4 * self.ch_stage_4, 16 * 16 * self.ch_stage_4),
            nn.BatchNorm1d(16 * 16 * self.ch_stage_4),
            nn.PReLU()
        )

        ############################## Define the structure of decoders ##############################
        ### This layer WILL DOUBLE the size of the features (32 x 32)
        self.cnn_decoder_stage1_layer1 = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(self.ch_stage_4, self.ch_stage_3, kernel_size=(3, 3), stride=1, padding=1),
            nn.BatchNorm2d(self.ch_stage_3),
            nn.PReLU()
        )

        ### This layer WILL NOT change the size of the features (32 x 32)
        self.cnn_decoder_stage1_layer2 = nn.Sequential(
            nn.Conv2d(self.ch_stage_3, self.ch_stage_3, kernel_size=(3, 3), stride=1, padding=1),
            nn.BatchNorm2d(self.ch_stage_3),
            nn.PReLU()
        )

        ### This layer WILL DOUBLE the size of the features (64 x 64)
        self.cnn_decoder_stage2_layer1 = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(self.ch_stage_3, self.ch_stage_2, kernel_size=(3, 3), stride=1, padding=1),
            nn.BatchNorm2d(self.ch_stage_2),
            nn.PReLU()
        )

        ### This layer WILL NOT change the size of the features (64 x 64)
        self.cnn_decoder_stage2_layer2 = nn.Sequential(
            nn.Conv2d(self.ch_stage_2, self.ch_stage_2, kernel_size=(3, 3), stride=1, padding=1),
            nn.BatchNorm2d(self.ch_stage_2),
            nn.PReLU()
        )

        ### This layer WILL DOUBLE the size of the features (128 x 128)
        self.cnn_decoder_stage3_layer1 = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(self.ch_stage_2, self.ch_out, kernel_size=(3, 3), stride=1, padding=1),
            nn.BatchNorm2d(self.ch_out),
            nn.PReLU()
        )

        ### This layer WILL NOT change the size of the features (128 x 128)
        self.cnn_decoder_stage3_layer2 = nn.Sequential(
            nn.Conv2d(self.ch_out, self.ch_out, kernel_size=(3, 3), stride=1, padding=1),
            nn.BatchNorm2d(self.ch_out),
            nn.PReLU()
        )

        self.cnn_decoder_stage4_layer1 = nn.Conv2d(self.ch_out, self.ch_out, kernel_size=(1, 1), stride=1, padding=0)

    def forward(self, x):
        x = self.cnn_encoder_stage1_layer1(x)

        x = self.cnn_encoder_stage2_layer1(x)
        x = self.cnn_encoder_stage2_layer2(x)

        x = self.cnn_encoder_stage3_layer1(x)
        x = self.cnn_encoder_stage3_layer2(x).view(x.size(0), self.ch_stage_4 * 4 * 4)

        x = self.full_layer1(x).view(x.size(0), self.ch_stage_4, 16, 16)

        x = self.cnn_decoder_stage1_layer1(x)
        x = self.cnn_decoder_stage1_layer2(x)

        x = self.cnn_decoder_stage2_layer1(x)
        x = self.cnn_decoder_stage2_layer2(x)

        x = self.cnn_decoder_stage3_layer1(x)
        x = self.cnn_decoder_stage3_layer2(x)

        return self.cnn_decoder_stage4_layer1(x)


class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(DoubleConv, self).__init__()
        self.conv_1 = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.PReLU()
        )
        self.conv_2 = nn.Sequential(
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.PReLU()
        )

    def forward(self, x):
        x = self.conv_1(x)
        x = self.conv_2(x)

        return x


class DownBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(DownBlock, self).__init__()
        self.double_conv = DoubleConv(in_ch, out_ch)
        self.down_sample = nn.MaxPool2d(2)

    def forward(self, x):
        skip_out = self.double_conv(x)
        down_out = self.down_sample(skip_out)

        return down_out, skip_out


class UpBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(UpBlock, self).__init__()
        self.up_sample = nn.Upsample(scale_factor=2, mode='bicubic', align_corners=True)
        self.double_conv = DoubleConv(in_ch, out_ch)

    def forward(self, down_input, skip_input):
        x = self.up_sample(down_input)
        x = torch.cat([x, skip_input], dim=1)

        return self.double_conv(x)


class UNet(nn.Module):
    def __init__(self, channel_num):
        super(UNet, self).__init__()

        # Downsampling Path
        self.down_conv1 = DownBlock(4, channel_num[0])
        self.down_conv2 = DownBlock(channel_num[0], channel_num[1])
        self.down_conv3 = DownBlock(channel_num[1], channel_num[2])
        self.down_conv4 = DownBlock(channel_num[2], channel_num[3])

        # Bottlenect
        self.double_conv = DoubleConv(channel_num[3], channel_num[4])

        # Upsampling Path
        self.up_conv4 = UpBlock(channel_num[3] + channel_num[4], channel_num[3])
        self.up_conv3 = UpBlock(channel_num[2] + channel_num[3], channel_num[2])
        self.up_conv2 = UpBlock(channel_num[1] + channel_num[2], channel_num[1])
        self.up_conv1 = UpBlock(channel_num[0] + channel_num[1], channel_num[0])

        # Final Convolution
        self.conv_last = nn.Conv2d(channel_num[0], 2, kernel_size=1)

    def forward(self, x):
        x, skip_out_1 = self.down_conv1(x)
        x, skip_out_2 = self.down_conv2(x)
        x, skip_out_3 = self.down_conv3(x)
        x, skip_out_4 = self.down_conv4(x)

        x = self.double_conv(x)

        x = self.up_conv4(x, skip_out_4)
        x = self.up_conv3(x, skip_out_3)
        x = self.up_conv2(x, skip_out_2)
        x = self.up_conv1(x, skip_out_1)

        x = self.conv_last(x)

        return x


class InverseNet(nn.Module):
    def __init__(self, u_net_channel_nums):
        super(InverseNet, self).__init__()

        self.conv_net_itr_1 = SEncodeNet()
        self.conv_net_itr_2 = SEncodeNet()
        self.conv_net_itr_3 = SEncodeNet()

        self.u_net_itr_1 = UNet(u_net_channel_nums)
        self.u_net_itr_2 = UNet(u_net_channel_nums)
        self.u_net_itr_3 = UNet(u_net_channel_nums)

        self.u_net_itr_tot = [self.u_net_itr_1, self.u_net_itr_2, self.u_net_itr_3]
        self.conv_net_itr_tot = [self.conv_net_itr_1, self.conv_net_itr_2, self.conv_net_itr_3]

    def forward(self, for_model, es, ez_field, x):

        itr_num = len(self.u_net_itr_tot)
        ez_inc = ez_field.clone()
        ez_scat = es.clone()

        es_inv = torch.zeros((es.shape[0], itr_num, es.shape[1], es.shape[2], es.shape[3])).to(device, torch.float)

        ### Update es and ez_field during the following iterations
        for itr in range(itr_num):
            itr_conv_model = self.conv_net_itr_tot[itr]
            itr_unet_model = self.u_net_itr_tot[itr]

            delta_x = itr_conv_model(es)

            ez_field_sum = torch.mean(ez_field, dim=4)

            delta_x = torch.cat((delta_x, ez_field_sum), dim=1)
            delta_x = itr_unet_model(delta_x)

            if itr == 2:
                stroke_x = delta_x

            x += delta_x

            x_cmp = x[:, 0, :, :] + 1j * x[:, 1, :, :]

            eps_recon = torch.real((x_cmp + 1) * utils.eps_b) / utils.eps_o
            sigma_recon = -1 * np.imag((x_cmp + 1) * utils.eps_b) * utils.w

            ### Find the pixels with eps less than 1 or sigma less than 0
            for mm in range(eps_recon.shape[0]):
                eps_zero_idx = torch.where(eps_recon[mm, :, :] <= 1)
                sigma_zero_idx = torch.where(sigma_recon[mm, :, :] <= 0)

                eps_recon[mm, eps_zero_idx[0], eps_zero_idx[1]] = utils.eps_r_b
                sigma_recon[mm, sigma_zero_idx[0], sigma_zero_idx[1]] = utils.sigma_b

            X_recon = torch.zeros((x.shape[0], x.shape[1], x.shape[2], x.shape[3])).to(device, torch.float)
            X_recon_cmp = eps_recon * utils.eps_o / utils.eps_b - 1j * sigma_recon / (utils.w * utils.eps_b) - 1
            X_recon[:, 0, :, :] = torch.real(X_recon_cmp)
            X_recon[:, 1, :, :] = torch.imag(X_recon_cmp)

            for forward_cnt in range(len(for_model)):
                forward_solver = for_model[forward_cnt]
                forward_solver.eval()
                for param in forward_solver.parameters():
                    param.requires_grad = False

                ez_field_tmp = forward_solver(torch.cat((X_recon.detach(), ez_inc[:, :, :, :, forward_cnt]), dim=1).to(device, torch.float))
                ez_field_tmp.requires_grad = True

                es_dash = utils.extract_ess_data_rx(e_tot_field=ez_field_tmp, e_inc_field=ez_inc[:, :, :, :, forward_cnt], Rx_N=16).view(es.shape[0], es.shape[1] * es.shape[2])
                es_dash = utils.es_normalization(es_dash, new_min=0.1, new_max=1.1, maxNorm_coeff=maxNorm_coefficent, method='maxNorm')

                es_inv[:, itr, :, forward_cnt, :] = es_dash.view([es.shape[0], es.shape[1], es.shape[2]])
                ez_field[:, :, :, :, forward_cnt] = ez_field_tmp

            es = ez_scat - es_inv[:, itr, :, :, :]

        return X_recon, stroke_x, es_inv
