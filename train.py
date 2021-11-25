import torch
import argparse
from utils.dataloader import dataloader
from model import Model
from tqdm import tqdm
from torchvision import utils
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
import numpy as np
from utils.print_options import print_options

parser = argparse.ArgumentParser(description='Cycle-free CycleGAN')

# Invertible Neural Network Architecture
parser.add_argument('--batch_size', default=1, type=int, help='batch size')
parser.add_argument('--in_channel', default=1, type=int, help='channel number of input image')
parser.add_argument('--n_block', default=4, type=int, help='number of stable coupling layers')
parser.add_argument('--squeeze_num', default=2, type=int, help='image divide for coupling layer')
parser.add_argument('--conv_lu', default=True, help='use LU decomposed convolution instead of plain version')
parser.add_argument('--block_type', default='dense', help='simple: Simple block, dense: Dense block, residual: Residual block')

# Training details
parser.add_argument('--gpu_id', default='0', type=str, help='device_ids for training')
parser.add_argument('--total_iter', default=150001, type=int, help='total training iterations')
parser.add_argument('--visualize_iter', default=5000, type=int, help='visualize test PSNR/SSIM result iterations')
parser.add_argument('--save_model_iter', default=5000, type=int, help='save model weights iterations')
parser.add_argument('--test_num', default=421, type=int, help='test set number')
parser.add_argument('--lr', default=1e-4, type=float, help='learning rate')
parser.add_argument('--lambda_gan', default=1.0, type=float, help='lambda value for gan loss')
parser.add_argument('--lambda_l1', default=10.0, type=float, help='lambda value for L1 loss')

# Image directories
parser.add_argument('--path_SDCT', required=True, type=str, help='Path to target image directory')
parser.add_argument('--path_LDCT', required=True, type=str, help='Path to input image directory')
parser.add_argument('--path_SDCT_test', required=True, type=str, help='Path to target test image folder')
parser.add_argument('--path_LDCT_test', required=True, type=str, help='Path to input test image folder')

opt = parser.parse_args()
print_options(parser, opt)
model = Model(opt)

LDCT_dataset = iter(dataloader(opt.path_LDCT, opt.batch_size, do_shuffle=True))
SDCT_dataset = iter(dataloader(opt.path_SDCT, opt.batch_size, do_shuffle=True))

LDCT_test_dataset = iter(dataloader(opt.path_LDCT_test, 1, do_shuffle=False))
SDCT_test_dataset = iter(dataloader(opt.path_SDCT_test, 1, do_shuffle=False))

with tqdm(range(opt.total_iter)) as pbar:
    for i in pbar:
        LDCT, _ = next(LDCT_dataset)
        SDCT, _ = next(SDCT_dataset)

        model.set_input(LDCT, SDCT)
        model.optimize_parameters()

        pbar.set_description(f'total_D_loss: {model.total_D_loss:.5f}; total_G_loss: {model.total_g_loss:.5f};')

        if (i+1) % opt.visualize_iter == 0:  # for test dataset
            psnr_out = 0
            ssim_out = 0

            psnr_LDCT = 0
            ssim_LDCT = 0

            with torch.no_grad():
                for num in range(opt.test_num):
                    LDCT_test, _ = next(LDCT_test_dataset)
                    SDCT_test, _ = next(SDCT_test_dataset)

                    model.set_input_val(LDCT_test, SDCT_test)
                    model.val_forward()

                    LDCT_numpy = np.squeeze(model.real_LDCT.cpu().numpy())
                    SDCT_numpy = np.squeeze(model.real_SDCT.cpu().numpy())
                    output_numpy = np.squeeze(model.fake_SDCT.cpu().numpy())

                    psnr_LDCT += psnr(SDCT_numpy, LDCT_numpy, data_range=np.amax(SDCT_numpy))
                    ssim_LDCT += ssim(SDCT_numpy, LDCT_numpy, data_range=np.amax(SDCT_numpy))

                    psnr_out += psnr(SDCT_numpy, output_numpy, data_range=np.amax(SDCT_numpy))
                    ssim_out += ssim(SDCT_numpy, output_numpy, data_range=np.amax(SDCT_numpy))

                mean_psnr_out = psnr_out / opt.test_num
                mean_ssim_out = ssim_out / opt.test_num

                mean_psnr_LDCT = psnr_LDCT / opt.test_num
                mean_ssim_LDCT = ssim_LDCT / opt.test_num

                print("\nMetrics on test set \t LDCT_PSNR/SSIM: %2.4f / %1.4f \t Output_PSNR/SSIM: %2.4f / %1.4f"
                    % (mean_psnr_LDCT, mean_ssim_LDCT, mean_psnr_out, mean_ssim_out))

        if (i+1) % opt.save_model_iter == 0:
            torch.save(
                model.netG.state_dict(), f'result/checkpoints/netG_{str(i + 1).zfill(6)}.pt'
            )
            torch.save(
                model.netD.state_dict(), f'result/checkpoints/netD_{str(i + 1).zfill(6)}.pt'
            )



