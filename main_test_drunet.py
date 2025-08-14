import os.path
import logging
import numpy as np
from collections import OrderedDict

import torch
import torch.nn.functional as F

from utils import utils_logger
from utils import utils_image as util

'''
Test script for DRUNet (grayscale, 2-channel input: noisy + sigma_map)
Handles size mismatch by padding input to multiple of 8
'''

def main():

    # ----------------------------------------
    # Settings
    # ----------------------------------------
    noise_level_img = 15
    noise_level_model = noise_level_img
    model_name = 'drunet_gray'
    testset_name = 'FINAL_TEST'
    gt_name = 'genoray_GT'
    task_current = 'dn'
    sf = 1  # unused
    show_img = False

    n_channels_img = 1  # 1채널 흑백 이미지
    nc = [64, 128, 256, 512, 512]
    nb = 4
    model_pool = 'model_zoo'
    testsets = 'testsets'
    results = 'results'
    result_name = testset_name + '_' + model_name
    model_path = os.path.join(model_pool, model_name + '.pth')
    border = 0

    # ----------------------------------------
    # Paths
    # ----------------------------------------
    L_path = os.path.join(testsets, testset_name)
    H_path = os.path.join(testsets, gt_name)
    E_path = os.path.join(results, result_name)
    util.mkdir(E_path)

    logger_name = result_name
    utils_logger.logger_info(logger_name, log_path=os.path.join(E_path, logger_name + '.log'))
    logger = logging.getLogger(logger_name)

    need_H = False
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # ----------------------------------------
    # Load model
    # ----------------------------------------
    from models.network_unet import UNetRes as net
    model = net(in_nc=2, out_nc=1, nc=nc, nb=nb, act_mode='R')
    model.load_state_dict(torch.load(model_path), strict=True)
    model.eval()
    for k, v in model.named_parameters():
        v.requires_grad = False
    model = model.to(device)
    logger.info(f'Model loaded: {model_path}')

    test_results = OrderedDict()
    test_results['psnr'] = []
    test_results['ssim'] = []

    L_paths = util.get_image_paths(L_path)
    H_paths = util.get_image_paths(H_path) if need_H else None

    for idx, img_path in enumerate(L_paths):

        img_name, ext = os.path.splitext(os.path.basename(img_path))

        # 1. 입력 이미지 로드 (1채널)
        img_L = util.imread_uint(img_path, n_channels=1)
        img_L = util.uint2single(img_L)
        img_L = util.single2tensor4(img_L).to(device)  # [1, 1, H, W]

        # 2. 시그마 맵 생성
        sigma_val = noise_level_model / 255.0
        sigma_map = torch.full_like(img_L, sigma_val)

        # 3. 입력 이미지와 시그마맵에 padding 적용 (8의 배수 맞추기)
        _, _, h, w = img_L.shape
        h_pad = (8 - h % 8) % 8
        w_pad = (8 - w % 8) % 8

        img_L = F.pad(img_L, (0, w_pad, 0, h_pad), mode='replicate')
        sigma_map = F.pad(sigma_map, (0, w_pad, 0, h_pad), mode='replicate')

        input_tensor = torch.cat([img_L, sigma_map], dim=1)  # [1, 2, H_pad, W_pad]

        # 4. 복원
        img_E = model(input_tensor)
        img_E = img_E[..., :h, :w]  # crop to original size
        img_E = util.tensor2uint(img_E)  # [H, W]

        # 5. GT 로딩 및 평가
        if need_H:
            img_H = util.imread_uint(H_paths[idx], n_channels=1)
            img_H = img_H.squeeze()  # [H, W, 1] → [H, W]

            psnr = util.calculate_psnr(img_E, img_H, border=border)
            ssim = util.calculate_ssim(img_E, img_H, border=border)
            test_results['psnr'].append(psnr)
            test_results['ssim'].append(ssim)
            logger.info(f'{img_name+ext} - PSNR: {psnr:.2f} dB; SSIM: {ssim:.4f}')

        # 6. 결과 저장
        util.imsave(img_E, os.path.join(E_path, img_name + ext))

    # 7. 평균 PSNR/SSIM
    if need_H:
        avg_psnr = sum(test_results['psnr']) / len(test_results['psnr'])
        avg_ssim = sum(test_results['ssim']) / len(test_results['ssim'])
        logger.info(f'Average PSNR/SSIM - {result_name} - PSNR: {avg_psnr:.2f} dB; SSIM: {avg_ssim:.4f}')


if __name__ == '__main__':
    main()
