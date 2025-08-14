import argparse
import cv2
import glob
import numpy as np
from collections import OrderedDict
import pandas as pd
import os
import torch
import requests
import time

from tqdm import tqdm
from models.network_swinir import SwinIR as net
from utils import util_calculate_psnr_ssim as util
from datetime import datetime
from models.loss import PerceptualLoss


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, default='lightweight_sr', help='classical_sr, lightweight_sr, real_sr, '
                                                                     'gray_dn, color_dn, jpeg_car, color_jpeg_car')
    parser.add_argument('--scale', type=int, default=4, help='scale factor: 1, 2, 3, 4, 8') # 1 for dn and jpeg car
    parser.add_argument('--noise', type=int, default=15, help='noise level: 15, 25, 50')
    parser.add_argument('--jpeg', type=int, default=40, help='scale factor: 10, 20, 30, 40')
    parser.add_argument('--training_patch_size', type=int, default=64, help='patch size used in training SwinIR. '
                                       'Just used to differentiate two different settings in Table 2 of the paper. '
                                       'Images are NOT tested patch by patch.')
    parser.add_argument('--large_model', action='store_true', help='use large model, only provided for real image sr')
    parser.add_argument('--model_path', type=str,
                        default='model_zoo/1100_G_2025-07-22_10-34-36.pth')
    parser.add_argument('--folder_lq', type=str, default= '2dceph_all_data', help='input low-quality test image folder')
    parser.add_argument('--folder_gt', type=str, default=None, help='input ground-truth test image folder')
    parser.add_argument('--tile', type=int, default=None, help='Tile size, None for no tile during testing (testing as a whole)')
    parser.add_argument('--tile_overlap', type=int, default=32, help='Overlapping of different tiles')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"현재 사용 중인 디바이스: {device}")
    if device.type == 'cuda':
        print(f"사용 중인 GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("GPU를 사용할 수 없어 CPU를 사용합니다.")
    # set up model
    if os.path.exists(args.model_path):
        print(f'loading model from {args.model_path}')
    else:
        os.makedirs(os.path.dirname(args.model_path), exist_ok=True)
        url = 'https://github.com/JingyunLiang/SwinIR/releases/download/v0.0/{}'.format(os.path.basename(args.model_path))
        r = requests.get(url, allow_redirects=True)
        print(f'downloading model {args.model_path}')
        open(args.model_path, 'wb').write(r.content)

    model = define_model(args)
    model.eval()
    model = model.to(device)
    print("folder_lq:", args.folder_lq)
    print("folder_gt:", args.folder_gt)

    # setup folder and path
    folder, save_dir, border, window_size = setup(args)
    folder = args.folder_lq
    os.makedirs(save_dir, exist_ok=True)
    test_results = OrderedDict()
    test_results['index'] = []
    test_results['filename'] = []
    test_results['origin_size'] = []
    test_results['psnr'] = []
    test_results['ssim'] = []
    test_results['psnr_y'] = []
    test_results['ssim_y'] = []
    test_results['psnrb'] = []
    test_results['psnrb_y'] = []
    test_results['perceptual'] = [] # Perceptual Loss 추가
    test_results['inference_time'] = []
    psnr, ssim, psnr_y, ssim_y, psnrb, psnrb_y, perceptual = 0, 0, 0, 0, 0, 0, 0

    elapsed_times = []
    
    image_paths = sorted(glob.glob(os.path.join(folder, '*')))
    total_images = len(image_paths)
    
    
    now_str = datetime.now().strftime("%Y-%m-%d_%H%M")
    
    for idx, path in enumerate(tqdm(image_paths, desc="Processing Images", total=total_images, ncols=100, leave=True)):
        
        # if idx >= 1:  # 15장까지만 처리
        #     break
        
        
        # read image
        print('\n',path)
        print('---------------------------------')
        imgname, img_lq, img_gt = get_image_pair(args, path)  # image to HWC-BGR, float32
        img_lq = np.transpose(img_lq if img_lq.shape[2] == 1 else img_lq[:, :, [2, 1, 0]], (2, 0, 1))  # HCW-BGR to CHW-RGB
        img_lq = torch.from_numpy(img_lq).float().unsqueeze(0).to(device)  # CHW-RGB to NCHW-RGB
        print(f"img_lq.shape before model input: {img_lq.shape}")
        print(f"[DEBUG] 현재 테스트하고 있는 이미지 : {imgname}")
        
        test_results['index'].append(idx + 1)
        test_results['filename'].append(imgname)
        
        # inference
        with torch.no_grad():
            
            # 추론 시작 시간
            start_time = time.time()
            
            # pad input image to be a multiple of window_size
            _, _, h_old, w_old = img_lq.size()
            h_pad = (h_old // window_size + 1) * window_size - h_old
            w_pad = (w_old // window_size + 1) * window_size - w_old
            img_lq = torch.cat([img_lq, torch.flip(img_lq, [2])], 2)[:, :, :h_old + h_pad, :]
            img_lq = torch.cat([img_lq, torch.flip(img_lq, [3])], 3)[:, :, :, :w_old + w_pad]
            print(f"img_lq.shape before model input: {img_lq.shape}")
            output = test(img_lq, model, args, window_size)
            output = output[..., :h_old * args.scale, :w_old * args.scale]
            
            # 추론 종료 시간
            end_time = time.time()
            elapsed = end_time - start_time
            input_size = (w_old, h_old)
            test_results['origin_size'].append(input_size)
            elapsed_times.append((imgname, elapsed, input_size))
            test_results["inference_time"].append(elapsed)
            print(f"\nInference time for {imgname}: {end_time - start_time:.4f} seconds\n")
            
        # save image
                
        output = output.data.squeeze().float().cpu().clamp_(0, 1).numpy()
        if output.ndim == 3:
            output = np.transpose(output[[2, 1, 0], :, :], (1, 2, 0))  # CHW-RGB to HCW-BGR
        output = (output * 255.0).round().astype(np.uint8)  # float32 to uint8
        
        save_subdir = os.path.join(save_dir, now_str)
        os.makedirs(save_subdir, exist_ok=True)
        
        cv2.imwrite(f'{save_subdir}/{imgname}_SwinIR.png', output)

        # evaluate psnr/ssim/psnr_b
        if img_gt is not None:
            img_gt = (img_gt * 255.0).round().astype(np.uint8)  # float32 to uint8
            img_gt = img_gt[:h_old * args.scale, :w_old * args.scale, ...]  # crop gt
            img_gt = np.squeeze(img_gt)

            psnr = util.calculate_psnr(output, img_gt, crop_border=border)
            ssim = util.calculate_ssim(output, img_gt, crop_border=border)
            test_results['psnr'].append(psnr)
            test_results['ssim'].append(ssim)
            if img_gt.ndim == 3:  # RGB image
                psnr_y = util.calculate_psnr(output, img_gt, crop_border=border, test_y_channel=True)
                ssim_y = util.calculate_ssim(output, img_gt, crop_border=border, test_y_channel=True)
                test_results['psnr_y'].append(psnr_y)
                test_results['ssim_y'].append(ssim_y)
            if args.task in ['jpeg_car', 'color_jpeg_car']:
                psnrb = util.calculate_psnrb(output, img_gt, crop_border=border, test_y_channel=False)
                test_results['psnrb'].append(psnrb)
                if args.task in ['color_jpeg_car']:
                    psnrb_y = util.calculate_psnrb(output, img_gt, crop_border=border, test_y_channel=True)
                    test_results['psnrb_y'].append(psnrb_y)
                    
            ############################################################
            # Perceptual Loss 계산 추가하기         
            
            with torch.no_grad():
                output_tensor = torch.from_numpy(np.transpose(output / 255.0, (2, 0, 1))).unsqueeze(0).float().to(device)
                gt_tensor = torch.from_numpy(np.transpose(img_gt / 255.0, (2, 0, 1))).unsqueeze(0).float().to(device)
                model.G_lossfn = PerceptualLoss().to(device)
                perceptual = model.G_lossfn(output_tensor, gt_tensor).item()
                test_results['perceptual'].append(perceptual)
                
            print('Testing {:d} {:20s} - PSNR: {:.2f} dB; SSIM: {:.4f};'
                  'PSNR_Y: {:.2f} dB; SSIM_Y: {:.4f}; Perceptual: {:.2f} dB.'.
                  format(idx, imgname, psnr, ssim, psnr_y, ssim_y, perceptual))
        else:
            print('Testing {:d} {:20s}'.format(idx, imgname))

        
    if elapsed_times:
        print("\n--- Inference Summary Per Image ---")
        for name, t, shape in elapsed_times:
            print(f"{name:20s} | Time: {t:.4f} sec | Original Size: {shape[1]}x{shape[0]}")

        total_time = sum(t for _, t, _ in elapsed_times)
        avg_time = total_time / len(elapsed_times)
        print(f"\nAverage Inference Time: {avg_time:.4f} seconds")
        print(f"Total Inference Time:   {total_time:.4f} seconds")
        
    # summarize psnr/ssim
    if img_gt is not None:
        ave_psnr = sum(test_results['psnr']) / len(test_results['psnr'])
        ave_ssim = sum(test_results['ssim']) / len(test_results['ssim'])
        print('\n{} \n-- Average PSNR/SSIM(RGB): {:.2f} dB; {:.4f}'.format(save_dir, ave_psnr, ave_ssim))
        if img_gt.ndim == 3:
            ave_psnr_y = sum(test_results['psnr_y']) / len(test_results['psnr_y'])
            ave_ssim_y = sum(test_results['ssim_y']) / len(test_results['ssim_y'])
            print('-- Average PSNR_Y/SSIM_Y: {:.2f} dB; {:.4f}'.format(ave_psnr_y, ave_ssim_y))
        if args.task in ['jpeg_car', 'color_jpeg_car']:
            ave_psnrb = sum(test_results['psnrb']) / len(test_results['psnrb'])
            print('-- Average PSNRB: {:.2f} dB'.format(ave_psnrb))
            if args.task in ['color_jpeg_car']:
                ave_psnrb_y = sum(test_results['psnrb_y']) / len(test_results['psnrb_y'])
                print('-- Average PSNRB_Y: {:.2f} dB'.format(ave_psnrb_y))

    
    # # 현재 날짜 및 시간 문자열 생성
    # now_str = datetime.now().strftime("%Y-%m-%d_%H%M")

    # # 파일명 생성
    # save_folder = "Results_After_FT"
    # save_filename = f"lightweightSR_{now_str}.xlsx"
    # os.makedirs(save_folder, exist_ok=True)
    
    # save_path = os.path.join(save_folder, save_filename)

    # for k, v in test_results.items():
    #     print(f"{k}: {len(v)}")
    
 
    # max_len = max(len(v) if isinstance(v, list) else 0 for v in test_results.values())

    # for k, v in test_results.items():
    #     if not isinstance(v, list):
    #         test_results[k] = [v] * max_len
    #     elif len(v) < max_len:
    #         test_results[k] += [0] * (max_len - len(v))
    
    # df = pd.DataFrame(test_results)
    # df.to_excel(save_path, index=False)

    # print(f"[INFO] 결과가 Excel로 저장되었습니다: {save_path}")
    
# ---------------------------------------------------------------------------------------------------------------------------------------------

def define_model(args):
    # 001 classical image sr
    if args.task == 'classical_sr':
        model = net(upscale=args.scale, in_chans=3, img_size=args.training_patch_size, window_size=8,
                    img_range=1., depths=[6, 6, 6, 6, 6, 6], embed_dim=180, num_heads=[6, 6, 6, 6, 6, 6],
                    mlp_ratio=2, upsampler='pixelshuffle', resi_connection='1conv')
        param_key_g = 'params'
        pretrained_model = torch.load(args.model_path)
        model.load_state_dict(pretrained_model[param_key_g] if param_key_g in pretrained_model.keys() else pretrained_model, strict=True)

    # 002 lightweight image sr
    # use 'pixelshuffledirect' to save parameters
    elif args.task == 'lightweight_sr':
        model = net(upscale=args.scale, in_chans=3, img_size=64, window_size=8,
                    img_range=1., depths=[6, 6, 6, 6], embed_dim=60, num_heads=[6, 6, 6, 6],
                    mlp_ratio=2, upsampler='pixelshuffledirect', resi_connection='1conv')
        
        # -----------------------------------------------------------------------------------------------------------------------------------------
        
        # model.load_state_dict(torch.load(args.model_path)['params'], strict=True)   # Fine-Tuning이 아닌 원 모델을 사용하고 싶을 땐 이것을 사용하기
        model.load_state_dict(torch.load(args.model_path), strict=True)    # Fine-Tuning된 모델을 사용하고 싶을 땐 이것을 사용하기
        
        # -----------------------------------------------------------------------------------------------------------------------------------------

    # 003 real-world image sr
    elif args.task == 'real_sr':
        if not args.large_model:
            # use 'nearest+conv' to avoid block artifacts
            model = net(upscale=4, in_chans=3, img_size=64, window_size=8,
                        img_range=1., depths=[6, 6, 6, 6, 6, 6], embed_dim=180, num_heads=[6, 6, 6, 6, 6, 6],
                        mlp_ratio=2, upsampler='nearest+conv', resi_connection='1conv')
        else:
            # larger model size; use '3conv' to save parameters and memory; use ema for GAN training
            model = net(upscale=4, in_chans=3, img_size=64, window_size=8,
                        img_range=1., depths=[6, 6, 6, 6, 6, 6, 6, 6, 6], embed_dim=248,
                        num_heads=[8, 8, 8, 8, 8, 8, 8, 8, 8],
                        mlp_ratio=2, upsampler='nearest+conv', resi_connection='3conv')
        model.load_state_dict(torch.load(args.model_path)['params_ema'], strict=True)
        
    return model
    

def setup(args):
    # 001 classical image sr/ 002 lightweight image sr
    if args.task in ['classical_sr', 'lightweight_sr']:
        save_dir = f'results/swinir_{args.task}_x{args.scale}'
        folder = args.folder_gt
        border = args.scale
        window_size = 8

    # 003 real-world image sr
    elif args.task in ['real_sr']:
        save_dir = f'results/swinir_{args.task}_x{args.scale}'
        if args.large_model:
            save_dir += '_large'
        folder = args.folder_lq
        border = 0
        window_size = 8
    
    return folder, save_dir, border, window_size


def get_image_pair(args, path):
    (imgname, imgext) = os.path.splitext(os.path.basename(path))
    print(imgname, imgext)
    # 001 classical image sr/ 002 lightweight image sr/ 003 real image sr (load lq-gt image pairs)
    if args.task in ['classical_sr', 'lightweight_sr', 'real_sr']:
        base_gtname = imgname.split('_')[-1]
        
        # GT
        # gt_path = os.path.join(args.folder_gt, f"{base_gtname}{imgext}")
        # print(f"[DEBUG] gt_path: {gt_path}")
        # print(f"[DEBUG] os.path.exists(gt_path): {os.path.exists(gt_path)}")
        # img_gt = cv2.imread(gt_path, cv2.IMREAD_COLOR).astype(np.float32) / 255.

        # LQ 
        img_lq = cv2.imread(f'{args.folder_lq}/{imgname}{imgext}', cv2.IMREAD_COLOR).astype(np.float32) / 255.
        print("[DEBUG] Entered get_image_pair function...")
        
        

    return imgname, img_lq, None  # GT가 없는 경우 None 반환


def test(img_lq, model, args, window_size):
    if args.tile is None:
        # test the image as a whole
        output = model(img_lq)
    else:
        # test the image tile by tile
        b, c, h, w = img_lq.size()
        print(f"b: {b}, c:{c}, h:{h}, w:{w}")
        tile = min(args.tile, h, w)
        assert tile % window_size == 0, "tile size should be a multiple of window_size"
        tile_overlap = args.tile_overlap
        sf = args.scale

        stride = tile - tile_overlap
        h_idx_list = list(range(0, h-tile, stride)) + [h-tile]
        w_idx_list = list(range(0, w-tile, stride)) + [w-tile]
        E = torch.zeros(b, c, h*sf, w*sf).type_as(img_lq)
        W = torch.zeros_like(E)

        for h_idx in h_idx_list:
            for w_idx in w_idx_list:
                print(f"img_lq: {img_lq.shape}")
                # in_patch = img_lq[..., h_idx:h_idx+tile, w_idx:w_idx+tile]
                in_patch = img_lq
                print(f"in_patch: {in_patch}")
                out_patch = model(in_patch)
                out_patch_mask = torch.ones_like(out_patch)

                E[..., h_idx*sf:(h_idx+tile)*sf, w_idx*sf:(w_idx+tile)*sf].add_(out_patch)
                W[..., h_idx*sf:(h_idx+tile)*sf, w_idx*sf:(w_idx+tile)*sf].add_(out_patch_mask)
        output = E.div_(W)

    return output

if __name__ == '__main__':
    main()