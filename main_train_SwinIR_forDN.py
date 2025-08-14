import os.path
import math
import argparse
import random
import numpy as np
import logging
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
import torch

from utils import utils_logger
from utils import utils_image as util
from utils import utils_option as option
from utils.utils_dist import get_dist_info, init_dist

from data.select_dataset import define_Dataset
from models.select_model import define_Model
import os
import torch
import warnings
from tqdm import tqdm
from datetime import datetime

import matplotlib.pyplot as plt

from models.loss_ssim import SSIMLoss

import matplotlib.pyplot as plt
import torchvision.transforms.functional as TF

# --------------------------------------------------------------------------------------
# 경고 무시
warnings.filterwarnings("ignore", category=UserWarning, module="torch")

# --------------------------------------------------------------------------------------


# --------------------------------------------------------------------------------------
# VRAM 사용량 확인하기    
import gc

def print_detailed_vram_usage(note=''):
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()

    print(f"\n[VRAM Monitoring] [{note}]")
    print(f"Allocated  : {torch.cuda.memory_allocated() / 1024**2:.1f} MB  /", f"  Cached     : {torch.cuda.memory_reserved() / 1024**2:.1f} MB  /",
          f"  Max alloc  : {torch.cuda.max_memory_allocated() / 1024**2:.1f} MB  /", f"  Max reserved: {torch.cuda.max_memory_reserved() / 1024**2:.1f} MB\n")
    
# --------------------------------------------------------------------------------------


'''
# --------------------------------------------
# training code for MSRResNet
# --------------------------------------------
# Kai Zhang (cskaizhang@gmail.com)
# github: https://github.com/cszn/KAIR
# --------------------------------------------
# https://github.com/xinntao/BasicSR
# --------------------------------------------
'''


def main(json_path='options/train_swinir_sr_lightweight.json'):
    

    '''
    # ----------------------------------------
    # Step--1 (prepare opt)
    # ----------------------------------------
    '''

    parser = argparse.ArgumentParser()
    parser.add_argument('--opt', type=str, default=json_path, help='Path to option JSON file.')
    parser.add_argument('--launcher', default='pytorch', help='job launcher')
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument('--dist', default=False)

    opt = option.parse(parser.parse_args().opt, is_train=True)
    opt['dist'] = parser.parse_args().dist
    # ----------------------------------------
    # distributed settings
    # ----------------------------------------
    if opt['dist']:
        init_dist('pytorch')
    opt['rank'], opt['world_size'] = get_dist_info()

    if opt['rank'] == 0:
        util.mkdirs((path for key, path in opt['path'].items() if 'pretrained' not in key))

    # ----------------------------------------
    # update opt
    # ----------------------------------------
    # -->-->-->-->-->-->-->-->-->-->-->-->-->-
    
    ##########################################################################################################
    # 이전 학습하던 모델을 그대로 이어서 학습하고 싶으면 아래 코드를 주석 해제하기 #####################################
    
    
    # init_iter_G, init_path_G = option.find_last_checkpoint(opt['path']['models'], net_type='G')

    # if init_path_G is not None:
    #     opt['path']['pretrained_netG'] = init_path_G
    
    ############################################################################################################
        
    init_iter_G = 0

    
    current_step = init_iter_G
    
    border = opt['scale']
    # --<--<--<--<--<--<--<--<--<--<--<--<--<-

    # ----------------------------------------
    # save opt to  a '../option.json' file
    # ----------------------------------------
    if opt['rank'] == 0:
        option.save(opt)

    # ----------------------------------------
    # return None for missing key
    # ----------------------------------------
    opt = option.dict_to_nonedict(opt)

    # ----------------------------------------
    # configure logger
    # ----------------------------------------
    if opt['rank'] == 0:
        logger_name = 'train'
        utils_logger.logger_info(logger_name, os.path.join(opt['path']['log'], logger_name+'.log'))
        logger = logging.getLogger(logger_name)
        logger.info(option.dict2str(opt))    # json 기반 설정 파일을 파싱해서 사람이 읽기 쉬운 텍스트 형식으로 바꿔 출력해주는 함수

    # ----------------------------------------
    # seed
    # ----------------------------------------
    seed = opt['train']['manual_seed']
    if seed is None:
        seed = random.randint(1, 10000)
    print('Random seed: {}'.format(seed))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    '''
    # ----------------------------------------
    # Step--2 (creat dataloader)
    # ----------------------------------------
    '''

    # ----------------------------------------
    # 1) create_dataset
    # 2) creat_dataloader for train and test
    # ----------------------------------------
    for phase, dataset_opt in opt['datasets'].items():
        if phase == 'train':
            train_set = define_Dataset(dataset_opt)    # select_dataset.py에서 가져온 함수
            
            train_size = int(math.ceil(len(train_set) / dataset_opt['dataloader_batch_size']))
            if opt['rank'] == 0:
                logger.info('Number of train images: {:,d}, iters: {:,d}'.format(len(train_set), train_size))
            if opt['dist']:
                train_sampler = DistributedSampler(train_set, shuffle=dataset_opt['dataloader_shuffle'], drop_last=True, seed=seed)
                train_loader = DataLoader(train_set,
                                          batch_size=dataset_opt['dataloader_batch_size']//opt['num_gpu'],
                                          shuffle=False,
                                          num_workers=dataset_opt['dataloader_num_workers']//opt['num_gpu'],
                                          drop_last=False,
                                          pin_memory=True,
                                          sampler=train_sampler)
            else:
                train_loader = DataLoader(train_set,
                                          batch_size=dataset_opt['dataloader_batch_size'],
                                          shuffle=dataset_opt['dataloader_shuffle'],
                                          num_workers=dataset_opt['dataloader_num_workers'],
                                          drop_last=False,
                                          pin_memory=True)
        # elif phase == 'val':
        #     val_set = define_Dataset(dataset_opt)
        #     val_loader = DataLoader(val_set, batch_size=1,
        #                             shuffle=False, num_workers=1,
        #                             drop_last=False, pin_memory=True)
        
        
        elif phase == 'val':
            val_set = define_Dataset(dataset_opt)
            val_loader = DataLoader(val_set,
                                    batch_size=dataset_opt.get('dataloader_batch_size', 1),  # default = 1
                                    shuffle=False,
                                    num_workers=dataset_opt.get('dataloader_num_workers', 1),  # default = 1
                                    drop_last=False,
                                    pin_memory=True)
    
        elif phase == 'test':
            test_set = define_Dataset(dataset_opt)
            test_loader = DataLoader(test_set, batch_size=1,
                                     shuffle=False, num_workers=1,
                                     drop_last=False, pin_memory=True)
        else:
            raise NotImplementedError("Phase [%s] is not recognized." % phase)

    '''
    # ----------------------------------------
    # Step--3 (initialize model)
    # ----------------------------------------
    '''
    # 설정 opt를 바탕으로 실제 모델 클래스 인스턴스를 생성
    model = define_Model(opt)
    
    
    
    model_pretrained = define_Model(opt)
    model_pretrained.load_network(
    load_path='model_zoo/1100_G_2025-07-22_10-34-36.pth',
    network=model_pretrained.netG,
    strict=True,
    param_key='params' 
    )
    
    # 학습을 위한 모든 내부 설정을 초기화
    model.init_train()
    
    if opt['rank'] == 0:
        
        # SwinIR 모델의 전체 네트워크 구조를 문자열로 출력해서 로그에 기록하는 역할
        # logger.info(model.info_network())
        
        # 모델의 파라미터 개수와 통계 요약을 출력
        # logger.info(model.info_params())
        pass
    '''
    # ----------------------------------------
    # Step--4 (main training)
    # ----------------------------------------
    '''
    # Early Stopping 변수 추가
    # best_psnr = 0.0
    best_loss = float('inf')
    patience = 7
    counter = 0
    early_stop = False
    min_delta = 0.0001
    train_loss_log = []
    val_loss_log = []


    for epoch in range(200):  
        if opt['dist']:
            train_sampler.set_epoch(epoch + seed)
            
        train_pbar = tqdm(train_loader, desc=f"[Epoch {epoch}] Training", ncols=100)
    
        for _, train_data in enumerate(train_pbar):

            current_step += 1
            
            # -------------------------------
            # 1) update learning rate
            # -------------------------------
            model.update_learning_rate(current_step)
            

            # -------------------------------
            # 2) feed patch pairs
            # -------------------------------
            model.feed_data(train_data)
            
            
            # -------------------------------
            # 3) optimize parameters 
            # !!!! 실제 학습이 진행되는 곳 !!!! 
            # -------------------------------
            model.optimize_parameters(current_step)
            
            print_detailed_vram_usage(f"After optimize_parameters, step {current_step}")
            
            print(f"[DEBUG] 현재 학습 step: {current_step}, 배치 크기: {train_data['L'].shape}")
                        
            # -------------------------------
            # 4) training information
            # -------------------------------

            # 로그 출력 시점
            
            # 각 threshold 변경해주기 (일정 checkpoint마다 출력하도록 변경해주어야함)
            
            # opt['train']['checkpoint_print'] = 1
            # opt['train']['checkpoint_save'] = 1
            # opt['train']['checkpoint_test'] = 1
            
            if current_step % opt['train']['checkpoint_print'] == 0 and opt['rank'] == 0:
                
                logs = model.current_log()  # such as loss
                
                
                
                g_loss = logs.get('G_loss', 0)
                print(f"[DEBUG] Step {current_step} | G_loss: {g_loss}")
                
                
                message = '<epoch:{:3d}, iter:{:8,d}, lr:{:.3e}> '.format(epoch, current_step, model.current_learning_rate())
                for k, v in logs.items():  # merge log information into message
                    message += '{:s}: {:.3e} '.format(k, v)
                # logger.info(message)

                train_pbar.set_postfix({
                    'Step': current_step,
                    'Loss': logs.get('G_loss', 0) if 'logs' in locals() else None,
                    'LR': model.current_learning_rate()
                })
                
                train_loss_log.append((current_step, logs.get('G_loss', 0) if 'logs' in locals() else None))
                
            
            # -------------------------------
            # 5) Early Stopping
            # -------------------------------

            loss_type = opt['train']['G_lossfn_type']
            ssim_loss_fn = SSIMLoss().to(device)

            if current_step % opt['train']['checkpoint_save'] == 0 and opt['rank'] == 0:
                

                # val_psnr = 0.0
                idx = 0
                val_loss = 0
                
                for val_data in val_loader:
                    idx += 1
                    model.feed_data(val_data)
                    model.test()
                    visuals = model.current_visuals()
                    
                    E_tensor = visuals['E'].detach().unsqueeze(0).to(device) 
                    H_tensor = visuals['H'].detach().unsqueeze(0).to(device) 
                    
                    if loss_type == 'ssim':
                        loss = 1 - ssim_loss_fn(E_tensor, H_tensor).item()
                    
                    elif loss_type == 'l1':
                        loss = torch.mean(torch.abs(E_tensor - H_tensor)).item()
                        
                    elif loss_type == 'l2':
                        mse_loss_fn = torch.nn.MSELoss().to(E_tensor.device)
                        loss = mse_loss_fn(E_tensor, H_tensor).item()
                        
                    elif loss_type == 'psnr':
                        
                        # 나중에 PSNR로 쓸거면 수정해야함.
                        E_img = util.tensor2uint(visuals['E'])
                        H_img = util.tensor2uint(visuals['H'])
                        psnr = util.calculate_psnr(E_img, H_img, border = border)
                        val_psnr += psnr
                        loss = np.mean(np.abs(E_img.astype(np.float32) - H_img.astype(np.float32)))
                        
                    elif loss_type == 'perceptual':
                        loss = model.G_lossfn(E_tensor, H_tensor).item()
                        
                    val_loss += loss
                    
#######################################################################################################################################
# ------------------------------------------------------------------------------------------------------------------------------------#
                    # if idx == 1:
                        
                    #     model_pretrained.feed_data(val_data)
                    #     model_pretrained.test()
                    #     visuals_pretrained = model_pretrained.current_visuals()
                        
                        
                    #     # fig, axs = plt.subplots(1, 4, figsize=(16, 4))
                    #     # axs[0].imshow(TF.to_pil_image(visuals['L'].cpu().squeeze(0)))
                    #     # axs[0].set_title('Input (LQ)')

                    #     # axs[1].imshow(TF.to_pil_image(visuals_pretrained['E'].cpu().squeeze(0)))
                    #     # axs[1].set_title('Pretrained SR')

                    #     # axs[2].imshow(TF.to_pil_image(visuals['E'].cpu().squeeze(0)))
                    #     # axs[2].set_title('Fine-tuned SR')

                    #     # axs[3].imshow(TF.to_pil_image(visuals['H'].cpu().squeeze(0)))
                    #     # axs[3].set_title('Ground Truth (HQ)')

                    #     # for ax in axs:
                    #     #     ax.axis('off')
                    #     # plt.tight_layout()
                    #     # plt.savefig(f'학습도중 시각화 확인/visualization_step_{current_step}.png')
                    #     # plt.close(fig)
                        
                    #     fig, axs = plt.subplots(1, 4, figsize=(16, 4))

                    #     # Helper function: 텐서 → PIL 이미지
                    #     def tensor_to_pil(tensor):
                    #         img = tensor.detach().cpu()
                    #         if img.ndim == 4:
                    #             img = img.squeeze(0)
                    #         if img.max() > 1.0:
                    #             img = img / 255.0  # normalize if needed
                    #         return TF.to_pil_image(img)

                    #     axs[0].imshow(tensor_to_pil(visuals['L']))
                    #     axs[0].set_title('Input (LQ)')

                    #     axs[1].imshow(tensor_to_pil(visuals_pretrained['E']))
                    #     axs[1].set_title('Pretrained SR')

                    #     axs[2].imshow(tensor_to_pil(visuals['E']))
                    #     axs[2].set_title('Fine-tuned SR')

                    #     axs[3].imshow(tensor_to_pil(visuals['H']))
                    #     axs[3].set_title('Ground Truth (HQ)')

                    #     for ax in axs:
                    #         ax.axis('off')

                    #     plt.tight_layout()
                    #     os.makedirs('학습도중 시각화 확인', exist_ok=True)
                    #     plt.savefig(f'학습도중 시각화 확인/visualization_step_{current_step}.png')
                    #     plt.close(fig)
                        

#######################################################################################################################################
# ------------------------------------------------------------------------------------------------------------------------------------#
                        
                        
                # val_psnr /= idx
                # logger.info(f'<epoch:{epoch:3d}, step:{current_step:8,d}, Validation PSNR: {val_psnr:.2f}dB')
                
                val_loss /= idx
                logger.info(f'<epoch:{epoch:3d}, step:{current_step:8,d}, Validation {loss_type} Loss: {val_loss:.2f}dB')
                val_loss_log.append((current_step, val_loss))
                
                # Early Stopping 체크
                # if val_psnr > best_psnr:
                #     best_psnr = val_psnr
                #     counter = 0
                    
                #     # -------------------------------
                #     # 6) save model
                #     # -------------------------------
                #     logger.info(f"\n[EarlyStopping] New best PSNR: {best_psnr:.2f}dB - model saved as best_model.pth\n")
                #     model.save(current_step)
                
                # else:
                #     counter += 1
                #     logger.info(f"\n[EarlyStopping] No improvement. Counter: {counter}/{patience}\n")
                #     if counter >= patience:
                #         logger.info(f"\n[EarlyStopping] Stop training - best PSNR: {best_psnr:.2f}dB\n")
                #         early_stop  = True
                #         break

                # Early Stopping 체크
                if best_loss - val_loss > min_delta:
                    best_loss = val_loss
                    counter = 0
                    print('-----------------------------------------------------\n\n\n\n\n-----------------------------------------------------')
                    logger.info(f"\n[EarlyStopping] New best validation {loss_type} loss: {best_loss:.6f} - model saved as best_model.pth\n")
                    print('-----------------------------------------------------\n\n\n\n\n-----------------------------------------------------')
                    model.save(current_step)
                else:
                    counter += 1
                    print('-----------------------------------------------------\n\n\n\n\n-----------------------------------------------------')
                    logger.info(f"\n[EarlyStopping] No improvement. Counter: {counter}/{patience}\n")
                    print('-----------------------------------------------------\n\n\n\n\n-----------------------------------------------------')
                    if counter >= patience:
                        print('-----------------------------------------------------\n\n\n\n\n-----------------------------------------------------')
                        logger.info(f"\n[EarlyStopping] Stop training - best validation {loss_type} loss: {best_loss:.6f}\n")
                        print('-----------------------------------------------------\n\n\n\n\n-----------------------------------------------------')
                        early_stop = True
                        break


        if early_stop:
            
        
            # 1. 학습 (train) 로그: 매 스텝마다 기록
            train_steps = [step for step, _ in train_loss_log]
            train_losses = [loss for _, loss in train_loss_log]

            # 2. 검증 (val) 로그: 20 스텝마다 기록
            val_steps = [step for step, _ in val_loss_log]
            val_losses = [loss for _, loss in val_loss_log]
            
            plt.figure(figsize=(10, 6))
            plt.plot(train_steps, train_losses, label=f'Train {loss_type} Loss', color='blue', linewidth=1)
            plt.plot(val_steps, val_losses, label=f'Validation {loss_type} Loss', color='red', marker='o', linestyle='--', linewidth=1.5)

            plt.title(f"Train vs Validation {loss_type} Loss over Steps")
            plt.xlabel("Step")
            plt.ylabel(f"{loss_type} Loss")
            plt.grid(True)
            plt.legend()
            plt.tight_layout()
            
            timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

            # 파일 이름에 붙이기
            filename = f"loss_comparison_curve_{timestamp}.png"
            save_path = os.path.join('Loss_Records', filename)   
            plt.savefig(save_path) 
            plt.show()
            
            break

    # Test는 따로 진행할 예정

    # # -------------------------------
    # # 6) testing
    # # 학습 종료 후 테스트 성능 확인
    # # -------------------------------
    # if opt['rank'] == 0:
    #     avg_psnr = 0.0
    #     idx = 0
        
    #     for test_data in test_loader:
    #         idx += 1
    #         image_name_ext = os.path.basename(test_data['L_path'][0])
    #         img_name, ext = os.path.splitext(image_name_ext)

    #         img_dir = os.path.join(opt['path']['images'], img_name)
    #         util.mkdir(img_dir)

    #         model.feed_data(test_data)
    #         model.test()

    #         visuals = model.current_visuals()
    #         E_img = util.tensor2uint(visuals['E'])
    #         H_img = util.tensor2uint(visuals['H'])
            
    #         # -----------------------
    #         # save estimated image E
    #         # -----------------------
    #         save_img_path = os.path.join(img_dir, '{:s}_{:d}.png'.format(img_name, current_step))
    #         util.imsave(E_img, save_img_path)

    #         # -----------------------
    #         # calculate PSNR
    #         # -----------------------
            
    #         current_psnr = util.calculate_psnr(E_img, H_img, border=border)

    #         logger.info('{:->4d}--> {:>10s} | {:<4.2f}dB'.format(idx, image_name_ext, current_psnr))

    #         avg_psnr += current_psnr

    #     avg_psnr = avg_psnr / idx
    
    #     # testing log
    #     logger.info('<epoch:{:3d}, iter:{:8,d}, Average PSNR : {:<.2f}dB\n'.format(epoch, current_step, avg_psnr))
                
                
                    # -----------------------------------------------------------------------------------
                    # Debug: 모델에 들어간 이미지와 GT 이미지 shape 출력
                    # L_img = test_data['L'][0]  # low-quality input
                    # H_img_gt = test_data['H'][0]  # ground truth
                    # print(f"\n[DEBUG] 테스트 중인 이미지: {image_name_ext}")
                    # print(f"[DEBUG] Input LQ 이미지 (test_data['L']): {L_img.shape}")
                    # print(f"[DEBUG] Ground Truth HR 이미지 (test_data['H']): {H_img_gt.shape}")

                    # visuals = model.current_visuals()
                    # E_img_tensor = visuals['E']
                    # H_img_tensor = visuals['H']

                    # # Debug: 모델 출력 E와 GT tensor 사이즈 확인
                    # if isinstance(E_img_tensor, torch.Tensor):
                    #     print(f"[DEBUG] 모델 출력 SR 이미지 (visuals['E']): {E_img_tensor.size()}")
                    # else:
                    #     print(f"[DEBUG] 모델 출력 SR 이미지 (visuals['E']): {E_img_tensor.shape}")

                    # if isinstance(H_img_tensor, torch.Tensor):
                    #     print(f"[DEBUG] GT tensor (visuals['H']): {H_img_tensor.size()}")
                    # else:
                    #     print(f"[DEBUG] GT tensor (visuals['H']): {H_img_tensor.shape}")
                    # -----------------------------------------------------------------------------------
                        
                        


if __name__ == '__main__':
    
    import torch.multiprocessing as mp
    mp.set_start_method('spawn', force = True)
    print(f"CUDA Available: {torch.cuda.is_available()}")
    print(f"Using device: {torch.cuda.current_device()} - {torch.cuda.get_device_name(torch.cuda.current_device())}")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    main()