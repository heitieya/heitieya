import sys
sys.path.append("..")
import os
import torch
import torch.nn as nn
from torch.autograd import Variable
from tqdm import tqdm
from torch.utils.data import DataLoader
# 自定义函数
from dataprocessing import SSIM, plot_loss
import model_v2
# from loss.losses import AmplitudeLoss
import muti_dataloader
from freeze import FreezeOtherBranches
# import coeff_dataloader


def create_ordered_folder(base_path, base_name):
    index = 1
    while True:
        folder_name = f"{base_name}_{index}"
        folder_path = os.path.join(base_path, folder_name)
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
            print(f"文件夹 '{folder_path}' 创建成功.")
            break
        index += 1

    return folder_path


def train(model, optimizer, scheduler, criterion1, criterion2, train_loader, val_loader, epoch, error, val_err, vla):
    zerror = []
    zval_err = []
    best_val_loss = float('inf')
    try:
        for e in range(epoch):
            h = 0
            h1 = 0
            e_train = -1
            for i, (data, label1, label2) in enumerate(tqdm(train_loader, desc=f"Epoch {e+1}/{epoch} - Training", ncols=150)):
                X_train, Y_train, Z_train = Variable(data).cuda(), Variable(label1).cuda(), Variable(label2).cuda()
                model.train()
                optimizer.zero_grad()
                Y_pre, Z_pre = model(X_train)
                loss2_1 = criterion2(Y_pre, Y_train)
                loss2_2 = criterion2(Z_pre, Z_train)
                ssim = ei(Y_pre, Y_train).item()
                freezer = FreezeOtherBranches(model, 'branch1')
                freezer.freeze2()
                loss2_1.backward(retain_graph=True)
                freezer.unfreeze2()
                freezer.freeze1()
                loss2_2.backward(retain_graph=True)
                freezer.unfreeze1()
                loss = loss2_1 + 0.5*loss2_2
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                h += loss2_1.item()
                h1 += loss2_2.item()
                e_train += ssim
            h = h / len(train_loader)
            error.append(h)
            zerror.append(h1 / len(train_loader))
            print('Epoch :' + str(e+1) + '   ' + 'MSE Loss: ' + str(h))
            print(f'Train_SSIM Avg: {e_train / len(train_loader)}')

            val_loss = 0
            val_loss1 = 0
            e_val = -1
            for i, (data, label1, label2) in enumerate(tqdm(val_loader, desc=f"Epoch {e+1}/{epoch} - Validation", ncols=150)):
                model.eval()
                X_train, Y_train, Z_train = data.cuda(), label1.cuda(), label2.cuda()
                out_val, z_val = model(X_train)
                val_error1 = criterion2(out_val, Y_train)
                val_error2 = criterion2(z_val, Z_train)
                ssim = ei(out_val, Y_train).item()
                val_loss += val_error1.item()
                val_loss1 += val_error2.item()
                e_val += ssim
            val_loss /= len(val_loader)
            val_err.append(val_loss)
            zval_err.append(val_loss1/len(val_loader))
            scheduler.step(val_loss)
            print('Epoch:' + str(e + 1) + '   ' + 'Val Loss:' + str(val_loss))
            print(f'Val_SSIM Avg: {e_val/len(val_loader)}')
            print('\n')

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model_parameter = model.state_dict()

            if e/10 == 0:
                path = f'SH-net_{model_name}.pkl'
                new_path = os.path.join(folder_path, path)
                torch.save(model.state_dict(), new_path)

    except KeyboardInterrupt:

        print("Training interrupted. Saving model...")

    return model, error, val_err, best_model_parameter, zerror, zval_err


if __name__ == '__main__':
    train_err = []
    val_err = []
    lr = []
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    model_name = 'freeze'
    base_path = f"/data/lzp/train_result/AttU_Net_complex/"
    base_name = 'train'
    folder_path = create_ordered_folder(base_path, base_name)
    vla = float('inf')
    pix = 768
    LR = 0.0001
    epoch = 400
    batch_size = 3
    num_workers = 3

    # # 设置DataLoader   SFI --> coe
    # train_data_loader = DataLoader(dataset=coeff_dataloader.dataset, num_workers=10, batch_size=batch_size, pin_memory=True,
    #                                shuffle=False, drop_last=True)
    # val_data_loader = DataLoader(dataset=coeff_dataloader.valdataset, num_workers=0, batch_size=1, pin_memory=True,
    #                                shuffle=False, drop_last=True)

    # 设置DataLoader   SFI --> wf
    train_data_loader = DataLoader(dataset=muti_dataloader.dataset, num_workers=num_workers, batch_size=batch_size, pin_memory=True,
                                   shuffle=True, drop_last=True)
    val_data_loader = DataLoader(dataset=muti_dataloader.valdataset, num_workers=0, batch_size=1, pin_memory=True,
                                   shuffle=False, drop_last=True)

    # 选择模型
    model = model_v2.AttU_Net_1024()

    # parallel
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs for training.")
        model = nn.DataParallel(model)

    model = model.cuda()
    # 优化器
    optimizer = torch.optim.SGD(model.parameters(), lr=LR)

    # 学习率调整方法
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=15, verbose=True,
                                                           threshold=0.00001, threshold_mode='rel', cooldown=0, min_lr=0
                                                           , eps=1e-8)

    # 设置损失函数
    criterion1 = nn.MSELoss(reduction='sum')
    criterion2 = nn.L1Loss()

    # evaluating indicator
    ei = SSIM.SSIM()

    model, error, val_err, best_model_parameter, zerror, zval_err = train(model, optimizer, scheduler, criterion1, criterion2,
                                   train_data_loader, val_data_loader, epoch, train_err, val_err, vla)

    model.eval().cuda()
    path = f'SH-net_{model_name}.pkl'
    new_path = os.path.join(folder_path, path)
    torch.save(model.state_dict(), new_path)
    torch.save(best_model_parameter, os.path.join(folder_path, 'best.pkl'))
    print("Model saved. Exiting...")
    plot_loss.plot_loss(error, val_err,
                        save_dir=folder_path, filename=f"loss.png")
    plot_loss.plot_loss(zerror, zval_err,
                        save_dir=folder_path, filename=f"zernike_loss.png")
