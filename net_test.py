import torch, time
from scipy.io import savemat
from thop import profile
from scipy.stats import norm
import os
from torch.utils.data import Dataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from model_v2 import AttU_Net as model
import muti_dataloader
from dataprocessing import SSIM
from dataprocessing import NormalDistributionFitter


os.environ['CUDA_VISIBLE_DEVICES'] = '1'
# model = model.Net().cuda().eval()
model = model().cuda().eval()
file_path = "/data/lzp/train_result/AttU_Net_complex/train_2/best.pkl"
test_data_loader = DataLoader(dataset=muti_dataloader.testdataset, num_workers=0, batch_size=1, pin_memory=True,
                             shuffle=False, drop_last=True)
# model.load_state_dict(torch.load(file_path))
state_dict = torch.load(file_path)
from collections import OrderedDict
new_state_dict = OrderedDict()
for k, v in state_dict.items():
    name = k[7:] if k.startswith('module.') else k  # 去掉 'module.' 前缀
    new_state_dict[name] = v
model.load_state_dict(new_state_dict)

ei = SSIM.SSIM()
h = 0
flag = 1
pix = 256
rmse_avg = 0
ssim_avg = 0
if flag == 1:
    for i, (x, label1, label2) in enumerate(test_data_loader):
        if i == h:
            X_train, Y_train, Z_train = x.cuda(), label1, label2
            out_val, z = model(X_train.reshape([1, 1, pix, pix]))
            # out_val = model(X_train.reshape([1, 1, pix, pix])).cpu()
            norm = Normalize(vmin=0, vmax=0.4)

            plt.imshow(np.array(X_train.reshape([pix, pix]).cpu()), cmap='hot', norm=norm, interpolation='nearest')
            plt.colorbar().set_ticks([0.1, 0.3, 0.5, 0.7, 0.9])  # 添加色标
            plt.title('Matrix Visualization')  # 设置标题
            plt.xlabel('Columns')  # 设置X轴标签
            plt.ylabel('Rows')  # 设置Y轴标签
            plt.show()

            tensor_min = torch.min(Y_train)
            tensor_max = torch.max(Y_train)
            y_1 = (Y_train - tensor_min) / (tensor_max - tensor_min)
            plt.imshow(y_1.reshape([pix, pix]).cpu(), cmap='jet', interpolation='nearest', vmin=0, vmax=1)
            plt.colorbar().set_ticks([0, 0.1, 0.3, 0.5, 0.7, 0.9])  # 添加色标
            plt.title('REAL')  # 设置标题
            plt.xlabel('Columns')  # 设置X轴标签
            plt.ylabel('Rows')  # 设置Y轴标签
            plt.show()

            # y_test = model(x[h].reshape([1, 1, 768, 768]))
            # tensor_min = torch.min(out_val)
            # tensor_max = torch.max(out_val)
            # y_test = (out_val - tensor_min) / (tensor_max - tensor_min)
            y1 = out_val.reshape([pix, pix]).cpu().detach().numpy()
            y1_min = np.min(y1)
            y1_max = np.max(y1)
            y1 = (y1-y1_min)/(y1_max-y1_min)

            plt.imshow(y1, cmap='jet', interpolation='nearest', vmin=0, vmax=1)
            plt.colorbar().set_ticks([0, 0.1, 0.3, 0.5, 0.7, 0.9])  # 添加色标
            plt.title('OUTPUT')  # 设置标题
            plt.xlabel('Columns')  # 设置X轴标签
            plt.ylabel('Rows')  # 设置Y轴标签
            plt.show()

            plt.imshow((np.array(Y_train.reshape([pix, pix]).cpu())-y1), cmap='jet', interpolation='nearest', vmin=-1, vmax=1)
            plt.colorbar().set_ticks([-0.9, -0.7, -0.5, -0.3, -0.1, 0, 0.1, 0.3, 0.5, 0.7, 0.9])  # 添加色标
            plt.title('RESIDUAL')  # 设置标题
            plt.xlabel('Columns')  # 设置X轴标签
            plt.ylabel('Rows')  # 设置Y轴标签
            plt.show()
            diff = out_val.cpu().detach().numpy() - np.array(Y_train)
            diffs = diff ** 2
            mean_diff_squared = np.mean(diffs)
            # 计算均方根误差
            rmse = np.sqrt(mean_diff_squared)
            print(rmse)
            print(ei(Y_train.cuda(), out_val.cuda()).item())
            # data_dict = {'my_array': np.array(Y_train.reshape([pix, pix]).cpu())-y1}
            # savemat('/data/lzp/data/ketaconus/compare/net/res/net_res.mat', data_dict)

if flag == 2:
    ssim = []
    rmsel = []

    for i, (x, label1, label2) in enumerate(test_data_loader):
        X_train, Y_train, Z_train = x.cuda(), label1, label2
        Y_train = Y_train.reshape([1, 1, pix, pix]).cpu()
        out_val, Z = model(X_train.reshape([1, 1, pix, pix]))
        ssim.append(ei(out_val.cuda(), Y_train.cuda()).item())

        savemat(f'/data/lzp/data/ketaconus/complex/test/Output_Zer/train_1/Zer_{i+1}.mat', {'Zer': Z.cpu().detach().numpy()})
        diff = out_val.cpu().detach().numpy() - np.array(Y_train)
        diffs = diff ** 2
        mean_diff_squared = np.mean(diffs)
        # 计算均方根误差
        rmse = np.sqrt(mean_diff_squared)
        rmsel.append(rmse)

    # x = list(range(len(ssim)))
    #
    # # 绘制散点图
    # plt.scatter(x, ssim)
    # max_value = max(ssim)
    # plt.show()
    # 找到最大值的索引
    # max_index = ssim.index(max_value)
    # # 输出最大值和索引
    # print(f"列表中的最大值是: {max_value}")
    # print(f"最大值的索引是: {max_index}")
    # plt.scatter(x, rmsel)
    # max_value = min(rmsel)
    # plt.show()
    # # 找到最大值的索引
    # max_index = rmsel.index(max_value)
    # # 输出最大值和索引
    # print(f"列表中的最大值是: {max_value}")
    # print(f"最大值的索引是: {max_index}")

    fitter1 = NormalDistributionFitter.NormalDistributionFitter(rmsel)
    fitter1.fit()
    fitter1.analyze_within_ci()
    fitter1.plot()

    fitter2 = NormalDistributionFitter.NormalDistributionFitter(ssim)
    fitter2.fit()
    fitter2.analyze_within_ci()
    fitter2.plot()

if flag == 3:
    input_data = torch.randn(1, 1, 256, 256).cuda()
    # 测量推理时间
    num_iterations = 100
    start_time = time.time()

    for _ in range(num_iterations):
        output = model(input_data)

    end_time = time.time()
    average_inference_time = (end_time - start_time) / num_iterations
    print(f"Average inference time: {average_inference_time:.6f} seconds")

    x = torch.rand([1, 1, 1024, 1024]).cuda()
    print(model(x).shape)
    flops, params = profile(model, inputs=(x,))

    print(f"FLOPs: {flops}")
    print(f"Parameters: {params}")
