import torch, time
from scipy.io import savemat
from thop import profile
import scipy.io
import os
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
# import FCN
# from model import model as model
from model_v2 import AttU_Net
import muti_dataloader as dataloader
from dataprocessing import SSIM
from dataprocessing import NormalDistributionFitter


os.environ['CUDA_VISIBLE_DEVICES'] = '0'
# model = model.Net().cuda().eval()
model = AttU_Net().cuda().eval()
file_path = "/data/lzp/train_result/AttU_Net_complex/train_1/best.pkl"
test_data_loader = DataLoader(dataset=dataloader.testdataset, num_workers=0, batch_size=1, pin_memory=True,
                             shuffle=False, drop_last=True)
# model.load_state_dict(torch.load(file_path))
state_dict = torch.load(file_path)
from collections import OrderedDict
new_state_dict = OrderedDict()
for k, v in state_dict.items():
    name = k[7:] if k.startswith('module.') else k  # 去掉 'module.' 前缀
    new_state_dict[name] = v
model.load_state_dict(new_state_dict)
x_values = np.arange(1, 33)
ei = SSIM.SSIM()
h = 6
flag = 2

rmse_avg = 0
ssim_avg = 0
if flag == 1:
    for i, (x, label1, label2) in enumerate(test_data_loader):
        if i == h:
            X_train, Y_train, Z_train = x.cuda(), label1, label2
            out_val, z = model(X_train.reshape([1, 1, 256, 256]))
            # out_val, z = model(X_train.reshape([1, 1, 1024, 1024]))
            Z_train = Z_train[0, :]
            z = z.cpu().detach().numpy()
            # plt.bar(x_values, Z_train, color='blue', alpha=0.7)
            # plt.ylim(-1, 2)
            # plt.grid(axis='y', alpha=0.75)
            # plt.xlabel('Value')
            # plt.ylabel('Frequency')
            # # plt.title('Histogram of 1D Array')
            # plt.show()

            # tensor_min = torch.min(Y_train)
            # tensor_max = torch.max(Y_train)
            # y_1 = (Y_train - tensor_min) / (tensor_max - tensor_min)
            # Y_train = Y_train.reshape([1, 66]).detach().numpy()
            # z = Z_train.cpu().detach().numpy()
            Z1 = Z_train.detach().numpy().astype(np.float64)
            # maxZ = np.max(Z1)
            # minZ = np.min(Z1)
            # z = np.zeros(66)
            # z[3:33] = Z1[3:33]+np.random.uniform(-0.05, 0.05, size=30)
            # z[33:-1] = Z1[33:-1]+np.random.uniform(-0.03, 0.03, size=32)
            # plt.bar(x_values, z, color='red', alpha=0.7)
            # plt.ylim(-1, 2)
            # plt.grid(axis='y', alpha=0.75)
            # plt.xlabel('Value')
            # plt.ylabel('Frequency')
            # # plt.title('Histogram of 1D Array')
            # plt.show()

            # zonal = scipy.io.loadmat("/data/lzp/data/ketaconus/compare/Zonal_zer.mat")
            # zonal = zonal['moments'].flatten()
            # zonal = zonal / 4 * 1.5

            # 设置条形的宽度
            bar_width = 0.3

            # 设置索引，以使每个条形居中
            index = np.arange(32)

            # 绘制柱状图
            plt.bar(index, Z1, bar_width, label='Group 1')
            plt.bar(index + bar_width, z[0, :], bar_width, label='Group 2')
            # plt.bar(index + 2 * bar_width, zonal[3:32], bar_width, label='Group 3')
            plt.ylim(-2, 2.5)
            plt.grid(axis='y', alpha=0.75)
            # 设置x轴刻度
            # plt.xticks(index + bar_width / 2, x_values)
            plt.show()


            diff = Z1 - z
            diffs = diff ** 2
            mean_diff_squared = np.mean(diffs)
            # 计算均方根误差
            rmse = np.sqrt(mean_diff_squared)
            print(rmse)

            plt.bar(x_values, diff, color='blue', alpha=0.7)
            plt.grid(axis='y', alpha=0.75)
            plt.grid(axis='y', alpha=0.75)
            plt.ylim(-1, 2)
            plt.xlabel('Value')
            plt.ylabel('Frequency')
            # plt.title('Histogram of 1D Array')
            plt.show()

            # data_dict = {'gt': Z1,
            #              'rec': z,
            #              'res': diff}
            # savemat('/data/lzp/data/ketaconus/compare/net/res/net_zer.mat', data_dict)


            break

if flag == 2:
    ssim = []
    rmsel = []

    for i, (x, label1, label2) in enumerate(test_data_loader):
        X_train, Y_train, Z_train = x.cuda(), label1, label2
        Y_train = Y_train.reshape([1, 1, 256, 256]).cpu()
        out_val, z = model(X_train.reshape([1, 1, 256, 256]))
        ssim.append(ei(out_val.cpu(), Y_train).item())
        diff = z.cpu().detach().numpy() - np.array(Z_train)
        # diff = out_val.cpu().detach().numpy() - np.array(Y_train)
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
