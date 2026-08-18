import json
import unittest

import numpy as np
from matplotlib import pyplot as plt


def process_loss(loss_file):
    print(f"file path is {loss_file}")
    # 读取 JSON 文件
    with open(loss_file, "r") as file:
        loss_dict = json.load(file)
    ori_loss_ncc = []
    method_loss_ncc = []
    ori_loss_ssim = []
    method_loss_ssim = []
    ori_loss_mi = []
    method_loss_mi = []
    ori_loss_mse = []
    method_loss_mse = []
    # 打印数据示例
    for study_idx, losses in loss_dict.items():
        # print(f"Study Index: {study_idx}")
        # print(f"  NCC Loss: {losses['ncc']}")
        # print(f"  SSIM Loss: {losses['ssim']}")
        # print(f"  MI Loss: {losses['mi']}")
        # print(f"  MSE Loss: {losses['mse']}")

        ori_loss_ncc.append(losses['ncc'][1])
        ori_loss_ssim.append(losses['ssim'][1])
        ori_loss_mi.append(losses['mi'][1])
        ori_loss_mse.append(losses['mse'][1])

        method_loss_ncc.append(losses['ncc'][0])
        method_loss_ssim.append(losses['ssim'][0])
        method_loss_mi.append(losses['mi'][0])
        method_loss_mse.append(losses['mse'][0])

    ori_mean_list = [np.mean(ori_loss_ncc),np.mean(ori_loss_ssim),np.mean(ori_loss_mi),np.mean(ori_loss_mse) * 1e3]
    method_mean_list = [np.mean(method_loss_ncc),np.mean(method_loss_ssim),np.mean(method_loss_mi),np.mean(method_loss_mse) * 1e3]
    print(f"ori_mean_list is {ori_mean_list}")
    print(f"method_mean_list is {method_mean_list}")

    return [method_loss_ncc,method_loss_ssim,method_loss_mi,method_loss_mse],\
            [ori_loss_ncc,ori_loss_ssim,ori_loss_mi,ori_loss_mse]


def process_loss_avg(loss_file):
    print(f"file path is {loss_file}")
    # 读取 JSON 文件
    with open(loss_file, "r") as file:
        loss_dict = json.load(file)
    ori_loss_ncc = []
    method_loss_ncc = []
    ori_loss_ssim = []
    method_loss_ssim = []
    ori_loss_mi = []
    method_loss_mi = []
    ori_loss_mse = []
    method_loss_mse = []
    # 打印数据示例
    for study_idx, losses in loss_dict.items():
        # print(f"Study Index: {study_idx}")
        # print(f"  NCC Loss: {losses['ncc']}")
        # print(f"  SSIM Loss: {losses['ssim']}")
        # print(f"  MI Loss: {losses['mi']}")
        # print(f"  MSE Loss: {losses['mse']}")

        ori_loss_ncc.append(losses['ncc'][1])
        ori_loss_ssim.append(losses['ssim'][1])
        ori_loss_mi.append(losses['mi'][1])
        ori_loss_mse.append(losses['mse'][1])

        method_loss_ncc.append(losses['ncc'][0])
        method_loss_ssim.append(losses['ssim'][0])
        method_loss_mi.append(losses['mi'][0])
        method_loss_mse.append(losses['mse'][0])

    ori_mean_list = [np.mean(ori_loss_ncc),np.mean(ori_loss_ssim),np.mean(ori_loss_mi),np.mean(ori_loss_mse) * 1e3]
    method_mean_list = [np.mean(method_loss_ncc),np.mean(method_loss_ssim),np.mean(method_loss_mi),np.mean(method_loss_mse) * 1e3]
    print(f"ori_mean_list is {ori_mean_list}")
    print(f"method_mean_list is {method_mean_list}")

    return [np.mean(method_loss_ncc),np.mean(method_loss_ssim)
        ,np.mean(method_loss_mi),np.mean(method_loss_mse)]

def process_loss_dice_avg(loss_file):
    print(f"file path is {loss_file}")
    # 读取 JSON 文件
    with open(loss_file, "r") as file:
        loss_dict = json.load(file)
    ori_loss_ncc = []
    method_loss_ncc = []
    ori_loss_ssim = []
    method_loss_ssim = []
    ori_loss_mi = []
    method_loss_mi = []
    ori_loss_mse = []
    method_loss_mse = []

    ori_loss_dice = []
    method_loss_dice = []

    # 打印数据示例
    for study_idx, losses in loss_dict.items():
        # print(f"Study Index: {study_idx}")
        # print(f"  NCC Loss: {losses['ncc']}")
        # print(f"  SSIM Loss: {losses['ssim']}")
        # print(f"  MI Loss: {losses['mi']}")
        # print(f"  MSE Loss: {losses['mse']}")

        ori_loss_ncc.append(losses['ncc'][1])
        ori_loss_ssim.append(losses['ssim'][1])
        ori_loss_mi.append(losses['mi'][1])
        ori_loss_mse.append(losses['mse'][1])
        ori_loss_dice.append(losses['dice'][1])

        method_loss_ncc.append(losses['ncc'][0])
        method_loss_ssim.append(losses['ssim'][0])
        method_loss_mi.append(losses['mi'][0])
        method_loss_mse.append(losses['mse'][0])
        method_loss_dice.append(losses['dice'][0])

    ori_mean_list = [np.mean(ori_loss_ncc), np.mean(ori_loss_ssim),
                     np.mean(ori_loss_mi), np.mean(ori_loss_mse) * 1e3,
                     np.mean(ori_loss_dice)]
    method_mean_list = [np.mean(method_loss_ncc), np.mean(method_loss_ssim), np.mean(method_loss_mi),
                        np.mean(method_loss_mse) * 1e3,
                        np.mean(method_loss_dice)]
    print(f"ori_mean_list is {ori_mean_list}")
    print(f"method_mean_list is {method_mean_list}")

    return [np.mean(method_loss_ncc), np.mean(method_loss_ssim)
        , np.mean(method_loss_mi), np.mean(method_loss_mse),
            np.mean(method_loss_dice)]


def plot_comparison_single_model(model_name, semi_loss_data, no_semi_loss_data):
    # 定义损失名称
    loss_names = ["NCC", "SSIM", "MI", "MSE"]
    num_losses = len(loss_names)

    # 绘制单个模型的四个指标对比图
    plt.figure(figsize=(12, 8))

    for i in range(num_losses):
        plt.subplot(1, 4, i + 1)

        # 获取当前模型在当前损失指标下的半监督和无监督损失值
        semi_value = semi_loss_data[model_name][i]
        no_semi_value = no_semi_loss_data[model_name][i]

        # 绘制柱状图对比，显示半监督和无监督的对比
        # plt.bar(['Semi', 'No Semi'], [semi_value, no_semi_value], color=['skyblue', 'salmon'])
        plt.bar(['自监督', '半监督'], [semi_value, no_semi_value], color=['skyblue', 'salmon'])
        plt.title(f'{model_name}的{loss_names[i]}')
        plt.ylabel("精度值")
        plt.grid(axis='y', linestyle='--', alpha=0.7)

    # 调整布局，并保存为图片
    plt.tight_layout()
    plt.savefig(f'{model_name}_comparison.png')
    plt.show()

class TestLossND(unittest.TestCase):
    def setUp(self):
        print("start")

    def test_real(self):
        file_path = r"D:\USERS\yq\code\TransMorph_Transformer\TransMorph\yqScript\Train1114\real_trans_simple_loss.json"
        real_trans_lists, ori_lists = process_loss(file_path)
        loss_names = ["NCC Loss", "SSIM Loss", "MI Loss", "MSE Loss"]
        num_losses = len(loss_names)
        plt.figure(figsize=(15, 8))

        for i in range(num_losses):
            plt.subplot(2, 2, i + 1)
            data = [
                real_trans_lists[i],
                ori_lists[i]
            ]
            plt.boxplot(data, notch=True, patch_artist=True, showmeans=True,
                        labels=['Real TransMorph Simple', 'Original'])
            plt.title(f"Boxplot for {loss_names[i]}")
            plt.ylabel("Accuracy Value")
            plt.grid(axis='y', linestyle='--', alpha=0.7)

        plt.tight_layout()
        plt.show()



    def test_plot_comparison(self):
        # 设置中文字体
        plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']  # 或者 'SimHei'
        plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

        # 定义文件路径
        semi_file_paths  = {
            'dust': r"D:\USERS\yq\code\TransMorph_Transformer\TransMorph_MultiScale\dust_semi_0210_loss.json",
            'trans': r"D:\USERS\yq\code\TransMorph_Transformer\TransMorph\yqScript\Train_0210_Semi\trans_0210_semi_loss.json",
            'vxm': r"D:\USERS\yq\code\TransMorph_Transformer\VoxelMorph_YQ\vxm_noSemi_0329_loss.json"
        }

        # no_semi_file_paths = {
        #     'dust': r"D:\USERS\yq\code\TransMorph_Transformer\TransMorph_MultiScale\dust_noSemi_0211_loss.json",
        #     'trans': r"D:\USERS\yq\code\TransMorph_Transformer\TransMorph\yqScript\Train_noSemi_0211\trans_0211_noSemi_loss.json",
        #     'vxm': r"D:\USERS\yq\code\TransMorph_Transformer\VoxelMorph_YQ\vxm_noSemi_0211_loss.json"
        # }
        no_semi_file_paths = {
            'dust': r"D:\USERS\yq\code\TransMorph_Transformer\TransMorph_MultiScale\dust_noSemi_0329_dice_loss.json",
            'trans': r"D:\USERS\yq\code\TransMorph_Transformer\TransMorph\yqScript\Train_noSemi_0211\trans_0217_noSemi_dice_loss.json",
            'vxm': r"D:\USERS\yq\code\TransMorph_Transformer\VoxelMorph_YQ\vxm_noSemi_0329_loss.json"
        }

        # 处理损失数据
        semi_loss_data = {}
        no_semi_loss_data = {}

        # for model in ["dust", "trans", "vxm"]:
        for model in ["dust", "trans", "vxm"]:
            # semi_loss_data[model] = process_loss_avg(semi_file_paths[model])
            # no_semi_loss_data[model] = process_loss_avg(no_semi_file_paths[model])
            # semi_loss_data[model] = process_loss(semi_file_paths[model])
            no_semi_loss_data[model] = process_loss_dice_avg(no_semi_file_paths[model])

        # # 对每个模型生成一张图，展示四个损失指标对比
        # for model in ["dust", "trans", "vxm"]:
        #     plot_comparison_single_model(model, semi_loss_data, no_semi_loss_data)


