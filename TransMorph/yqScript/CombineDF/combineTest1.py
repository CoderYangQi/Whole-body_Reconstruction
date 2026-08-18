import numpy as np
from scipy.optimize import minimize
import unittest,os
import SimpleITK as sitk
import TransMorph.losses
import TransMorph.utils as utils
from torch.utils.data import DataLoader
from TransMorph.data import datasets, trans
import numpy as np
import torch
from torchvision import transforms
import matplotlib.pyplot as plt
from natsort import natsorted
from TransMorph.models.TransMorph import CONFIGS as CONFIGS_TM
import TransMorph.models.TransMorph as TransMorph
def useDF(x_path, flow_path, save_path):
    x = sitk.ReadImage(x_path)
    x = sitk.GetArrayFromImage(x)
    x = torch.from_numpy(x).unsqueeze_(0).unsqueeze_(0)
    if flow_path.endswith('.npy'):
        flow = np.load(flow_path)
    elif flow_path.endswith('npz'):
        flow = np.load(flow_path)
        flow = flow['arr_0']
    else:
        print(f"{flow_path} is wrong")
        return None

    flow = torch.from_numpy(flow).unsqueeze_(0).float()
    H, W, D = 64, 256, 256
    config = CONFIGS_TM['TransMorph']
    config.img_size = (H, W, D)
    config.window_size = (H // 16, W // 32, D // 32)
    model = TransMorph.TransMorph(config)
    # debug my flow test
    out = model.spatial_trans(x, flow)
    out = np.squeeze(out.detach().cpu().numpy())
    sitk.WriteImage(sitk.GetImageFromArray(out),
                    save_path)
class TestCombine(unittest.TestCase):
    def test_print(self):
        print("Testing SplitAndTortue")
    def test_readNPZ(self):

        root = r"Z:\users\yq\MorphDatasets\model\TransMorph\0813\test_result"
        # 形变场数据大小
        channels, depth, height, width = 3, 64, 256, 256
        global_wh = 256 * 4 - 50 * 3
        overlap = 50

        # 假设 deformation_fields 是 4x4 的局部形变场矩阵
        deformation_fields = np.zeros((4, 4, channels, depth, height, width))  # 示例数据
        stdy_idx = 0
        for i in range(4):
            for j in range(4):
                # i ,j reverse
                flow = np.load(os.path.join(root, ('disp_{:04d}' + '.npz').format(stdy_idx)))
                flow = flow['arr_0']
                stdy_idx += 1
                deformation_fields[i,j,...] = flow
                # deformation_fields[j, i, ...] = flow

        return deformation_fields
    def test_demo(self):

        # 形变场数据大小
        channels, depth, height, width = 3, 64, 256, 256
        global_wh = 256 * 4 - 50 * 3
        overlap = 50

        # 假设 deformation_fields 是 4x4 的局部形变场矩阵
        deformation_fields = np.random.rand(4, 4, channels, depth, height, width)  # 示例数据

        # # 假设 deformation_fields 是 4x4 的局部形变场矩阵
        # deformation_fields = np.random.rand(4, 4, 256, 256, 64, 3)
        final_shape = (channels, depth, global_wh, global_wh)
        overlap = 50
        lambda_smooth = 0.1  # 平滑度权重

        # 初步整合形变场作为初始解
        def integrate_deformation_fields(deformation_fields, overlap, final_shape):
            final_deformation_field = np.zeros(final_shape)
            weight_matrix = np.ones(final_shape[:3])  # 简化: 初始化权重矩阵为1

            for i in range(deformation_fields.shape[0]):
                for j in range(deformation_fields.shape[1]):
                    start_x = i * (deformation_fields.shape[2] - overlap)
                    start_y = j * (deformation_fields.shape[3] - overlap)
                    end_x = start_x + deformation_fields.shape[2]
                    end_y = start_y + deformation_fields.shape[3]

                    final_deformation_field[start_x:end_x, start_y:end_y, :, :] += \
                        deformation_fields[i, j]
                    weight_matrix[start_x:end_x, start_y:end_y, :] += 1

            return final_deformation_field / weight_matrix[..., np.newaxis]

        initial_field = integrate_deformation_fields(deformation_fields, overlap, final_shape)

        # 定义能量函数
        def energy_function(u):
            u = u.reshape(final_shape)

            # 数据一致性能量 (data term)
            data_energy = 0
            for i in range(deformation_fields.shape[0]):
                for j in range(deformation_fields.shape[1]):
                    start_x = i * (deformation_fields.shape[2] - overlap)
                    start_y = j * (deformation_fields.shape[3] - overlap)
                    end_x = start_x + deformation_fields.shape[2]
                    end_y = start_y + deformation_fields.shape[3]

                    local_field = u[start_x:end_x, start_y:end_y, :, :]
                    data_energy += np.sum((local_field - deformation_fields[i, j]) ** 2)

            # 光滑度能量 (smoothness term)
            smooth_energy = np.sum(np.gradient(u, axis=(0, 1, 2)) ** 2)

            # 总能量
            total_energy = lambda_smooth * smooth_energy + (1 - lambda_smooth) * data_energy
            return total_energy

        # 优化
        result = minimize(energy_function, initial_field.ravel(), method='L-BFGS-B')

        # 获取优化后的形变场
        smoothed_field = result.x.reshape(final_shape)

        # smoothed_field 现在包含平滑过渡的全局形变场
    def test_init(self):
        import torch
        import numpy as np

        # 假设图1和图2的形变场分别为 field1 和 field2
        field1 = torch.tensor(np.random.rand(3, 64, 256, 256), dtype=torch.float32, requires_grad=False)  # 示例形变场
        field2 = torch.tensor(np.random.rand(3, 64, 256, 256), dtype=torch.float32, requires_grad=False)  # 示例形变场

        # 重叠区域大小
        overlap_size = 50

        # 提取重叠区域的形变场
        overlap_field1 = field1[:, :, :, -overlap_size:]  # 图1的右边50像素
        overlap_field2 = field2[:, :, :, :overlap_size]  # 图2的左边50像素

        # 初始的加权平均重叠区域
        def blend_overlap(region1, region2):
            weights = torch.linspace(0, 1, overlap_size)
            weight1 = weights.view(1, 1, 1, -1)
            weight2 = 1 - weight1
            blended = weight1 * region1 + weight2 * region2
            return blended

        # 得到初始平滑的重叠区域形变场
        smoothed_overlap_initial = blend_overlap(overlap_field1, overlap_field2)

        # 将初始平滑结果转为可优化的 PyTorch 参数
        smoothed_overlap = torch.tensor(smoothed_overlap_initial.clone().detach(), requires_grad=True)

        # 定义优化器
        optimizer = torch.optim.LBFGS([smoothed_overlap], lr=1, max_iter=20)

        # 定义损失函数，包括与非重叠区域的平滑性
        def closure():
            optimizer.zero_grad()

            # 与图1的平滑性
            transition_energy1 = torch.sum((smoothed_overlap[:, :, :, 0] - field1[:, :, :, -overlap_size - 1]) ** 2)
            smoothness_energy1 = torch.sum(torch.gradient(smoothed_overlap, dim=(1, 2, 3)) ** 2)

            # 与图2的平滑性
            transition_energy2 = torch.sum((smoothed_overlap[:, :, :, -1] - field2[:, :, :, 0]) ** 2)
            smoothness_energy2 = torch.sum(torch.gradient(smoothed_overlap, dim=(1, 2, 3)) ** 2)

            # 总损失
            loss = smoothness_energy1 + smoothness_energy2 + transition_energy1 + transition_energy2
            loss.backward()

            return loss

        # 优化重叠区域
        optimizer.step(closure)

        # 将平滑的重叠区域分别更新回图1和图2
        field1[:, :, :, -overlap_size:] = smoothed_overlap.detach()
        field2[:, :, :, :overlap_size] = smoothed_overlap.detach()

        # 输出或保存结果
        np.save('optimized_field1.npy', field1.detach().numpy())
        np.save('optimized_field2.npy', field2.detach().numpy())
        print("Optimized deformation fields saved.")
    def test_linerDF(self):
        import torch
        import numpy as np

        # 假设图1和图2的形变场分别为 field1 和 field2
        field1 = torch.tensor(np.random.rand(3, 64, 256, 256), dtype=torch.float32)  # 示例形变场
        field2 = torch.tensor(np.random.rand(3, 64, 256, 256), dtype=torch.float32)  # 示例形变场

        # 重叠区域大小
        overlap_size = 50

        # 提取重叠区域的形变场
        overlap_field1 = field1[:, :, :, -overlap_size:]  # 图1的右边50像素
        overlap_field2 = field2[:, :, :, :overlap_size]  # 图2的左边50像素

        # 定义过渡带
        def create_transition_band(overlap_size):
            # 生成从 0 到 1 的线性插值权重
            transition_band = torch.linspace(0, 1, overlap_size)
            return transition_band

        # 应用线性插值
        def apply_linear_interpolation(overlap_field1, overlap_field2, transition_band):
            weight1 = 1 - transition_band.view(1, 1, 1, -1)
            weight2 = transition_band.view(1, 1, 1, -1)
            interpolated_field = weight1 * overlap_field1 + weight2 * overlap_field2
            return interpolated_field

        # 生成过渡带权重
        transition_band = create_transition_band(overlap_size)

        # 对重叠区域进行线性插值
        smoothed_overlap = apply_linear_interpolation(overlap_field1, overlap_field2, transition_band)

        # 将平滑的重叠区域分别更新回图1和图2
        field1[:, :, :, -overlap_size:] = smoothed_overlap
        field2[:, :, :, :overlap_size] = smoothed_overlap

        # 输出或保存结果
        # np.save('optimized_field1.npy', field1.detach().numpy())
        # np.save('optimized_field2.npy', field2.detach().numpy())
        print("Optimized deformation fields saved.")
    def test_linerDF2(self):
        import torch
        import numpy as np
        saveRoot = r"Z:\users\yq\MorphDatasets\model\TransMorph\0813\test_result"

        # 假设 deformation_fields 是 4x4 的局部形变场矩阵，每个形变场大小为 (3, 64, 256, 256)
        # deformation_fields = torch.tensor(np.random.rand(4, 4, 3, 64, 256, 256), dtype=torch.float32)  # 示例数据

        deformation_fields = torch.tensor(self.test_readNPZ())
        # 重叠区域大小
        overlap_size = 50

        # 加权平均平滑函数
        def weighted_average_smoothing(deformation_fields, overlap_size):
            for i in range(4):
                for j in range(4):
                    if j < 3:  # 水平相邻小图
                        field1 = deformation_fields[i, j]
                        field2 = deformation_fields[i, j + 1]

                        # 提取重叠区域
                        # overlap_field1 = field1[:, :, :, -overlap_size:]  # 图1的右边50像素
                        # overlap_field2 = field2[:, :, :, :overlap_size]  # 图2的左边50像素
                        overlap_field1 = field1[:, :, -overlap_size:, :]  # 图1的下边50像素
                        overlap_field2 = field2[:, :, :overlap_size, :]  # 图2的上边50像素

                        # 生成线性权重
                        weights = torch.linspace(0, 1, overlap_size)
                        # weight1 = weights.view(1, 1, 1, -1)
                        weight1 = weights.view(1, 1, -1, 1)
                        weight2 = 1 - weight1

                        # 加权平均
                        smoothed_overlap = weight1 * overlap_field1 + weight2 * overlap_field2

                        # 更新原始形变场
                        # deformation_fields[i, j, :, :, :, -overlap_size:] = smoothed_overlap
                        # deformation_fields[i, j + 1, :, :, :, :overlap_size] = smoothed_overlap

                        deformation_fields[i, j, :, :, -overlap_size:, :] = smoothed_overlap
                        deformation_fields[i, j + 1, :, :, :overlap_size, :] = smoothed_overlap
                    if i < 3:  # 垂直相邻小图
                        field1 = deformation_fields[i, j]
                        field2 = deformation_fields[i + 1, j]

                        # 提取重叠区域
                        # overlap_field1 = field1[:, :, -overlap_size:, :]  # 图1的下边50像素
                        # overlap_field2 = field2[:, :, :overlap_size, :]  # 图2的上边50像素

                        overlap_field1 = field1[:, :, :, -overlap_size:]  # 图1的右边50像素
                        overlap_field2 = field2[:, :, :, :overlap_size]  # 图2的左边50像素
                        # 生成线性权重
                        weights = torch.linspace(0, 1, overlap_size)
                        # weight1 = weights.view(1, 1, -1, 1)
                        weight1 = weights.view(1, 1, 1, -1)
                        weight2 = 1 - weight1

                        # 加权平均
                        smoothed_overlap = weight1 * overlap_field1 + weight2 * overlap_field2

                        # 更新原始形变场
                        # deformation_fields[i, j, :, :, -overlap_size:, :] = smoothed_overlap
                        # deformation_fields[i + 1, j, :, :, :overlap_size, :] = smoothed_overlap
                        deformation_fields[i, j, :, :, :, -overlap_size:] = smoothed_overlap
                        deformation_fields[i + 1, j, :, :, :, :overlap_size] = smoothed_overlap

            return deformation_fields

        # 应用加权平均平滑方法
        smoothed_fields = weighted_average_smoothing(deformation_fields, overlap_size)

        # # 输出或保存优化后的形变场
        smoothed_fields_np = smoothed_fields.detach().numpy()
        for i in range(4):
            for j in range(4):
                np.save(os.path.join(saveRoot,f'optimized_field_{i}_{j}.npy'), smoothed_fields_np[i, j])
        print("Optimized deformation fields saved.")

    # todo df done
    def test_useDF(self):
        root = r"Z:\users\yq\MorphDatasets\model\TransMorph\0813\test_result"
        stdy_idx = 0
        for i in range(4):
            for j in range(4):
                # flow_path = os.path.join(root, ('disp_{:04d}' + '.npz').format(stdy_idx))
                flow_path = os.path.join(root, f'optimized_field_{i}_{j}.npy')
                x_path = os.path.join(root, ('x_{:04d}.tif').format(stdy_idx))
                save_path = os.path.join(root, ('refine_out_{:04d}' + '.tif').format(stdy_idx))
                stdy_idx += 1
                useDF(x_path, flow_path, save_path)
                print(f"{i} {j} finished")
                pass

    def test_combineFun(self,imgFormat,savePath):
        # os.path.join(root, ('refine_out_{:04d}' + '.tif'))
        # os.path.join(root,"refine_outall.mha")

        stdy_idx = 0
        overlap_size = 50
        imgAll = np.zeros((64,1000,1000),dtype=np.float32)
        for i in range(4):
            for j in range(4):
                # 计算每个小图在完整图像中的位置
                start_x = i * (256 - overlap_size)
                start_y = j * (256 - overlap_size)
                end_x = start_x + 256
                end_y = start_y + 256

                img_path = imgFormat.format(stdy_idx)
                stdy_idx += 1

                x = sitk.ReadImage(img_path)
                x = sitk.GetArrayFromImage(x)
                # 将当前小图的形变场添加到完整图像的对应位置
                imgAll[:,start_y:end_y , start_x:end_x] = x
                print(f"{i} {j} finished")

        sitk.WriteImage(sitk.GetImageFromArray(imgAll),savePath)

    def test_combineImg(self):
        root = r"Z:\users\yq\MorphDatasets\model\TransMorph\0813\test_result"
        # imgFormat, savePath
        # imgFormat = os.path.join(root, ('y_{:04d}' + '.tif'))
        # savePath = os.path.join(root,"y_out.mha")
        # self.test_combineFun(imgFormat, savePath)

        imgFormat = os.path.join(root, ('x_def_{:04d}' + '.tif'))
        savePath = os.path.join(root, "x_def_out.mha")
        self.test_combineFun(imgFormat, savePath)