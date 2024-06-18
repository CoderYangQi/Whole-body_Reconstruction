from VISoR_Reconstruction.reconstruction.yq_reconstruct import *
from VISoR_Brain.utils.elastix_files import *


def Preprocess(surface, threshold):
    # if img_path == None:
    #     return None
    # surface = sitk.ReadImage(img_path)
    # threshold = 120
    surface = sitk.Threshold(surface, threshold, 65535, threshold)
    back_log_value = np.log(threshold)
    # back_log_value = 0
    surface = sitk.Clamp((sitk.Log(sitk.Cast(surface + 1, sitk.sitkFloat32)) - back_log_value) * 39.4,
                         sitk.sitkFloat32, 0, 255)
    # surface = sitk.Clamp((sitk.Log(sitk.Cast(surface + 1, sitk.sitkFloat32)) - back_log_value) * 39.4,
    #                      sitk.sitkFloat32, 0, 255)
    # surface = sitk.Clamp((sitk.Log(sitk.Cast(surface + 1, sitk.sitkFloat32)) - back_log_value) * 39.4,
    #                      sitk.sitkUInt8, 0, 255)
    return surface

def _alignTask(prev_path,next_path,rate,ref_size,prev_points,next_points,save_prev_df,
               save_next_df,save_prev,save_next,save_prev_2,save_next_2):
    start = time.time()
    prev_surface = sitk.ReadImage(prev_path)[::rate, ::rate]

    next_surface = sitk.ReadImage(next_path)[::rate, ::rate]
    prev_surface = Preprocess(prev_surface, 120)
    next_surface = Preprocess(next_surface, 120)
    prev_surface.SetSpacing([1, 1])
    prev_surface.SetOrigin([0, 0])
    next_surface.SetSpacing([1, 1])
    next_surface.SetOrigin([0, 0])
    # d1, d2, _prev_surface, _next_surface = align_surfaces(prev_surface=prev_surface, next_surface=next_surface, method='yqRefine_elasitx',
    #                         ref_img=None,
    #                         outside_brightness=2, ref_scale=1, ref_size=ref_size, prev_points=prev_points,
    #                         next_points=next_points)
    d1, d2, _prev_surface, _next_surface = align_surfaces(prev_surface=prev_surface, next_surface=next_surface,
                                                          method='yqROI_0529',
                                                          ref_img=None,
                                                          outside_brightness=2, ref_scale=1, ref_size=ref_size,
                                                          prev_points=prev_points,
                                                          next_points=next_points)
    print(f"finished time is {time.time() - start}")
    # sitk.WriteImage(sitk.DisplacementFieldJacobianDeterminant(d1),
    #                 save_prev_df)
    # sitk.WriteImage(sitk.DisplacementFieldJacobianDeterminant(d2),
    #                 save_next_df)
    sitk.WriteImage(d1, save_prev_df)
    sitk.WriteImage(d2, save_next_df)
    t1 = sitk.DisplacementFieldTransform(sitk.Image(d1))
    t2 = sitk.DisplacementFieldTransform(sitk.Image(d2))
    out1 = sitk.Resample(prev_surface, d1, t1)
    out2 = sitk.Resample(next_surface, d2, t2)
    sitk.WriteImage(out1, save_prev)
    sitk.WriteImage(out2, save_next)
    sitk.WriteImage(_prev_surface, save_prev_2)
    sitk.WriteImage(_next_surface, save_next_2)
import multiprocessing
import time


def run_multiprocess(numsThread, taskParas):
    # todo use multiprocess
    pool = multiprocessing.Pool(numsThread)
    result = []
    for i in range(len(taskParas)):
        msg = 'hello %s' % i
        result.append(pool.apply_async(func=_alignTask, args=taskParas[i]))

    pool.close()
    pool.join()

    # for res in result:
    #     print('***:', res.get())  # get()函数得出每个返回结果的值

    print('All end--')
def main():
    save_root = r'D:\USERS\yq\TH2_Reconstruction\ROI_76_102\ROIReconstruction\th2_0528'
    prevFormat = r"D:\USERS\yq\TH2_Reconstruction\ROI_76_102\ROIReconstruction\th2_0528\2_1_1_{:03d}_561nm_10X_ls.mha"
    nextFormat = r"D:\USERS\yq\TH2_Reconstruction\ROI_76_102\ROIReconstruction\th2_0528\2_1_1_{:03d}_561nm_10X_us.mha"
    ls_name_format = "2_1_1_{:03d}_561nm_10X_ls"
    pointsFlag = True
    printMsg = True
    taskChunks = []
    for i in range(77 ,102):
        start = time.time()
        prev_index = i
        next_index = i + 1
        if pointsFlag:
            prev_points = r"D:\USERS\yq\TH2_Reconstruction\ROI_76_102\ROIReconstruction\Points\{}_lp.txt".format(prev_index)
            next_points = r"D:\USERS\yq\TH2_Reconstruction\ROI_76_102\ROIReconstruction\Points\{}_up.txt".format(next_index)
            if  os.path.exists(prev_points) and os.path.exists(next_points):
                print("point Align")
            else:
                prev_points = None;
                next_points = None
        else:
            prev_points = None; next_points = None
            # 77_720.tif
        prev_path = prevFormat.format(prev_index)
        next_path = nextFormat.format(next_index)
        save_prev = os.path.join(save_root, "{:03d}_ls_re.mha".format(prev_index))
        save_next = os.path.join(save_root, "{:03d}_us_re.mha".format(next_index))
        save_prev_2 = os.path.join(save_root, "2_{:03d}_ls_re.mha".format(prev_index))
        save_next_2 = os.path.join(save_root, "2_{:03d}_us_re.mha".format(next_index))

        save_prev_df = os.path.join(save_root, "2_1_1_{:03d}_561nm_10X_lxy.mha".format(prev_index))
        save_next_df = os.path.join(save_root, "2_1_1_{:03d}_561nm_10X_uxy.mha".format(next_index))
        if printMsg:
            print(f"prev_path is {prev_path}")
            print(f"next_path is {next_path}")
            print(f"prev_points is {prev_points}")
            print(f"next_points is {next_points}")
            print(f"save_prev is {save_prev}")
            print(f"save_next is {save_next}")
            print(f"save_prev_df is {save_prev_df}")
            print(f"save_next_df is {save_next_df}")

        rate = 1
        ref_size = [8000 // rate, 7200 // rate]
        tempChunk = (prev_path,next_path,rate,ref_size,prev_points,next_points,save_prev_df,
               save_next_df,save_prev,save_next,save_prev_2,save_next_2)
        taskChunks.append(tempChunk)
    num_threads = 12  # 设置线程数量
    run_multiprocess(num_threads, taskChunks)


if __name__ == '__main__':
    main()