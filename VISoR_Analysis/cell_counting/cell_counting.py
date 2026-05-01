from VISoR_Brain.positioning.visor_brain import VISoRBrain, VISoRSample
import SimpleITK as sitk
from VISoR_Brain.utils.ome_tiff import write_ome_tiff
from VISoR_Analysis.common.ilastik_cell_counter import run_ilastik
import os, multiprocessing
import numpy as np


def preprocess(image: sitk.Image):
    image = (sitk.Log(image) - 4.6) * 39.4
    image = sitk.Clamp(image, sitk.sitkUInt8, 0, 255)
    return image


def convert_image(path, output_path):
    image = sitk.ReadImage(path)
    write_ome_tiff(image, output_path)


def detect_spots(image):
    image.SetSpacing([1, 1, 1])
    image = sitk.Cast(image, sitk.sitkFloat32)
    image = sitk.AdditiveGaussianNoise(image, 0.001)
    #img_th = sitk.BinaryThreshold(image, 1.5, 255)
    d = sitk.DiscreteGaussian(image, 2)
    #d2 = sitk.DiscreteGaussian(image, 4)
    #dog = d1 - d2
    d_th = sitk.BinaryThreshold(d, 1.02, 255)
    #sitk.WriteImage(d, 'F:/chaoyu/test/z_.mha')
    #cd = sitk.GrayscaleDilate(dog, 2) - dog
    #cd = sitk.BinaryThreshold(cd, 0, 0.001)
    cd = sitk.RegionalMaxima(d)
    cd = sitk.Cast(cd, sitk.sitkUInt8)
    cd = sitk.And(cd, d_th)
    #cd = sitk.And(cd, img_th)
    #sitk.WriteImage(cd, 'F:/chaoyu/test/z.mha')
    cd = sitk.GetArrayFromImage(cd)
    #cd = cd[2:cd.shape[0] - 2, 2:cd.shape[1] - 2, 2:cd.shape[2] - 2]
    l = np.nonzero(cd)
    l = np.transpose(l)
    l = np.flip(l, 1)
    return l


def write_points(points, file):
    with open(file, 'w') as f:
        lines = []
        for i in range(len(points)):
            pos = points[i]
            line = '{0},{1},{2}\n'.format(pos[0], pos[1], pos[2])
            lines.append(line)
        f.writelines(lines)


def count_cells(path: str, output_dir, ilastik_file):
    file_name = os.path.basename(path)
    f1 = os.path.join(output_dir, file_name)
    if not os.path.exists(f1):
        convert_image(path, f1)
    f2 = os.path.join(output_dir, file_name.split('.')[0] + '_segmentation.tiff')
    if not os.path.exists(f2):
        run_ilastik(ilastik_file, f1, f2)
    f3 = os.path.join(output_dir, file_name.split('.')[0] + '_points.txt')
    if not os.path.exists(f3):
        image = sitk.ReadImage(f2)
        points = detect_spots(image)
        write_points(points, f3)


if __name__ == '__main__':
    input_dir = "F:/TEST_DATA/Mouse_Brain/20180914_ZMN_WH_438_1_1/Reconstruction/Slice/4"
    output_dir = "F:/TEST_DATA/Mouse_Brain/20180914_ZMN_WH_438_1_1/Anaylsis/CellCounting"
    ilastik_file = 'F:/chaoyu/test/thy1/cells.ilp'
    input_paths = []
    results = []

    with multiprocessing.Pool(processes=6) as pool:
        for f in os.listdir(input_dir):
            results.append(pool.apply_async(count_cells, args=[os.path.join(input_dir, f), output_dir, ilastik_file]))
        for r in results:
            r.get()