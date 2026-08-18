import SimpleITK as sitk


def read():
    path1 = r"Z:\users\yq\MorphDatasets\Bspine\0203\out_0000-3.tif"
    path2 = r"Z:\users\yq\MorphDatasets\Bspine\0203\out_0000-4.tif"
    img1 = sitk.ReadImage(path1)
    img2 = sitk.ReadImage(path2)
    sitk.WriteImage(img1[:,:,0], "img1.png")
    sitk.WriteImage(img2[:,:,0], "img2.png")
if __name__ == '__main__':
    read()