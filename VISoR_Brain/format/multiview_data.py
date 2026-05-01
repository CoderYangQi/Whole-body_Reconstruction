import json
from VISoR_Brain.format.visor_data import VISoRData
import SimpleITK as sitk
import os


def read_multiview_transform(path):
    with open(path, 'r') as f:
        metadata = json.load(f)
    view_transforms = {i: {a: sitk.ReadTransform(os.path.join(os.path.dirname(path), s[a])) for a in s}
                       for i, s in metadata['transforms'].items()}
    views = {a: VISoRData(v) for a, v in metadata['views'].items()}
    return views, view_transforms


if __name__ == '__main__':
    views, view_transforms = read_multiview_transform(r'X:\multiview_reconstruction\20201028_SY_USTC_CFOS-FS_296_1\transform_local.json')
    print(views, view_transforms)
    import tifffile
