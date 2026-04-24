from VISoR_Brain.positioning.visor_brain import VISoRBrain, VISoRSample
from VISoR_Brain.misc import *
#from VISoR_Reconstruction.misc import PARAMETER_DIR
from VISoR_Brain.utils.elastix_files import *
import SimpleITK as sitk
import os, json, re, shutil
from math import ceil


def generate_freesia_input(input_path: str, shape, channel, pixel_size, group_size, **kwargs):
    doc = {
        "group_size": group_size,
        "image_path": os.path.split(input_path)[1],
        "images": [],
        "pixel_size": pixel_size,
        "slide_thickness": pixel_size,
        "version": "1.1.2"
    }
    h, w = shape[1], shape[0]

    files = {f for f in os.listdir(input_path) if re.match('Z\d+_C\d+.tif', f) is not None}

    ct = 0
    while ct < 100000 and len(files) > 0:
        name = 'Z{:05d}_C{}.tif'.format(ct, channel)
        if name not in files:
            ct += 1
            continue
        files.remove(name)
        d = {
            "index": ct,
            "height": h,
            "width": w,
            "file_name": name
        }
        doc['images'].append(d)
        ct += 1

    output_file = os.path.join(os.path.split(input_path)[0],
                               'freesia_{}_C{}.json'.format(os.path.split(input_path)[1], channel))
    #with open(output_file, 'w') as fp:
    #    json.dump(doc, fp, indent=2)
    return json.dumps(doc)


def write_freesia2_image(image: sitk.Image, path: str, name: str, pixel_size, group_size):
    doc = {
        "group_size": group_size,
        "image_path": name,
        "images": [],
        "voxel_size": pixel_size,
        #"slide_thickness": pixel_size
    }
    h, w = image.GetSize()[1], image.GetSize()[0]

    image_path = os.path.join(path, name)
    if not os.path.exists(image_path):
        os.mkdir(image_path)
    files = []
    for z in range(image.GetSize()[2]):
        file = '{:04d}_{}.tif'.format(z, pixel_size)
        files.append(os.path.join(image_path, file))
        d = {
            "index": z + 1,
            "height": h,
            "width": w,
            "file_name": file
        }
        doc['images'].append(d)
    sitk.WriteImage(image, files)

    doc['freesia_project'] = {}
    freesia_project = doc['freesia_project']
    freesia_project['transform_2d'] = []
    for i in range(int(ceil(image.GetSize()[2] / group_size))):
        freesia_project['transform_2d'].append({
            "group_index": i,
            "rotation": "0",
            "scale": "1 1",
            "translation": "0 0"})
    freesia_project["transform_3d"] = {
        "rotation": "0 0",
        "scale": "1 1 1",
        "translation": "0 0 0"}
    freesia_project['warp_markers'] = []

    output_file = os.path.join(path, '{}.json'.format(name))
    with open(output_file, 'w') as fp:
        json.dump(doc, fp, indent=4)


def read_freesia2_image(file: str):
    with open(file) as f:
        doc = json.load(f)
    files = []
    for i in doc['images']:
        files.append(os.path.join(os.path.dirname(file), doc['image_path'], i['file_name']))
    return sitk.ReadImage(files)


def register_brain(image_list_file:str, output_path: str, template_file: str, output_name:str='',
                   brain_transfrom_path: str=''):

    with open(template_file) as f:
        doc = json.load(f)
        template = sitk.ReadImage(os.path.join(ROOT_DIR, 'data', doc['file_name']))
        template_pixel_size = doc['voxel_size']
        atlas = None
        if 'atlas_file_name' in doc:
            atlas = sitk.ReadImage(os.path.join(ROOT_DIR, 'data', doc['atlas_file_name']))

    template.SetSpacing([1, 1, 1])
    input_file = os.path.join(output_path, 'Thumbnail_{}.json'.format(template_pixel_size))
    if os.path.exists(input_file):
        image = read_freesia2_image(input_file)
        #image = sitk.Clamp((sitk.Log(sitk.Cast(image, sitk.sitkFloat32)) - 4.6) * 39.4, sitk.sitkFloat32, 0, 255)
    else:
        brain_image_files = []
        with open(image_list_file) as f:
            doc_ = json.load(f)
            for i in doc_['images']:
                brain_image_files.append(os.path.join(os.path.dirname(image_list_file), doc_['image_path'], i['file_name']))
            pixel_size = doc_['pixel_size']
            group_size = doc_['group_size']
        image = []
        scale = pixel_size / template_pixel_size
        for f in brain_image_files:
            im = sitk.ReadImage(f)
            im.SetSpacing([pixel_size for i in range(2)])
            size = [int(im.GetSize()[i] * scale) for i in range(2)]
            im = sitk.Resample(im, size, sitk.Transform(), sitk.sitkLinear, [0, 0], [template_pixel_size for i in range(2)])
            image.append(im)
        image = sitk.JoinSeries(image)
        size = [image.GetSize()[0], image.GetSize()[1], int(image.GetSize()[2] * scale)]
        image.SetSpacing([template_pixel_size, template_pixel_size, pixel_size])
        image = sitk.Resample(image, size, sitk.Transform(), sitk.sitkLinear, [0, 0, 0], [template_pixel_size for i in range(3)])
        image.SetSpacing([1, 1, 1])
        write_freesia2_image(image, output_path, 'Thumbnail_{}'.format(template_pixel_size), template_pixel_size,
                             int(group_size * scale))
    image.SetSpacing([1, 1, 1])
    image = sitk.Clamp((sitk.Log(sitk.Cast(image, sitk.sitkFloat32)) - 4.6) * 39.4, sitk.sitkFloat32, 0, 255)
    out, tf, inv_tf = get_align_transform(image, template, [#os.path.join(PARAMETER_DIR, 'tp_brain_registration_rigid.txt'),
                                                            os.path.join(PARAMETER_DIR, 'tp_brain_registration_bspline.txt')],
                                          inverse_transform=True)
    sitk.WriteImage(out, os.path.join(output_path, 'registered.mha'))
    df = sitk.TransformToDisplacementField(tf, sitk.sitkVectorFloat32, template.GetSize())
    sitk.WriteImage(df, os.path.join(output_path, 'deformation_{}.mhd'.format(output_name)))
    df = sitk.TransformToDisplacementField(inv_tf, sitk.sitkVectorFloat32, image.GetSize())
    sitk.WriteImage(df, os.path.join(output_path, 'inverse_deformation_{}.mhd'.format(output_name)))
    if atlas is not None:
        atlas = sitk.Cast(atlas, sitk.sitkUInt16)
        atlas.SetSpacing([1, 1, 1])
        atlas = sitk.Resample(atlas, image, tf, sitk.sitkNearestNeighbor)
        atlas = sitk.Flip(atlas, [False, True, False])
        atlas_path = os.path.join(output_path, 'atlas')
        if not os.path.exists(atlas_path):
            os.mkdir(atlas_path)
        sitk.WriteImage(atlas, os.path.join(atlas_path, 'deformed_atlas_{}.mhd'.format(output_name)))
        if 'atlas' in doc:
            atlas_info = doc['atlas'][0]
            atlas_info['annotation_path'] = os.path.relpath(os.path.join(atlas_path, 'deformed_atlas_{}.raw'.format(output_name)),
                                                            os.path.dirname(input_file))
            atlas_info['image_dimension'] = '{} {} {}'.format(*atlas.GetSize())
            shutil.copy(os.path.join(ROOT_DIR, 'data', 'atlas_data', atlas_info['structures_path']), atlas_path)
            atlas_info['structures_path'] = os.path.relpath(os.path.join(atlas_path, atlas_info['structures_path']),
                                                            os.path.dirname(input_file))
            with open(input_file, 'r') as f:
                d = json.load(f)
            d['freesia_project']['atlas'] = atlas_info
            with open(input_file, 'w') as f:
                json.dump(d, f, indent=4)
            #atlas_info = {'atlas': [atlas_info]}
            #with open(os.path.join(atlas_path, 'freesia-atlas.json'), 'w') as f_:
            #    json.dump(atlas_info, f_, indent=4)

    if len(brain_transfrom_path) > 0:
        br = VISoRBrain()
        br.load(brain_transfrom_path)
        br.atlas_transform = tf
        br.save(brain_transfrom_path)


if __name__ == '__main__':
    register_brain('D:/VISoR12/Mouse/20180914_ZMN_WH_THY-CRE-8_1_1\\Reconstruction\\BrainTransform\\visor_brain.txt',
                   'D:/VISoR12/Mouse/20180914_ZMN_WH_THY-CRE-8_1_1\\Reconstruction\\BrainImage\\freesia_4.0_C1.json',
                   'D:/VISoR12/Mouse/20180914_ZMN_WH_THY-CRE-8_1_1\\Reconstruction/BrainRegistration',
                   'C:\\Users\\chaoyu\\Documents\\projects\\VISoR-data-analysis\\VISoR_Brain\\data\\ccf_v3_template.tif',
                   4.0, 25.0)
