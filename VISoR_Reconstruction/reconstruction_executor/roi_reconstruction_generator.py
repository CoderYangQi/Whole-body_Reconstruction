from .executor import *
from .generator import create_task, create_target
from VISoR_Brain.format.visor_data import VISoRData
from VISoR_Brain.positioning.visor_brain import VISoRBrain
import os, math
import numpy as np
from VISoR_Reconstruction.misc import VERSION


default_param = {
  'name': 'ROIReconstruction',
  "slice_source": "raw",
  "pixel_size": 1.0,
  "exclude_channel": None,
  "exclude_slice": None,
  "roi": None,
  "output_path": None,
  "blockwise_reconstruction": True,
  "block_size": 5000,
  "block_generation_mask": None
}


def gen_roi_reconstruction_pipeline(dataset: VISoRData, **param):
    param = {**default_param, **param}
    all_targets = {}
    tasks = {}

    def _create_target(*args, **kwargs):
        t = create_target(*args, **kwargs)
        all_targets[t['name']] = t
        return t

    _create_target('null', 'null', None)

    def _create_task(*args):
        t = create_task(*args)
        tasks[t['name']] = t
        return t

    brain_transform = VISoRBrain(os.path.join(dataset.path, 'Reconstruction/BrainTransform/visor_brain.txt'))
    roi = param['roi']
    pixel_size = param['pixel_size']
    if roi is None:
        roi = brain_transform.sphere
    roi[1][2] -= 0.5 * pixel_size
    slice_range = (brain_transform.get_slice_position(roi[0])[0],
                   brain_transform.get_slice_position([i - 0.001 for i in roi[1]])[0] + 1)
    roi[1][2] += 0.5 * pixel_size
    ignore_slice = {}
    ignore_channel = {}
    if param['exclude_slice'] is not None:
        ignore_slice = {int(s) for s in param['exclude_slice'].split(',')}
    if param['exclude_channel'] is not None:
        ignore_channel = {s for s in param['exclude_channel'].split(',')}
    channels = {c: v for c, v in dataset.channels.items() 
                if v['ChannelName'] not in ignore_channel and v['LaserWavelength'] not in ignore_channel}
    slices = {i for i in range(slice_range[0], slice_range[1]) if i not in ignore_slice}
    output_path = param['output_path']
    if output_path is None:
        output_path = os.path.join(dataset.path, 'Analysis')
    result_path = os.path.join(output_path, param['name'])

    if param['blockwise_reconstruction']:
        block_size = param['block_size']
        block_roi_list = {(x, y, i): [[roi[0][0] + x * block_size, roi[0][1] + y * block_size, roi[0][2]],
                           [min(roi[0][0] + (x + 1) * block_size, roi[1][0]),
                            min(roi[0][1] + (y + 1) * block_size, roi[1][1]),
                            roi[1][2]]]
                          for i in slices
                          for y in range(int(np.ceil((roi[1][1] - roi[0][1]) / block_size)))
                          for x in range(int(np.ceil((roi[1][0] - roi[0][0]) / block_size)))}
        if param['block_generation_mask'] is not None:
            block_generation_mask = sitk.ReadImage(param['block_generation_mask'])
            block_generation_mask = sitk.GetArrayFromImage(block_generation_mask)
            block_roi_list = {k: v for k, v in block_roi_list.items() if block_generation_mask[k[2] - 1, k[1], k[0]] != 0}
    else:
        block_roi_list = {(0, 0, i): roi for i in slices}

    slice_roi = {}
    slice_image_roi = {}
    for k in block_roi_list:
        x, y, i = k
        if i not in brain_transform.slice_spheres:
            continue
        transform = brain_transform.transform(i)
        df = transform.GetDisplacementField()
        spacing = df.GetSpacing()
        block_roi = block_roi_list[k]
        slice_roi[(i, x, y)] = \
            [[max(brain_transform.slice_spheres[i][0][j], block_roi[0][j]) for j in range(3)],
             [min(brain_transform.slice_spheres[i][1][j], block_roi[1][j]) for j in range(3)]]
        transform_roi = [[math.floor((block_roi[0][j])/ spacing[j]) for j in range(2)],
                         [math.ceil((block_roi[1][j]) / spacing[j]) for j in range(2)]]
        transform_roi[0].append(0)
        transform_roi[1].append(2)
        mesh = np.meshgrid(*[np.linspace(transform_roi[0][j] * spacing[j] + df.GetOrigin()[j],
                                         (transform_roi[1][j] - 1) * spacing[j] + df.GetOrigin()[j],
                                         transform_roi[1][j] - transform_roi[0][j]) for j in range(2, -1, -1)],
                           indexing='ij')
        df_ = sitk.GetArrayFromImage(df[transform_roi[0][0]:transform_roi[1][0],
                                     transform_roi[0][1]:transform_roi[1][1]])
        slice_image_roi[(i, x, y)] = \
            [[np.min(df_[:, :, :, j] + mesh[2 - j]) - 5 * pixel_size for j in range(3)],
             [np.max(df_[:, :, :, j] + mesh[2 - j]) + 5 * pixel_size for j in range(3)]]

    for c in channels:
        for k in block_roi_list:
            x, y, i = k
            if i not in dataset.acquisition_results[c]:
                continue
            channel_name = dataset.channels[c]['ChannelName']
            slice_name = '{}_{:03d}_{}_X{}_Y{}'.format(dataset.name, i, channel_name, x, y)

            t_rawdata = _create_target('raw_data_{}'.format(slice_name), 'raw_data', dataset.acquisition_results[c][i])
            t_slice = _create_target('slice_{}'.format(slice_name), 'reconstructed_slice', dataset.slice_transform[c][i])
            t_slice_image = _create_target('slice_image_{}'.format(slice_name), 'ome_tiff',
                                           os.path.join(result_path, 'SliceROIImage/' + str(pixel_size),
                                                        slice_name + '.tif'),
                                           ['SliceROIImage', 'SliceROIImage'],
                                           {"SliceID": i, "ChannelName": channel_name, "PixelSize": pixel_size})

            _create_task('reconstruct_image', slice_name,
                         {'pixel_size': pixel_size, 'source': param['slice_source'], 'method': 'gpu_resample',
                          'roi': slice_image_roi[(i, x, y)]},
                         {'sample_data': t_slice, 'rawdata': t_rawdata}, [t_slice_image])

    t_brain = _create_target('brain_transfrom', 'reconstructed_brain', dataset.brain_transform)

    for c in channels:
        channel_name = dataset.channels[c]['ChannelName']
        for i in slices:
            if i not in dataset.acquisition_results[c]:
                continue
            tile_image_list = {}
            n_start = int(max(brain_transform.slice_spheres[i][0][2], roi[0][2]) / pixel_size)
            for k in block_roi_list:
                if k[2] != i:
                    continue

                slice_name = '{}_{:03d}_{}_X{}_Y{}'.format(dataset.name, i, channel_name, k[0], k[1])

                path = os.path.join(result_path, 'ROIBlockImage', str(pixel_size), 'X{}_Y{}'.format(k[0], k[1]))
                category = ['ROIBlockImage', 'ROIBlockImage']
                if len(block_roi_list) == 1:
                    path = os.path.join(result_path, 'ROIImage', str(pixel_size))
                    category = ['ROIImage', 'ROIImage']
                name_format = os.path.join(path, 'Z{:05d}_' + 'C{}.tif'.format(c))
                t_image_list = _create_target('d_{}'.format(slice_name), 'file',
                                              os.path.join(path, slice_name + '.txt'),
                                              category,
                                              {"SliceID": i, "ChannelName": channel_name, "PixelSize": pixel_size})
                tile_image_list['{},{}'.format(k[0], k[1])] = t_image_list
                _create_task('generate_brain_image', slice_name,
                             {'slice_index': i, 'input_pixel_size': pixel_size, 'name_format': name_format,
                              'output_pixel_size': pixel_size, 'n_start': n_start, 'roi': slice_roi[(i, k[0], k[1])],
                              'slice_origin': slice_image_roi[(i, k[0], k[1])][0]},
                             {'brain': t_brain, 'img': all_targets['slice_image_{}'.format(slice_name)]}, [t_image_list])

            if len(block_roi_list) > 1:
                slice_name = '{}_{:03d}_{}'.format(dataset.name, i, channel_name)
                path = os.path.join(result_path, 'ROIImage', str(pixel_size))
                t_image_list = _create_target('d_{}'.format(slice_name), 'file',
                                         os.path.join(path, slice_name + '.txt'),
                                         ['ROIImage', 'ROIImage'],
                                         {"SliceID": i, "ChannelName": channel_name, "PixelSize": pixel_size})
                name_format = os.path.join(path, 'Z{:05d}_' + 'C{}.tif'.format(c))
                _create_task('tile_images', slice_name,
                             {'name_format': name_format,
                              'width': int((roi[1][0] - roi[0][0]) / pixel_size),
                              'height': int((roi[1][1] - roi[0][1]) / pixel_size),
                              'n_end': int(min(roi[1][2], brain_transform.slice_spheres[i][1][2]) / pixel_size),
                              'n_start': n_start,
                              'block_size': int(param['block_size'] / pixel_size)},
                             tile_image_list, [t_image_list])
    metadata = {
        "ROIImage": {"ROIImageInfo": {
            "PixelSize": pixel_size,
            "Type": "Projection",
            "Software": "VISOR_Reconstruction",
            "Parameter": "../Parameters.json",
            "Version": VERSION,
            "Time": time.asctime(),
            "Transform": os.path.join(dataset.path, 'Reconstruction/BrainTransform')
        }}
    }
    doc = {'tasks': tasks, 'name': param['name'],
           'path': output_path,
           'parameters': param,
           'metadata': metadata,
           'raw_data_info': dataset.to_dict(output_path)}
    return json.dumps(doc, indent=2)
