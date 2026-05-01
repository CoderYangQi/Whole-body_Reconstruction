import json, re
import os, sys, multiprocessing
import numpy as np
import yaml, tifffile
import SimpleITK as sitk
from distutils.version import LooseVersion
from VISoR_Brain.lib import flsmio
import time, importlib


#import win32file

#win32file._setmaxstdio(2048)

class RawData:
    _rct = 0

    def __init__(self, data_file, device_file=None, real_path=None, overwrite_pixel_size=None):
        self.name = os.path.basename(data_file).split('.')[0]
        self.device_file = device_file
        self.info = {}
        self.path = real_path
        self.file = data_file
        self.columns = []
        self.overlap_columns = []
        self.column_label = []
        self.column_pos0 = []
        self.column_pos1 = []
        self.roi = [0, 0, 2048, 2048]
        self.thumbnail_roi = [0, 0, 512, 512]
        self.x_coefficient = 1000
        self.y_coefficient = 1000
        self.pixel_size = 1
        self.overwrite_pixel_size = overwrite_pixel_size
        self.column_spacing = []
        self.angle = -45 / 180 * np.pi
        self.pos0 = (0, 0, 0)
        self.pos1 = (0, 0, 0)
        self.image_files = {}
        self.scales = {}
        self.wave_length = 0
        self.caption = 'unknown'
        self.version = '0.0.0'
        self.z_index = -1
        self.size = [0, 0, 0, 0]
        self._load_func = self._load_pre24
        self._fastlsm_data(data_file, device_file)

        self._cached_thumb = None
        self._construct_count()

    @classmethod
    def _construct_count(cls):
        cls._rct += 1
        if cls._rct > 200:
            importlib.reload(flsmio)
            cls._rct = 0

    def _device(self, device_file):
        f = open(device_file)
        d = yaml.load(f)
        ch = d['channels'][0]
        self.x_coefficient = d['stage']['x_coefficient']
        self.y_coefficient = d['stage']['y_coefficient']
        self.roi = [ch['roi'][0], ch['roi'][1], ch['roi'][2], ch['roi'][3]]
        self.pixel_size =ch['pixel_size']
        self.angle = float(ch['angle']) / 180 * np.pi

    def _fastlsm_data(self, data_file, device_file=None):
        if self.path is None:
            self.path = os.path.dirname(data_file)
        f = open(data_file)
        flsm_info = json.load(f)
        self.info = flsm_info
        try:
            self.version = flsm_info['version']
        except:
            self.version = '1.0.0'

        # Fake raw data file
        if LooseVersion(self.version) == LooseVersion('0.0.0'):
            self._fake_rawdata(flsm_info)
            return

        if LooseVersion(self.version) < LooseVersion('2.0.0'):
            self._device(device_file)
            self._fastlsm_data_v1(flsm_info)
            return
        elif LooseVersion(self.version) < LooseVersion('2.3.1'):
            self._device(device_file)
            self._fastlsm_data_v2_pre24(flsm_info)
            return
        elif LooseVersion(self.version) < LooseVersion('2.4.0'):
            self._fastlsm_data_v2_pre24(flsm_info)
            return
        elif 'columns' in flsm_info:
            self._fastlsm_data_h5(data_file)
        else:
            self._fastlsm_data_v2(data_file)

    def _fastlsm_data_h5(self, flsm_file):
        import h5py
        self.angle = 45 / 180 * np.pi
        self._load_func = self._load_h5
        self.caption = self.info['caption']
        self.wave_length = self.info['wave_length']
        self.z_index = int(self.info['slices_index'])
        spacing = float(self.info["exposure"]) * float(self.info["velocity"])
        self.pixel_size = float(self.info["pixel_size"])
        if self.overwrite_pixel_size is not None:
            self.pixel_size = self.overwrite_pixel_size
        self.image_files['raw'] = h5py.File(os.path.join(self.path, self.info['all_images']['raw']['path']), 'r')
        self.image_files['thumbnail'] = h5py.File(os.path.join(self.path, self.info['all_images']['raw']['path']), 'r')
        self.scales['raw'] = 1
        self.scales['thumbnail'] = 4
        self.roi = [0, 0, int(self.info['columns'][0]['dimensions'].split(' ')[0]),
                    int(self.info['columns'][0]['dimensions'].split(' ')[1])]
        for col in self.info['columns']:
            self.columns.append(range(int(col['dimensions'].split(' ')[2])))
            self.column_spacing.append(spacing)
            origin = [float(i) for i in col['origin'].split(' ')]
            self.column_pos0.append([origin[2], origin[0], origin[1]])

    def _fastlsm_data_v2(self, flsm_file):

        self.angle = 45 / 180 * np.pi
        self._load_func = self._load

        reader = flsmio.FlsmReader(flsm_file)
        self.caption = reader.value('caption')
        self.wave_length = reader.value('wave_length')
        self.z_index = int(reader.value('slices_index'))
        spacing = float(reader.value("exposure")) * float(reader.value("velocity"))
        self.pixel_size = float(reader.value("pixel_size"))
        if self.overwrite_pixel_size is not None:
            self.pixel_size = self.overwrite_pixel_size
        self.image_files['raw'] = reader
        self.image_files['thumbnail'] = reader
        all_size = reader.size()
        n_stacks = int(all_size[0])
        n_images = int(all_size[1])
        width = int(all_size[2])
        height = int(all_size[3])
        thumbnail_width = int(all_size[4])
        thumbnail_height = int(all_size[5])
        self.size = [n_stacks, width, height, n_images]
        self.scales['raw'] = 1
        self.scales['thumbnail'] = width / thumbnail_width
        self.image_files['raw'] = None
        self.image_files['thumbnail'] = None
        self.roi = [0, 0, width, height]
        self.thumbnail_roi = [0, 0, thumbnail_width, thumbnail_height]
        for i in range(n_stacks):
            self.column_spacing.append(spacing)
            #pos = None
            j = n_images // 2
            #for j in range(1, n_images):
            #    image = reader.thumbnail(i, j)
            #    if image is not None:
            #        pos = image.position()
            #        break
            image = reader.thumbnail(i, j)
            pos = image.position()
            pos0 = [pos[0] - j * spacing, pos[1], 0]
            pos1 = [pos[0] + n_images * spacing,
                    pos[1] + self.pixel_size * width,
                    np.cos(self.angle) * self.pixel_size * height]
            pos0, pos1 = [min(pos0[i], pos1[i]) for i in range(3)], [max(pos0[i], pos1[i]) for i in range(3)]
            self.column_pos0.append(pos0)
            self.column_pos1.append(pos1)
            self.columns.append(range(n_images))
        self.pos0 = [min(self.column_pos0, key=lambda x: x[i])[i] for i in range(3)]
        self.pos1 = [max(self.column_pos1, key=lambda x: x[i])[i] for i in range(3)]

    def _fake_rawdata(self, flsm_info):
        self._load_func = lambda : exec('raise NotImplementedError(''Fake raw data have no image data'')')
        self.info = flsm_info.copy()
        self.caption = flsm_info['caption']
        self.z_index = int(flsm_info['slices_index'])
        self.wave_length = flsm_info['wave_length']
        self.pixel_size = float(flsm_info['pixel_size'])

    def _fastlsm_data_v2_pre24(self, flsm_info):
        self._load_func = self._load_pre24
        self.info = flsm_info.copy()
        self.info['slices_index'] = int(flsm_info['slices_index'])
        self.info['slides_index'] = int(flsm_info['slides_index'])
        if LooseVersion(self.version) < LooseVersion('2.4.0'):
            self.z_index = self.info['slides_index'] + 8 * self.info['slices_index'] - 8
        else:
            self.z_index = int(self.info['slices_index'])
        if 'wave_length' in self.info:
            self.wave_length = self.info['wave_length']
        spacing = float(flsm_info["exposure"]) * float(flsm_info["velocity"])
        label_alias = {'i':'i', 'index':'index', 'j':'j', 'x':'x', 'y':'y'}
        self.pixel_size = float(flsm_info['pixel_size'])
        if LooseVersion(self.version) < LooseVersion('2.3.1'):
            if LooseVersion(self.version) < LooseVersion('2.2.0'):
                image_file_name = os.path.join(self.path, 'origin.ome.tiff')
            else:
                image_file_name = os.path.join(self.path, 'Raw.ome.tif')
            self.image_files['raw'] = tifffile.TiffFile(image_file_name)
            self.scales['raw'] = 1
            self.image_files['overlap'] = tifffile.TiffFile(os.path.join(self.path, 'result', 'Overlap.ome.tif'))
            self.scales['overlap'] = 1
            self.image_files['thumbnail'] = tifffile.TiffFile(os.path.join(self.path, 'result', 'Thumbnail.ome.tif'))
            self.scales['thumbnail'] = 1
            ome = self.image_files['raw'].ome_metadata
            self.image_position = ome['Image']['Pixels']['TiffData']
            label_alias = {'i':'FirstZ', 'index':'IFD', 'j':'FirstT', 'x':'X', 'y':'Y'}

        else: # version >= 2.3.1
            self.roi[3] *= float(self.info['image_height'])
            for k, v in self.info['all_images'].items():
                image_file_name = os.path.join(self.path, self.info['all_images'][k]['path'])
                if not os.path.isfile(image_file_name):
                    continue
                if image_file_name.split('.')[-1] == 'tif' or image_file_name.split('.')[-1] == 'tiff':
                        self.image_files[k] = tifffile.TiffFile(image_file_name)
                self.scales[k] = float(self.info['all_images'][k]['factor'])
            with open(os.path.join(self.path, self.info['positions'])) as positions:
                self.image_position = json.load(positions)

        for v in self.image_position:
            m = v[label_alias['i']]
            while len(self.columns) < m + 1:
                self.columns.append([])
            self.columns[m].append(v)
        for j in range(len(self.columns)):
            s = self.columns[j]
            s.sort(key=lambda x : x[label_alias['x']])
            self.column_spacing.append(-spacing)
            offset = 0.
            xp = [s[i][label_alias['x']] * self.x_coefficient for i in range(50, len(s) - 50, 10)]
            xp_ = [xp[i] - i * self.column_spacing[j] * 10 for i in range(0, len(xp))]
            for i in range(3):
                offset = np.average(xp_)
                div_ = np.std(xp_)
                xp__ = []
                for x in xp_:
                    if abs(x - offset) < div_:
                        xp__.append(x)
                xp_ = xp__
            #if self.stack_spacing[j] < 0:
            #    offset += 40
            pos0 = (offset - 50 * self.column_spacing[j],
                    s[50][label_alias['y']] * self.y_coefficient + self.roi[0] * self.pixel_size,
                    np.cos(self.angle) * self.pixel_size * self.roi[1])
            pos1 = (offset + (len(s) - 50) * self.column_spacing[j],
                    s[50][label_alias['y']] * self.y_coefficient + (self.roi[0] + self.roi[2]) * self.pixel_size,
                    np.cos(self.angle) * self.pixel_size * (self.roi[1] + self.roi[3]))
            pos0, pos1 = (min(pos0[0], pos1[0]), min(pos0[1], pos1[1]), min(pos0[2], pos1[2])),\
                         (max(pos0[0], pos1[0]), max(pos0[1], pos1[1]), max(pos0[2], pos1[2]))
            self.column_pos0.append(pos0)
            self.column_pos1.append(pos1)
            self.columns[j] = [x[label_alias['index']] for x in s]
        self.pos0 = [min(self.column_pos0, key=lambda x: x[0])[0],
                     min(self.column_pos0, key=lambda x: x[1])[1],
                     min(self.column_pos0, key=lambda x: x[2])[2]]
        self.pos1 = [max(self.column_pos1, key=lambda x: x[0])[0],
                     max(self.column_pos1, key=lambda x: x[1])[1],
                     max(self.column_pos1, key=lambda x: x[2])[2]]

        try:
            self.image_files['overlap']
        except:
            return
        ome = self.image_files['overlap'].ome_metadata
        for v in ome['Image']['Pixels']['TiffData']:
            m = v['FirstZ']
            while len(self.overlap_columns) < m + 1:
                self.overlap_columns.append([])
            self.overlap_columns[m].append(v)
        for j in range(len(self.overlap_columns)):
            s = self.overlap_columns[j]
            s.sort(key=lambda x: x['FirstT'])
            self.overlap_columns[j] = [x['IFD'] for x in s]

    def _fastlsm_data_v1(self, flsm_info):
        image_info = flsm_info["images"]["motor"]
        spacing = float(flsm_info["exposure"]) * float(flsm_info["speed"])
        try:
            self.pixel_size = float(flsm_info['pixel_size'])
        except:
            pass
        image_height = 1.0
        try:
            image_height = float(flsm_info['imageHeight'])
        except:
            pass
        try:
            self.caption = flsm_info['caption']
        except:
            pass
        try:
            self.info['slices_index'] = int(flsm_info['slices_index'])
            self.info['slides_index'] = int(flsm_info['slides_index'])
            self.z_index = self.info['slides_index'] + 8 * self.info['slices_index'] - 8
        except:
            try:
                cap = re.search('_\\d+_\\d+$', self.caption)
                self.info['slices_index'] = int(cap.group(0).split('_')[-1])
                self.info['slides_index'] = int(cap.group(0).split('_')[-2])
                self.z_index = self.info['slides_index'] + 8 * self.info['slices_index'] - 8
                self.caption = self.caption[:-len(cap.group(0))]
            except:
                self.z_index = 0
        self.roi[3] *= image_height
        for idx, v in image_info.items():
            idx = idx.split('_')
            m = int(idx[0])
            n = int(idx[1])
            while len(self.columns) < m + 1:
                self.columns.append([])
                self.column_label.append(v["filename"].split('_')[0].split('/')[1])
            self.columns[m].append([os.path.join(self.path, v["filename"]),
                                    float(v['position_x']) * self.x_coefficient,
                                    float(v['position_y']) * self.y_coefficient,
                                    n])
        for j in range(len(self.columns)):
            s = self.columns[j]
            s.sort(key=lambda x : x[3])
            if s[0][1] < s[1][1]:
                self.column_spacing.append(spacing)
            else:
                self.column_spacing.append(-spacing)
            offset = 0.
            xp = [s[i][1] for i in range(50, len(s) - 50, 10)]
            xp_ = [xp[i] - i * self.column_spacing[j] * 10 for i in range(0, len(xp))]
            for i in range(3):
                offset = np.average(xp_)
                div_ = np.std(xp_)
                xp__ = []
                for x in xp_:
                    if abs(x - offset) < div_:
                        xp__.append(x)
                xp_ = xp__
            if self.column_spacing[j] < 0:
                offset += 40
            pos0 = (offset - 50 * self.column_spacing[j],
                    s[50][2] + self.roi[0] * self.pixel_size,
                    np.cos(self.angle) * self.pixel_size * self.roi[1])
            pos1 = (offset + (len(s) - 50) * self.column_spacing[j],
                    s[50][2] + (self.roi[0] + self.roi[2]) * self.pixel_size,
                    np.cos(self.angle) * self.pixel_size * (self.roi[1] + self.roi[3]))
            pos0, pos1 = (min(pos0[0], pos1[0]), min(pos0[1], pos1[1]), min(pos0[2], pos1[2])),\
                         (max(pos0[0], pos1[0]), max(pos0[1], pos1[1]), max(pos0[2], pos1[2]))
            self.column_pos0.append(pos0)
            self.column_pos1.append(pos1)
            self.columns[j] = [x[0] for x in s]
        self.pos0 = [min(self.column_pos0, key=lambda x : x[0])[0],
                     min(self.column_pos0, key=lambda x : x[1])[1],
                     min(self.column_pos0, key=lambda x : x[2])[2]]
        self.pos1 = [max(self.column_pos1, key=lambda x : x[0])[0],
                     max(self.column_pos1, key=lambda x : x[1])[1],
                     max(self.column_pos1, key=lambda x : x[2])[2]]

    def release(self):
        if self._load_func == self._load:
            #self.image_files['raw'].release()
            del self.image_files['raw']
            self.image_files['raw'] = None

    def load(self, idx, range_=None, source_type='auto', **kwargs):
        if range_ == None:
            range_ = (0, len(self.columns[idx]))
        if isinstance(range_, int):
            range_ = [range_, range_ + 1]
        return self._load_func(idx, range_, source_type, **kwargs)

    def _load_h5(self, idx, range_=None, source_type='auto', output_format='sitk', roi=None):
        print(idx, range_)
        if idx == 0:
            idx = ''
        else:
            idx = str(idx)
        source = self.image_files['raw']['DataSet{}/ResolutionLevel 0/TimePoint 0/Channel 0/Data'.format(idx)]
        if source_type == 'thumbnail':
            source = self.image_files['thumbnail']['DataSet{}/ResolutionLevel 2/TimePoint 0/Channel 0/Data'.format(idx)]
        image = source[range_[0]:range_[1]]
        print(image.shape)
        if output_format == 'sitk':
            image = sitk.GetImageFromArray(image)
        return image


    def _load(self, idx, range_=None, source_type='auto', output_format='sitk'):
        if self.image_files['raw'] is None:
            reader = flsmio.FlsmReader(self.file)
            self.image_files['raw'] = reader
            self.image_files['thumbnail'] = reader
        #t1 = time.time()
        size = [self.roi[2], self.roi[3]]
        image_func = self.image_files['raw'].raw
        if source_type == 'thumbnail':
            size = [self.thumbnail_roi[2], self.thumbnail_roi[3]]
            image_func = self.image_files['raw'].thumbnail

        if output_format == 'sitk':
            images = []
            for i in range(0, range_[1] - range_[0]):
                image = image_func(idx, i + range_[0])
                if image is None:
                    image = sitk.Image(size, sitk.sitkUInt16)
                else:
                    image.decode()
                    image = np.array(image, copy=False)
                    image = sitk.GetImageFromArray(image)
                images.append(image)
            images = sitk.JoinSeries(images)
            if source_type == 'thumbnail':
                images.SetSpacing([self.scales[source_type], self.scales[source_type], 1])
        else:
            images = np.zeros((range_[1] - range_[0], size[1], size[0]), np.uint16)
            for i in range(0, range_[1] - range_[0]):
                image = image_func(idx, i + range_[0])
                if image is None:
                    continue
                image.decode()
                image = np.array(image, copy=False)
                np.copyto(images[i], image)
        return images

    def _load_pre24(self, idx, range_=None, source_type='auto'):
        if source_type == 'thumbnail':
            try:
                self.image_files['thumbnail']
            except:
                print('Cannot find thumbnail file, use raw')
                source_type = 'auto'
        overlap_side = 0
        if source_type == 'overlap_l':
            source_type = 'overlap'
        if source_type == 'overlap_r':
            overlap_side = 1
            source_type = 'overlap'

        if source_type == 'overlap':
            try:
                self.image_files['overlap']
            except:
                print('Cannot find overlap file, use raw')
                source_type = 'auto'

        if source_type == 'auto':
            if len(self.image_files) > 0:
                source_type = 'raw'
            elif os.path.isfile(self.columns[idx][100]):
                source_type = 'image_sequence'
            elif os.path.isfile(os.path.join(self.path, 'compressed', idx + '.mp4')):
                source_type = 'video'
            else:
                raise Exception('Cannot find raw data image file.')

        if source_type == 'image_sequence':
            r = sitk.ImageSeriesReader()
            file_list = self.columns[idx][range_[0]:range_[1]]
            assert len(file_list) > 0
            r.SetFileNames(file_list)

            with open('NUL', 'w') as devnull: # Trick for suppressing libtiff warning
                stderr = os.dup(sys.stderr.fileno())
                os.dup2(devnull.fileno(), sys.stderr.fileno())

                img = r.Execute()

                os.dup2(stderr, sys.stderr.fileno())

            img.SetSpacing([1, 1, 1])
            return img

        if source_type == 'thumbnail' and self._cached_thumb is not None:
            scale = self.scales[source_type]
            ifd_list = [self.columns[idx][i] for i in range(range_[0], range_[1])]
            img = [self._cached_thumb[i] for i in ifd_list]
            img = np.array(img)
            img = sitk.GetImageFromArray(img)
            img.SetSpacing([scale, scale, 1])
            return img

        if source_type == 'raw' or source_type == 'thumbnail' or source_type == 'overlap':
            source = self.image_files[source_type]
            scale = self.scales[source_type]
            stacks = self.columns
            if source_type == 'overlap':
                idx = idx * 2 + overlap_side
                stacks = self.overlap_columns
            ifd_list = [stacks[idx][i] for i in range(range_[0], range_[1])]
            img = read_ome_tiff(source, ifd_list)
            img = sitk.JoinSeries(img)
            img.SetSpacing([scale, scale, 1])
            if source_type == 'overlap' and overlap_side == 1:
                img.SetOrigin([float(self.roi[2] - img.GetSize()[0]), 0, 0])
            return img

        if source_type == 'video':
            from VISoR_Brain.video_decoder import VideoDecoder
            img = np.zeros([len(self.columns[idx]), int(self.roi[3]), int(self.roi[2])], np.uint16)
            vc = VideoDecoder()
            vc.open(os.path.join(self.path, 'compressed', self.column_label[idx] + '.mp4'))
            if vc.isOpened():
                img_list = open(os.path.join(self.path, 'compressed', self.column_label[idx] + '.txt'))
                img_list = [f[:-1] for f in img_list.readlines()]
                vpos = 0
                for i in range(len(self.columns[idx])):
                    if os.path.basename(self.columns[idx][i]) == img_list[vpos]:
                        frame = vc.read()
                        if frame is None:
                            break
                        vpos += 1
                    else:
                        frame = np.zeros((int(self.roi[3]), int(self.roi[2])), np.uint16)
                    np.copyto(img[i,:,:], frame[:int(self.roi[3]), :int(self.roi[2])])
                vc.close()
                return sitk.GetImageFromArray(img)

    def cache(self):
        self._cached_thumb = self.image_files['thumbnail'].asarray(maxworkers=12)
        shape = self._cached_thumb.shape
        self._cached_thumb = np.transpose(self._cached_thumb, (1, 0, 2, 3))
        self._cached_thumb = np.reshape(self._cached_thumb, (shape[0] * shape[1], shape[2], shape[3]))

    def delete_cache(self):
        self._cached_thumb = None

    def __getitem__(self, item):
        assert isinstance(item, int)
        return RawDataStack(item, self)


def read_ome_tiff(source: tifffile.TiffFile, ifd_list):
    #imgs = {}
    import asyncio
    async def read_page(ifd):
        img = sitk.GetImageFromArray(source.pages[ifd].asarray())
        return ifd, img
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    tasks = []
    for ifd in ifd_list:
        tasks.append(asyncio.ensure_future(read_page(ifd)))
    loop.run_until_complete(asyncio.wait(tasks))
    imgs = {t.result()[0]:t.result()[1] for t in tasks}
    imgs = [imgs[ifd] for ifd in ifd_list]
    return imgs



class RawDataStack:
    def __init__(self, index, parent:RawData):
        self.index = index
        self.parent = parent
        self.source_type = 'auto'

    def __getitem__(self, item):
        if isinstance(item, slice):
            return self.parent.load(self.index, [item.start, item.stop], source_type=self.source_type)


def load_raw_data(flsm_file):
    raw_data = RawData(flsm_file)
    return raw_data
