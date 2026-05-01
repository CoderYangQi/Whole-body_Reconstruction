import subprocess as sp
import numpy as np
import json
import queue
from threading import Thread
import torch

FFMPEG_EXEC = 'C:/Users/chaoyu/Documents/ffmpeg/bin/ffmpeg.exe'
FFPROBE_EXEC = 'C:/Users/chaoyu/Documents/ffmpeg/bin/ffprobe.exe'

class VideoDecoder:
    shape = (0, 0)
    size = 0
    is_opened = False
    def __init__(self):
        self.pool = queue.Queue(4)
        self._buf1 = queue.Queue(4)
        self._buf2 = queue.Queue(4)
        self._buf3 = queue.Queue(4)

    def open(self, vfile):
        self.pipe = sp.Popen([FFPROBE_EXEC, '-v', 'quiet',
                             '-print_format', 'json',
                             '-show_format', '-show_streams',
                              vfile], stdout=sp.PIPE)
        info = self.pipe.stdout.read()
        info = json.loads(info)
        if len(info) == 0:
            return
        self.shape = (info['streams'][0]['height'], info['streams'][0]['width'])
        self.data_shape = (self.shape[0] + 64, self.shape[1])
        self.size = self.data_shape[0] * self.data_shape[1] * 2
        self.pipe.terminate()
        self.pipe = sp.Popen([FFMPEG_EXEC, '-v', 'quiet', '-hwaccel', 'auto',
                              '-i', vfile, '-f', 'rawvideo', '-pix_fmt', 'gray12le',
                              #'-s', str(self.shape[1]) + 'x' + str(self.shape[0]),
                              '-'],
                             stdout=sp.PIPE)
        if self.pipe.poll() is None:
            self.is_opened = True
            self._th = Thread(target=self._buffering)
            self._th.start()

    def _read(self):
        while self.is_opened:
            frame = np.frombuffer(self.pipe.stdout.read(self.size), np.int16)
            if(len(frame)) == 0:
                break
            frame = np.reshape(frame, self.data_shape)[:self.shape[0], :]
            self._buf1.put(frame)

    def _post_process1(self):
        while self.is_opened:
            try:
                frame = self._buf1.get(timeout=1)
            except:
                continue
            frame = torch.from_numpy(frame).cuda()
            frame = frame.type(torch.cuda.FloatTensor)
            frame = (frame + 2905.3) / 631.6
            # frame = frame.type(torch.cuda.ShortTensor)
            self._buf2.put(frame)

    def _post_process2(self):
        while self.is_opened:
            try:
                frame = self._buf2.get(timeout=1)
            except:
                continue
            frame = torch.exp(frame)
            frame = torch.clamp(frame, 0, 65535)
            frame = np.uint16(frame.cpu().numpy())
            self._buf3.put(frame)

    def _buffering(self):
        t1 = Thread(target=self._read)
        t1.start()
        t2 = Thread(target=self._post_process1)
        t2.start()
        t3 = Thread(target=self._post_process2)
        t3.start()
        while self.is_opened:
            try:
                frame = self._buf3.get(timeout=1)
            except:
                continue
            self.pool.put(frame)

    def isOpened(self):
        if self.is_opened == True:
            if self.pipe.poll() is None:
                return True
        self.is_opened = False
        return False

    def read(self):
        if not self.is_opened and self.pool.empty():
            return None
        return self.pool.get()

    def close(self):
        self.is_opened = False
        self.pipe.terminate()

path = 'F:/NewData/Thy1-6521-compressed/Thy1_6521/2017-10-28_20-05-18-2'

if __name__ == '__main__':
    import os, cv2, SimpleITK
    vc = VideoDecoder()
    vc.open(os.path.join(path, 'compressed/1.mp4'))
    ct = 1
    while 1:
        frame = vc.read()
        if frame is None:
            vc.close()
            break
        SimpleITK.WriteImage(SimpleITK.GetImageFromArray(frame), os.path.join(path, 'decomp', '1_' + str(ct) + '.tiff'))
        #cv2.imwrite(os.path.join(path, 'decomp', '1_' + str(ct) + '.tiff'), frame)
        print(ct)
        ct += 1
