from VISoR_Brain.positioning import visor_sample
import sys, os
from VISoR_Reconstruction.reconstruction.sample_reconstruct import reconstruct_image


def usage():
    print('''usage: python visor_reconstruction_ui.py <input_reconstruct_info> <output_image> <x> <y> <w> <h> <scale>
example: python visor_reconstruction_ui.py D:\chaoyu\monkey\\3.tar D:\chaoyu\monkey\\3_roi.mha 5976 6792 1656 1440 1
    ''')


def main():
    args = sys.argv
    if len(args) < 8:
        usage()
        return
    sample_file = args[1]
    dst = args[2]
    sl = visor_sample.VISoRSample()
    try:
        #raise FileExistsError
        sl.load(sample_file)
    except:
        from VISoR_Reconstruction.reconstruction.sample_reconstruct import reconstruct_sample
        from VISoR_Brain.format.raw_data import RawData
        path = os.path.split(os.path.split(sample_file)[0])[0]
        file = [os.path.join(path, f) for f in os.listdir(path) if f.split('.')[-1] == 'flsm'][0]
        r = RawData(file)
        sl = reconstruct_sample(r, {})
        try:
            sl.save(sample_file)
        except Exception:
            pass
    sl.load_columns()
    pixel_size = sl.raw_data.pixel_size * 4
    p1 = [float(args[3]) * pixel_size + sl.sphere[0][0],
          float(args[4]) * pixel_size + sl.sphere[0][1], 0]
    p1[2] = sl.sphere[0][2]
    p2 = [(float(args[3]) + float(args[5])) * pixel_size + sl.sphere[0][0],
          (float(args[4]) + float(args[6])) * pixel_size + sl.sphere[0][1], 0]
    p2[2] = sl.sphere[1][2]
    roi = [p1, p2]
    scale = float(args[7])
    img = reconstruct_image(sl, scale, roi)
    sl.set_image(img)
    sl.save_image(dst)

if __name__ == '__main__':
    main()

