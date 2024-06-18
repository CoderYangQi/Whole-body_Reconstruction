import os
import tifffile
import concurrent.futures

def create_folder(name):
    if not os.path.exists(name):
        os.mkdir(name)

def downsample_image(file_path, save_path, rate):
    if os.path.exists(save_path):
        return
    print("Processing", file_path)
    img = tifffile.imread(file_path)
    resize_img = img[::rate, ::rate]
    tifffile.imwrite(save_path, resize_img, compress=1)

def native_down_sample(root, save_root, rate=2):
    create_folder(save_root)
    nums = 32799  # Assuming 19900 is the total number of files you want to process
    end = 39599  # Assuming 19900 is the total number of files you want to process
    start = 30800
    name_format = os.path.join(root, "Z{:05d}_C1.tif")
    out_path_format = os.path.join(save_root, "Z{:05d}_C1.tif")

    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = []
        for i in range(start,end):
            if i % rate == 0:
                down_index = i // rate
                file_path = name_format.format(i)
                save_path = out_path_format.format(down_index)
                futures.append(executor.submit(downsample_image, file_path, save_path, rate))

        for future in concurrent.futures.as_completed(futures):
            future.result()  # Wait for each thread to complete and handle exceptions if any

if __name__ == '__main__':
    root = r"D:\USERS\yq\TH2_Part\90_110\Reconstruction\BrainImage\1.0"
    save_root = r"D:\USERS\yq\TH2_Part\90_110\Reconstruction\BrainImage\1.0\down_sample_combine"
    native_down_sample(root, save_root)

