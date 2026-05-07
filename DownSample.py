

import os
import gc
import tifffile
import concurrent.futures as cf

def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

def safe_imread_tiff(path: str):
    """
    优先尝试 memmap（不一定所有tif都支持，比如压缩tif可能不支持），失败再回退到普通imread。
    """
    try:
        return tifffile.memmap(path)
    except Exception:
        return tifffile.imread(path)

def safe_imwrite_tiff(path: str, arr):
    """
    兼容不同版本 tifffile 的压缩参数写法。
    """
    try:
        # 新版本 tifffile 写法（推荐）
        tifffile.imwrite(path, arr, compression="zlib", compressionargs={"level": 1})
    except TypeError:
        # 老版本可能支持 compress=1
        tifffile.imwrite(path, arr, compress=1)

def downsample_image(file_path: str, save_path: str, rate: int):
    if os.path.exists(save_path):
        return

    if not os.path.exists(file_path):
        print(f"[Skip] missing: {file_path}")
        return

    # 读 + 下采样 + 写
    img = safe_imread_tiff(file_path)
    ds = img[::rate, ::rate].copy()   # copy成小图，避免写入时隐式复制/引用大图
    safe_imwrite_tiff(save_path, ds)

    # 释放引用，降低峰值（线程里更稳一点）
    del img, ds
    gc.collect()

def native_down_sample(root, save_root, rate=2, start=0, end=21500, max_workers=4, max_in_flight=None, ch_index = 1):
    ensure_dir(save_root)

    name_format = os.path.join(root, "Z{:05d}_C"+str(ch_index)+ ".tif")
    out_format  = os.path.join(save_root, "Z{:05d}_C"+str(ch_index)+ ".tif")

    # 同时“在途”的任务数：越小越省内存，但整体速度会慢一点
    if max_in_flight is None:
        max_in_flight = max_workers * 2  # 一个比较稳的默认值

    def submit_one(executor, i):
        down_index = i // rate
        file_path = name_format.format(i)
        save_path = out_format.format(down_index)
        return executor.submit(downsample_image, file_path, save_path, rate)

    # 只处理 i%rate==0 的切片 —— 等价于 range(start, end, rate)（更快更干净）
    indices = range(start - (start % rate), end, rate) if start % rate != 0 else range(start, end, rate)

    futures = set()
    done_count = 0

    with cf.ThreadPoolExecutor(max_workers=max_workers) as executor:
        for i in indices:
            # 控制在途任务数量，避免瞬间把内存顶爆
            while len(futures) >= max_in_flight:
                done, futures = cf.wait(futures, return_when=cf.FIRST_COMPLETED)
                for f in done:
                    f.result()  # 抛出线程中的异常
                    done_count += 1
                    if done_count % 200 == 0:
                        print(f"[Progress] done={done_count}")

            futures.add(submit_one(executor, i))

        # 收尾：等剩余任务完成
        for f in cf.as_completed(futures):
            f.result()
            done_count += 1
            if done_count % 200 == 0:
                print(f"[Progress] done={done_count}")

    print(f"[All Done] total_done={done_count}, max_workers={max_workers}, max_in_flight={max_in_flight}")

if __name__ == "__main__":

    # root = r"Y:\SIAT_SIAT\YaoYuchen\Wholebody\Mouse\B85\B85_temp\temp_block\BrainTrans_151_154\BrainImage_New\4.0_C2"
    # save_root = r"D:\yq\temp_151_153"

    # # 建议先从 2~4 个线程开始；内存紧张就把 max_in_flight 调小
    # native_down_sample(
    #     root, save_root,
    #     rate=2,
    #     start=0,
    #     end=20000,
    #     max_workers=8,  # 线程池数量（你要的设定项）
    #     max_in_flight=16,  # 同时最多处理8张图（关键：控内存）
    #     ch_index=2
    # )

    root = r"Y:\SIAT_SIAT\YaoYuchen\Wholebody\Mouse\B94\Reconstruction\Refinement\BrainTrans_67_72\BrainImage\4.0"
    save_root = r"D:\yq\b82\temp_67_73"

    # 建议先从 2~4 个线程开始；内存紧张就把 max_in_flight 调小
    native_down_sample(
        root, save_root,
        rate=2,
        start=0,
        end=20000,
        max_workers=8,  # 线程池数量（你要的设定项）
        max_in_flight=16,  # 同时最多处理8张图（关键：控内存）
        ch_index=1
    )