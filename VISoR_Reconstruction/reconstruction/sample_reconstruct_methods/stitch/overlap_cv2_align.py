from VISoR_Brain.positioning.visor_sample import *
import time, cv2, os
import SimpleITK as sitk
import numpy as np
from VISoR_Brain.lib.flsmio import flsmio
saveTest = False
saveoverlap = False
PXSIZE=4
CHECKIMG=False

def align(p, o, i, x='x', thre=255):
    time0 = time.time()

    B1 = p.astype(np.int32)
    B1 = B1 / thre * 255
    B1[B1 > 255] = 255
    B1[B1 < 10] = 0
    B1 = B1.astype(np.uint8)
    if CHECKIMG==True:
        cv2.namedWindow(x, 0)
        cv2.imshow(x, B1)

    B2 = o.astype(np.int32)
    B2 = B2/ thre * 255
    B2[B2 > 255] = 255
    B2[B2 < 10] = 0
    B2 = B2.astype(np.uint8)
    if CHECKIMG == True:
        cv2.namedWindow(x+x,0)
        cv2.imshow(x+x, B2)
        cv2.waitKey(0)
    sp = B1.shape
    # 检测特征点  调用SIFT
    s = cv2.xfeatures2d.SIFT_create()
    kp1, des1 = s.detectAndCompute(B1, None)
    kp2, des2 = s.detectAndCompute(B2, None)
    # 蛮力匹配算法,有两个参数，距离度量(L2(default),L1)，是否交叉匹配(默认false)
    bf = cv2.BFMatcher()
    # 返回k个最佳匹配
    try:
        matches = bf.knnMatch(des1, des2, k=2)
    except:
        return [0, 0], 0
    # cv2.drawMatchesKnn expects list of lists as matches.
    # opencv3.0有drawMatchesKnn函数
    # Apply ratio tests
    # 比值测试，首先获取与A 距离最近的点B（最近）和C（次近），只有当B/C小于阈值时认为是匹配
    flag = 1
    err = 0.01
    good = []
    while (flag == 1):
        try:
            for m, n in matches:
                if m.distance < err * n.distance:
                    good.append((m.trainIdx, m.queryIdx))
        except:
            pass
        if len(good) < (len(matches) / 100):
            err += 0.01
        else:
            flag = 0
        if err > 1:
            err = 1
            print('cal ' + str(i) + x, 'not enough matches')
            break
    if len(good) == 0:
        thre = np.median(B1) + 10
        B1 = cv2.blur(B1, (5, 5))
        B2 = cv2.blur(B2, (5, 5))
        ret, B1 = cv2.threshold(B1, thre, 255, cv2.THRESH_BINARY)
        ret, B2 = cv2.threshold(B2, thre, 255, cv2.THRESH_BINARY)
        if CHECKIMG==True:
            cv2.imshow("i", B1)
            cv2.imshow("i+1", B2)
            cv2.waitKey(0)
        kp1, des1 = s.detectAndCompute(B1, None)
        kp2, des2 = s.detectAndCompute(B2, None)
        bf = cv2.BFMatcher()
        # 返回k个最佳匹配
        try:
            matches = bf.knnMatch(des1, des2, k=2)
        except:
            print('not match')
        flag = 1
        err = 0.01
        good = []
        while (flag == 1):
            try:
                for m, n in matches:
                    if m.distance < err * n.distance:
                        good.append((m.trainIdx, m.queryIdx))
            except:
                pass
            if len(good) < (len(matches) / 100):
                err += 0.01
            else:
                flag = 0
            if err > 1:
                err = 1
                print('cal ' + str(i - 1), 'not enough matches')
                break
        print('good', len(good), len(matches))
    # img3 = cv2.drawMatchesKnn(B1, kp1, B2, kp2, good[:10], None, flags=2)
    # cv2.namedWindow('',0)
    # cv2.imshow('',img3)
    # kps = np.float32([kp.pt for kp in kps])
    p1 = [kp1[k].pt for (_, k) in good]
    p2 = [kp2[k].pt for (k, _) in good]
    p1 = np.array(p1, dtype=float)
    p2 = np.array(p2, dtype=float)
    p = np.subtract(p1, p2)
    tp = np.sum(p, axis=0) / len(p)
    # compute the homography between the two sets of points
    print(tp)
    H = np.array([[1, 0, tp[0]], [0, 1, tp[1]], [0, 0, 1]])
    B20 = cv2.warpPerspective(B2, H, (sp[1], sp[0]))
    time1 = time.time()
    R = round(1 - err, 2)
    print(str(i) + x, ':', round(time1 - time0, 2), '  R:', R, ' matches: ', len(good), 'of', len(matches))
    if CHECKIMG==True:
        img_add = cv2.merge([np.zeros_like(B1), B20, B1])
        cv2.namedWindow("Warped Source Image", 0)
        cv2.imshow("Warped Source Image", img_add)
        cv2.waitKey(0)
    return [tp[0]*PXSIZE,tp[1]*PXSIZE], R


def calculate(py, oy, px, ox, saveoverlap, i):
    # print(i)
    offseti = [0, 0, 0]
    time0 = time.time()

    if i == 0:
        return offseti
    print('Aligning stack {0} to stack {1}'.format(i, i - 1))

    meanz = np.median(ox)
    thy = int(6.5 * meanz)
    thx = thy

    tpy, Ry = align(py, oy, i, 'y', thy)
    tpx, Rx = align(px, ox, i, 'x', thx)

    # tpz,Rz = align(prev_z_proj, z_proj, i, 'z',thz)
    print('[tp]', str(i) + 'x:', tpx, 'y:', tpy, meanz)
    # offseti[0] = -tpy[0]
    # offseti[1] = -tpx[1]
    # offseti[2] = -tpy[1]

    offseti = [-tpy[0], -tpy[1], -tpx[1]]
    print(str(i) + 'offset :', offseti)
    return offseti


def read_slice(path, side=False, pixel_size=4):
    print('reading slice')
    file_names = os.listdir(path)
    for name in file_names:
        if name.endswith('.flsm') and len(name.split('.')) == 2:
            path = path + '/' + name
    print(path)
    flsm_reader = flsmio.FlsmReader(path)

    if side==True:
        print('side_projection')
        reader = flsmio.SlideReader(flsm_reader.path("side_projection"))
    else:
        reader = flsmio.SlideReader(flsm_reader.path("projection"))
    rect = reader.region()
    pixel_sizes = reader.pixel_size()
    print(rect, pixel_sizes, reader.path())
    
    cols = int(np.round(rect[2] / pixel_size) + 10)
    rows = int(np.round(rect[3] / pixel_size) + 10)

    image_all = np.zeros((rows, cols), dtype=np.uint16)
    print(np.shape(image_all))

    columns = {}

    for i in range(reader.image_number()):
        image = None
        scale = 1.0

        if pixel_size > pixel_sizes[1]:
            image = reader.thumbnail(i)
            scale = pixel_sizes[1] / pixel_size
        else:
            image = reader.raw(i)
            scale = pixel_sizes[0] / pixel_size

        print(image.position(), np.shape(image))
        pos = image.position()

        h, w = np.shape(image)
        h = int(np.round(h * scale ))
        w = int(np.round(w * scale ))

        x = int(np.round((pos[0] - rect[0]) / pixel_size))
        y = int(np.round((pos[1] - rect[1]) / pixel_size))
        # print(x, y, h, w, np.shape(image), scale)

        image.decode()
        # cv2.imwrite("E:/tmp/"+str(i)+".tif", np.array(image))

        src = np.array(image)
        image1 = cv2.resize(src, (w, h))
        # print(np.shape(image1), src.dtype)

        image_all[y:y + h, x:x + w] = image1
        # print(np.shape(image_all[y:y+h, x:x+w]))

        column_index = pos[2]
        if column_index not in columns:
            columns[column_index] = np.zeros((h, cols), dtype=np.uint16)
        columns[column_index][0:h, x:x + w] = image1
    return columns


def reconstruct(sample_data: VISoRSample):
    time00 = time.time()
    r = sample_data.raw_data

    print('#########reading images\n', r.path)
    timea = time.time()
    columnsy = read_slice(r.path, side=False, pixel_size=PXSIZE)
    columnsx = read_slice(r.path, side=True, pixel_size=PXSIZE)
    timeb = time.time()
    print('read image:', timeb - timea)

    offset = np.zeros([len(sample_data.column_images), 3], np.float64)
    # dst = 'D:\Qianwei/Test/'+r.info['all_images']['projection']['path'][7:-10]+'/'
    # print(time.time() - time0)
    k = len(sample_data.column_images)
    results = []
    #img = columnsx[0]
    #sp = img.shape
    #overlaplen = int(300/PXSIZE)



    #
    for i in range(k):
        affine_t = [0, 1 / r.pixel_size, 0,
                    0, 0, 1 / r.pixel_size / np.cos(r.angle),
                    1 / r.column_spacing[i], 0, 1 / r.column_spacing[i] / np.tan(r.angle)]
        # 把真实空间的坐标转换成照片中的坐标
        p0 = r.column_pos0[i]
        af = sitk.AffineTransform(3)
        af.SetMatrix(affine_t)

        if r.column_spacing[i] < 0:
            p0 = (r.column_pos1[i][0], p0[1], p0[2])  # x y z
        #print('p0',p0)
        #p0_ = af.TransformPoint(p0)
        #print(p0_)
        tl = np.subtract(r.pos0, p0).tolist()
        tl_ = af.TransformPoint(tl)

        af.Translate(tl_)

        #print('tl',tl)
        tl2 = np.subtract(r.pos0, r.column_pos1[i]).tolist()
        #print('tl2',tl2)
        #print(af)
        if i == 0:
            af.Translate(offset[i])
            sample_data.transforms[i] = af
            results.append([0, 0, 0])
            continue

        print('generating stack {0} and stack {1}'.format(i - 1, i))
        timea = time.time()
        overlaplen = np.subtract(r.column_pos1[i-1], r.column_pos0[i])[1]
        overlaplen=int(np.round(overlaplen/PXSIZE))
        #print('**************overlaplen', overlaplen)
        #overlap_roi = [np.subtract(r.column_pos0[i], r.pos0),np.subtract(r.column_pos1[i - 1], r.pos0)]
        #overlap_roi[0][2]=-300
        #overlap_roi[0]=af.TransformPoint(overlap_roi[0])
        #overlap_roi[1]=af.TransformPoint(overlap_roi[1])
        #print(np.array(overlap_roi))
        py = columnsy[i-1][-overlaplen:]
        oy = columnsy[i][:overlaplen]
        #cv2.imshow('py',py)
        #cv2.imshow('oy',oy)
        px = columnsx[i * 2-1]
        ox = columnsx[i * 2]

        timeb = time.time()
        results.append(calculate(py, oy, px, ox, saveoverlap, i))
        print('********align image******** ', time.time() - timeb)

        offset[i] = np.add(offset[i - 1], results[i])
        af.Translate([-offset[i][0],offset[i][1],0],True)
        af.Translate([0,0,offset[i][2]])
        # print(offset[i])
        sample_data.transforms[i] = af
        #print(af)
    print('********TIME********* ', time.time() - time00)
