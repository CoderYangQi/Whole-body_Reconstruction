import sys, cv2, os
import numpy as np
from os.path import join
from subprocess import call
from shutil import copyfile
from tifffile import imsave
from PIL import Image

ELASTIX_EXEC = "D:/Elastix_Fundamental/elastix_v4.8/elastix"
TRANSFORMIX_EXEC = "D:/Elastix_Fundamental/elastix_v4.8/transformix"

START_SLICE = 2
second = START_SLICE + 1
PICTURE_SIZE = (600, 600)
IMG_NUM = 41

flip = False

def generateChessboard(dstfile, shape, dtype=np.uint8):
    cb = np.zeros(shape, dtype)
    for i in range(0, shape[0], 16):
        for j in range(0, shape[1], 16):
            if (i + j)%32 == 0:
                cb[i:min(i + 16, shape[0]), j:min(j + 16, shape[1])] += 255
    cv2.imwrite(dstfile, cb)


def elastix(fix, mov, out, p, t0 = None):
    param = [ELASTIX_EXEC, "-f", fix, "-m", mov, "-out", out]
    for par in p:
        param.append("-p")
        param.append(par)
    if not t0 is None:
        param.append("-t0")
        param.append(t0)
    if not call(param) == 0:
        input("error")


def transfomix(inp, out, tp, def_ = None):
    param = []
    if not def_ is None:
        param = [TRANSFORMIX_EXEC, "-def", def_, "-out", out, "-tp", tp]
    else:
        param = [TRANSFORMIX_EXEC, "-in", inp, "-out", out, "-tp", tp]
    if not call(param) == 0:
        input("error")


def preProcess(srcfile, dstfile, shape):
    src = cv2.imread(srcfile, -1)
    img = np.zeros(shape, src.dtype)
    roi = img[(shape[0] - src.shape[0]) / 2: (shape[0] + src.shape[0]) / 2,
              (shape[1] - src.shape[1]) / 2: (shape[1] + src.shape[1]) / 2]
    np.copyto(roi, src)
    _, img = cv2.threshold(255 - img, 240, 0, cv2.THRESH_TRUNC)
    img = cv2.medianBlur(255 - img, 3)
    msk = np.zeros((shape[0] + 2, shape[1] + 2), np.uint8)
    cv2.floodFill(img, msk, (0, 0), 0, 0, 2, cv2.FLOODFILL_FIXED_RANGE)
    img = np.uint16(img)
    cv2.normalize(img, img, 16384, cv2.NORM_L2)
    cv2.imwrite(dstfile, img)


def convertRawImage(srcfile, dstfile, shape):
    img = np.fromfile(srcfile, np.int16)
    img = np.reshape(img, shape)
    cv2.imwrite(dstfile, img)

def averageParamFile(srcfile, dstfile):
    src = open(srcfile, 'r')
    dst = open(dstfile, 'w')
    while 1:
        line = src.readline()
        if len(line) == 0:
            break
        l = line.split(' ')
        if l[0] == "(TransformParameters":
			for p in l:
				if p[0] == '(':
					line = p + " "
				elif p[len(p) - 2] == ')':
					line += "{0:.6f}".format(float(p[0:len(p) - 2]) / 2) + ")\n"
				else:
					line += "{0:.6f} ".format(float(p) / 2)
        dst.write(line)

def editParamFile2(srcfile1, srcfile2, dstfile, a):
    src1 = open(srcfile1, 'r')
    src2 = open(srcfile2, 'r')
    dst = open(dstfile, 'w')
    while 1:
        line1 = src1.readline()
        line2 = src2.readline()
        if (len(line1) == 0 or len(line2) == 0):
            break
        l1 = line1.split(' ')
        l2 = line2.split(' ')

        if l1[0] == "(TransformParameters":
            line1 = l1[0] + " "
            i = 1
            while True:
                if (l1[i])[-2] == ')':
                    line1 += "{0:.6f}".format( float((l1[i])[0:len(l1[i]) - 2]) + (float((l2[i])[0:len(l2[i]) - 2]) - float((l1[i])[0:len(l1[i]) - 2 ]))* a) + ")\n"
                    break
                line1 += "{0:.6f} ".format( float(l1[i]) + (float(l2[i]) - float(l1[i])) * a )  
                i = i + 1
        dst.write(line1)


def editParamFile(srcfile, dstfile, initialtp = None, a = 1.0):
    src = open(srcfile, 'r')
    dst = open(dstfile, 'w')
    while 1:
        line = src.readline()
        if len(line) == 0:
            break
        l = line.split(' ')
        if l[0] == "(InitialTransformParametersFileName":
            if not initialtp is None:
                line = l[0] + " " + initialtp + ")\n"
        if l[0] == "(TransformParameters":
            for p in l:
                if p[0] == '(':
                    line = p + " "
                elif p[len(p) - 2] == ')':
                    line += "{0:.6f}".format(float(p[0:len(p) - 2]) * a) + ")\n"
                else:
                    line += "{0:.6f} ".format(float(p) * a)
        dst.write(line)


def deform(srcdirlist, dstdir, tmpdir):
    cfgpath = os.path.dirname(__file__)
    grid = join(cfgpath, "cb.tif")
    generateChessboard(grid, PICTURE_SIZE)
    sl3 = None

    for f in srcdirlist:
        if f[0] == START_SLICE:
            sl3 = f[1]
            break
    if sl3 is None:
        return

    step1 = join(tmpdir, "step1")
    if not os.path.exists(step1):
        os.mkdir(step1)

    e = 0
    for f in srcdirlist:
        imglist = os.listdir(f[1])
        e = len(imglist) - 1
        
        for i in range(0, len(imglist)):
            preProcess(join(f[1], imglist[i]), join(step1, str(f[0]) + "_" + imglist[i]), PICTURE_SIZE)
		

    step2 = join(tmpdir, "step2")
    img2 = join(step2, "img")
    par = join(dstdir, "par")
    par2 = join(dstdir, "par2")
    img3 = join(step2, "img2")


    if not os.path.exists(step2):
        os.mkdir(step2)
    if not os.path.exists(img2):
        os.mkdir(img2)
    if not os.path.exists(img3):
        os.mkdir(img3)
    if not os.path.exists(par):
        os.mkdir(par)
    if not os.path.exists(par2):
        os.mkdir(par2)

    if flip:
        j_range = range(e - 1, -1, -1)
        s = e
    else:
        j_range = range(1, e + 1, 1)
        s = 0

    start = 0

    
    elastix(join(cfgpath, "cb.tif"),
            join(step1, str(START_SLICE) + "_" + str(s) + ".tif"),
            step1,
            [join(cfgpath, "parameters_Rigid.txt")])

    convertRawImage(join(step1, "result.0.raw"), join(img2, str(START_SLICE) + "_" + str(start) + ".tif"), PICTURE_SIZE)
    convertRawImage(join(step1, "result.0.raw"), join(img3, str(START_SLICE) + "_" + str(start) + ".tif"), PICTURE_SIZE)

    copyfile(join(step1, "TransformParameters.0.txt"),
             join(par, "r.p." + str(START_SLICE) + "_" + str(start) +".txt"))

    for j in j_range:
        transfomix(join(step1, str(START_SLICE) + "_" + str(j) + ".tif"),
                   step1,
                   join(par, "r.p." + str(START_SLICE) + "_" + str(0) +".txt"))

        convertRawImage(join(step1, "result.raw"), join(img2, str(START_SLICE) + "_" + str(j) + ".tif"), PICTURE_SIZE)
        convertRawImage(join(step1, "result.raw"), join(img3, str(START_SLICE) + "_" + str(j) + ".tif"), PICTURE_SIZE)
        j += 1

    for k in range(second, START_SLICE + IMG_NUM):
        elastix(join(img2, str(k - 1) + "_" + str(e) + ".tif"),
                join(step1, str(k) + "_" + str(start) + ".tif"),
                step1,
                [join(cfgpath, "parameters_Rigid.txt")])
        convertRawImage(join(step1, "result.0.raw"), join(img2, str(k) + "_" + str(start) + ".tif"), PICTURE_SIZE)

        copyfile(join(step1, "TransformParameters.0.txt"),
             join(par, "r.p." + str(k) + "_" + str(start) +".txt"))

        for m in range(start + 1, e + 1):
            transfomix(join(step1, str(k) + "_" + str(m) + ".tif"),
                       step1,
                       join(par, "r.p." + str(k) + "_" + str(start) +".txt"))

            convertRawImage(join(step1, "result.raw"), join(img2, str(k) + "_" + str(m) + ".tif"), PICTURE_SIZE)




    elastix(join(img2, str(START_SLICE) + "_" + str(e) + ".tif"),
                join(img2, str(second) + "_" + str(start) + ".tif"),
                step1,
                [join(cfgpath, "parameters_Rigid.txt"),
                 join(cfgpath, "parameters_BSpline.txt")])
    copyfile(join(step1, "TransformParameters.0.txt"),
                join(par2, "r.p." + str(second) + "_" + str(start) + "_ex.txt"))
    editParamFile(join(step1, "TransformParameters.1.txt"),
                  join(par2, "b.p." + str(second) + "_" + str(start) + ".txt"),
                  join(par2, "r.p." + str(second) + "_" + str(start) + "_ex.txt"))


    for k in range(second + 1, START_SLICE + IMG_NUM):

        elastix(join(img2, str(k - 1) + "_" + str(e) + ".tif"),
                join(img2, str(k) + "_" + str(start) + ".tif"),
                step1,
                [join(cfgpath, "parameters_Rigid.txt"),
                 join(cfgpath, "parameters_BSpline.txt")])
        copyfile(join(step1, "TransformParameters.0.txt"),
                 join(par2, "r.p." + str(k) + "_" + str(start) + "_ex.txt"))
        editParamFile(join(step1, "TransformParameters.1.txt"),
                      join(par2, "b.p." + str(k) + "_" + str(start) + "_ex.txt"),
                      join(par2, "r.p." + str(k) + "_" + str(start) + "_ex.txt"))

        elastix(join(img2, str(k) + "_" + str(start) + ".tif"),
                join(img2, str(k - 1) + "_" + str(e) + ".tif"),
                step1,
                [join(cfgpath, "parameters_Rigid.txt"),
                 join(cfgpath, "parameters_BSpline.txt")])
        copyfile(join(step1, "TransformParameters.0.txt"),
                join(par2, "r.p." + str(k - 1) + "_" + str(e) + "_ex.txt"))
        editParamFile(join(step1, "TransformParameters.1.txt"),
                      join(par2, "b.p." + str(k - 1) + "_" + str(e) + "_ex.txt"),
                      join(par2, "r.p." + str(k - 1) + "_" + str(start) + "_ex.txt"))

        averageParamFile(join(par2, "b.p." + str(k) + "_" + str(start) + "_ex.txt"),join(par2, "b.p." + str(k) + "_" + str(start) + ".txt"))
        averageParamFile(join(par2, "b.p." + str(k - 1) + "_" + str(e) + "_ex.txt"),join(par2, "b.p." + str(k - 1) + "_" + str(e) + ".txt"))


    for k in range(second, IMG_NUM + START_SLICE - 1):
        for m in range(start + 1, e):
            editParamFile2(join(par2, "b.p." + str(k) + "_" + str(start) + ".txt"),
                           join(par2, "b.p." + str(k) + "_" + str(e) + ".txt"), 
                           join(par2, "b.p." + str(k) + "_" + str(m) + ".txt"),
                           (float(m) / float(e)))


    for m in range(start + 1, e + 1):
        editParamFile(join(par2, "b.p." + str(START_SLICE  + IMG_NUM - 1)  + "_" + str(start) + ".txt"),
                      join(par2, "b.p." + str(START_SLICE  + IMG_NUM - 1) + "_" + str(m) + ".txt"), 
                      join(par2, "r.p." + str(START_SLICE  + IMG_NUM - 1) + "_" + str( start ) + "_ex.txt"),
                      abs(float(e - m )) / float(e))



    for k in range(second, IMG_NUM + START_SLICE):
        for m in range(start , e + 1):
            transfomix(join(img2, str(k) + "_" + str(m) + ".tif"),
                        step1,
                        join(par2, "b.p." + str(k) + "_" + str(m) + ".txt"))
            convertRawImage(join(step1, "result.raw"), join(img3, str(k) + "_" + str(m) + ".tif"), PICTURE_SIZE)
	

    stack = []
    for j in range(START_SLICE, START_SLICE + IMG_NUM):
        for m in range(0 , e + 1):
            img = cv2.imread(join(img3, str(j) + "_" + str(m) + ".tif"), -1)
            stack.append(img)
    stack = np.array(stack)
    imsave( "brain.tif", stack)




if __name__ == "__main__":
    dir = "F:/brain_processing/brain_stitching/data/activity/flatten_25"
    dirlist = [f
               for f in os.listdir(dir)
               if os.path.isdir(join(dir, f))]
    lst = []
    for f in dirlist:
        lst.append((int(f), os.path.join(dir, f)))
    lst.sort(cmp=lambda x, y: x[0] - y[0])
    deform(lst,
           "F:/brain_processing/brain_stitching/code/baseline/Deform_activity_50/res",
           "F:/brain_processing/brain_stitching/code/baseline/Deform_activity_50/tmp")
