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
PICTURE_SIZE = (3500, 3500)

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

    #if src.dtype == np.uint16:
    #    src = np.uint8((cv2.log(np.float32(src + 100)) - 4.6) * 39.4)
    #src = np.uint16(cv2.exp(np.float32(src) / 39.4 + 4.6)) - 145

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
    #'''
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
    if not os.path.exists(step2):
        os.mkdir(step2)
    if not os.path.exists(img2):
        os.mkdir(img2)
    if not os.path.exists(par):
        os.mkdir(par)

    if flip:
        j_range = range(e - 1, -1, -1)
        s = e
    else:
        j_range = range(1, e + 1, 1)
        s = 0
    ct = 0

    elastix(join(cfgpath, "cb.tif"),
            join(step1, str(START_SLICE) + "_" + str(s) + ".tif"),
            step1,
            [join(cfgpath, "tp_align_surface_rigid.txt")])
    convertRawImage(join(step1, "result.0.raw"), join(img2, str(ct) + ".tif"), PICTURE_SIZE)
    copyfile(join(step1, "TransformParameters.0.txt"),
             join(par, "r.p." + str(START_SLICE) + ".txt"))

    
    elastix(grid, grid, step1, [join(cfgpath, "parameters_Inverse_Rigid.txt")],
            join(step1, "TransformParameters.0.txt"))
    editParamFile(join(step1, "TransformParameters.0.txt"),
                  join(par, "r.i." + str(START_SLICE) + ".txt"), "\"NoInitialTransform\"")
	

    for j in j_range:
        ct += 1

        transfomix(join(step1, str(START_SLICE) + "_" + str(j) + ".tif"),
                   step1,
                   join(par, "r.p." + str(START_SLICE) + ".txt"))
        convertRawImage(join(step1, "result.raw"), join(img2, str(ct) + ".tif"), PICTURE_SIZE)

        j += 1


    i = START_SLICE + 1
    ct = ct + 1

    while os.path.exists(join(step1, str(i) + "_" + str(0) + ".tif")):

        elastix(join(img2, str(ct - 1) + ".tif"),
                join(step1, str(i) + "_" + str(s) + ".tif"),
                step1,
                [join(cfgpath, "tp_align_surface_rigid.txt"),
                 join(cfgpath, "tp_align_surface_bspline.txt")])
        copyfile(join(step1, "TransformParameters.0.txt"),
                 join(par, "r.p." + str(i) + ".txt"))
        editParamFile(join(step1, "TransformParameters.1.txt"),
                      join(par, "b.p." + str(i) + "." + str(s) + ".txt"),
                      join(par, "r.p." + str(i) + ".txt"))
        convertRawImage(join(step1, "result.1.raw"), join(img2, str(ct) + ".tif"), PICTURE_SIZE)

        
        elastix(grid, grid, step1, [join(cfgpath, "parameters_Inverse_Rigid.txt")],
                join(step1, "TransformParameters.0.txt"))
        editParamFile(join(step1, "TransformParameters.0.txt"),
                      join(par, "r.i." + str(i) + ".txt"), "\"NoInitialTransform\"")
        editParamFile(join(step1, "TransformParameters.1.txt"),
                      join(step1, "t.txt"), "\"NoInitialTransform\"")

        elastix(grid, grid, step1, [join(cfgpath, "parameters_Inverse_bs.txt")],
                join(step1, "t.txt"))
        editParamFile(join(step1, "t.txt"),
                      join(par, "b.i." + str(i) + "." + str(s) + ".txt"),
                      join(par, "r.i." + str(i) + ".txt"))
        copyfile(join(par, "b.i." + str(i) + "." + str(s) + ".txt"),
                 join(par, "b.i." + str(i) + ".txt"))
		


        ct += 1
        for j in j_range:
            editParamFile(join(par, "b.p." + str(i) + "." + str(s) + ".txt"),
                          join(par, "b.p." + str(i) + "." + str(j) + ".txt"),
                          join(par, "r.p." + str(i) + ".txt"), abs(float(e - s - j)) / float(e))
            transfomix(join(step1, str(i) + "_" + str(j) + ".tif"),
                       step1,
                       join(par, "b.p." + str(i) + "." + str(j) + ".txt"))
            convertRawImage(join(step1, "result.raw"), join(img2, str(ct) + ".tif"), PICTURE_SIZE)


            #editParamFile(join(par, "b.p." + str(i) + "." + str(0) + ".txt"),
            #              join(step1, "t.txt"), "\"NoInitialTransform\"")
            #elastix(grid, grid, step1, [join(cfgpath, "parameters_Inverse_bs.txt")],
            #        join(step1, "t.txt"))
            #editParamFile(join(step1, "TransformParameters.1.txt"),
            #              join(par, "b.i." + str(i) + "." + str(j) + ".txt"),
            #              join(par, "r.i." + str(i) + ".txt"))


            ct += 1
        i += 1

    step2 = join(tmpdir, "step2")
    img2 = join(step2, "img")
    ct = 0
    stack = []
    while 1:
        img = cv2.imread(join(img2, str(ct) + ".tif"), -1)
        if img is None:
            break
        stack.append(img)
        ct += 1
    stack = np.array(stack)
    imsave(join(step2, "brain.tif"), stack)
    
    '''
    elastix(join(step2, "brain.tif"), join(cfgpath, "template.tif"), step2,
            [join(cfgpath, "tp_align_surface_rigid.txt"),
             join(cfgpath, "parameters_Affine.txt"),
             join(cfgpath, "parameters_BSpline_3d.txt")])
    copyfile(join(step2, "TransformParameters.0.txt"),
             join(dstdir, "br.r.p.txt"))
    editParamFile(join(step2, "TransformParameters.1.txt"),
                  join(dstdir, "br.a.p.txt"),
                  join(dstdir, "br.r.p.txt"))
    editParamFile(join(step2, "TransformParameters.2.txt"),
                  join(dstdir, "br.b.p.txt"),
                  join(dstdir, "br.a.p.txt"))
    '''



if __name__ == "__main__":
    dir = "F:/brain_processing/brain_stitching/data/activity/flatten_8"
    dirlist = [f
               for f in os.listdir(dir)
               if os.path.isdir(join(dir, f))]
    lst = []
    for f in dirlist:
        lst.append((int(f), os.path.join(dir, f)))
    lst.sort(cmp=lambda x, y: x[0] - y[0])
    deform(lst,
           "F:/brain_processing/brain_stitching/code/baseline/Deform_activity_8/res",
           "F:/brain_processing/brain_stitching/code/baseline/Deform_activity_8/tmp")

