import sys

import SimpleITK as sitk
from PyQt5 import QtWidgets, QtCore

from interactive_registration.visor_interactive_registration_ui import *


class elastixThread(QtCore.QThread):
    progress = QtCore.pyqtSignal(int)
    def __init__(self):
        super(elastixThread, self).__init__()
        self.filter = sitk.ElastixImageFilter()
        self.filter.SetLogToFile(False)
        self.filter.SetOutputDirectory('interactive_registration/elastix_tmp')
        self.paramMaps_interactive = sitk.VectorOfParameterMap()
        self.paramMaps_interactive.append(sitk.ReadParameterFile('parameters/parameters_BSpline_interactive.txt'))
        self.paramMaps_initial = sitk.VectorOfParameterMap()
        self.paramMaps_initial.append(sitk.ReadParameterFile('parameters/parameters_Rigid.txt'))
        self.paramMaps_initial.append(sitk.ReadParameterFile('parameters/parameters_Affine.txt'))
        self.paramMaps_initial.append(sitk.ReadParameterFile('parameters/parameters_BSpline.txt'))
        self.filter.SetParameterMap(self.paramMaps_initial)
        self.tfilter = sitk.TransformixImageFilter()
        self.tfilter.SetOutputDirectory('interactive_registration/elastix_tmp')
        self.initial = True

    def run(self):
        self.progress.emit(0)
        self.filter.Execute()
        self.progress.emit(100)


class mainwindow(QtWidgets.QMainWindow, Ui_MainWindow):
    def __init__(self):
        super(mainwindow, self).__init__()
        self.setupUi(self)
        self.actionchange_templete.triggered.connect(self.loadFixImage)
        self.actionopen.triggered.connect(self.loadMoveImage)
        self.avFix.scene.pos_changed_info.connect(self.statusbar.showMessage)
        self.avMove.scene.pos_changed_info.connect(self.statusbar.showMessage)
        self.avFix.scene.pos_changed.connect(self.avMove.paintSyncCursor)
        self.avMove.scene.pos_changed.connect(self.avFix.paintSyncCursor)
        self.mode = 'inactive'
        self.original_move_image = None
        self.initial_transform = None
        self.additional_transform = None
        self.elastix_thread = elastixThread()

    def loadFixImage(self):
        d = QtWidgets.QFileDialog()
        d.setFileMode(QtWidgets.QFileDialog.ExistingFile)
        file = d.getOpenFileName(self, 'openFile')
        try:
            img = sitk.ReadImage(file[0])
            if img.GetDimension() == 2:
                img = sitk.JoinSeries([img])
        except:
            return
        self.avFix.loadImage(img)
        if self.avMove.mode != 'inactive':
            self.setMode('view')

    def loadMoveImage(self):
        d = QtWidgets.QFileDialog()
        d.setFileMode(QtWidgets.QFileDialog.ExistingFile)
        file = d.getOpenFileName(self, 'openFile')
        try:
            img = sitk.ReadImage(file[0])
            if img.GetDimension() == 2:
                img = sitk.JoinSeries([img])
        except:
            return
        self.original_move_image = img
        self.avMove.loadImage(img)
        if self.avFix.mode != 'inactive':
            self.setMode('view')

    def exportImage(self):
        d = QtWidgets.QFileDialog()
        d.setAcceptMode(QtWidgets.QFileDialog.AcceptSave)
        d.setFileMode(QtWidgets.QFileDialog.AnyFile)
        file = d.getSaveFileName(self, 'Export')[0]
        try:
            sitk.WriteImage(self.avMove.image, file)
        except:
            print('fail to save' + file)

    def setMode(self, mode):
        if self.avMove.mode == 'inactive' or self.avFix.mode == 'inactive':
            return
        if self.mode == 'inactive':
            self.pbAddPoint.clicked.connect(self.clicked_pbAddPoint)
            self.pbView.clicked.connect(self.clicked_pbView)
            self.pbEdit.clicked.connect(self.clicked_pbEdit)
            self.pbElastix.clicked.connect(self.runElastix)
            self.avFix.mark_updated.connect(self.updatePointList)
            self.avMove.mark_updated.connect(self.updatePointList)
            self.elastix_thread.progress.connect(self.elastixProgress)
            self.actionexport_points.setEnabled(True)
            self.actionexport_points.triggered.connect(self.action_exportpoints)
            self.actionexport_registrated_image.setEnabled(True)
            self.actionexport_registrated_image.triggered.connect(self.exportImage)
            self.actionimport_points.triggered.connect(self.action_importpoints)
            self.pbEdit.setEnabled(True)
            self.pbView.setEnabled(True)
            self.pbElastix.setEnabled(True)
        if self.mode == 'edit':
            self.tableWidget.cellPressed.disconnect(self.avFix.setMarkIdx)
            self.tableWidget.cellPressed.disconnect(self.avMove.setMarkIdx)
        if mode == 'view':
            self.mode = mode
            self.avFix.setMode('view')
            self.avMove.setMode('view')
            self.pbAddPoint.setEnabled(False)
        if mode == 'edit':
            self.mode = mode
            self.avFix.setMode('mark')
            self.avMove.setMode('mark')
            self.avFix.mark_idx = 0
            self.avFix.mark_idx = 0
            self.tableWidget.setCurrentCell(0, 0)
            self.tableWidget.cellPressed.connect(self.avFix.setMarkIdx)
            self.tableWidget.cellPressed.connect(self.avMove.setMarkIdx)
            self.pbAddPoint.setEnabled(True)

    def clicked_pbAddPoint(self):
        self.tableWidget.insertRow(self.tableWidget.rowCount())
        self.tableWidget.setCurrentCell(self.tableWidget.rowCount() - 1, 0)
        self.avFix.setMarkIdx(self.tableWidget.rowCount() - 1)
        self.avMove.setMarkIdx(self.tableWidget.rowCount() - 1)
        self.updatePointList()

    def clicked_pbView(self):
        self.setMode('view')

    def clicked_pbEdit(self):
        self.setMode('edit')

    def updatePointList(self):
        for i in range(self.tableWidget.rowCount()):
            if i in self.avFix.markList:
                if self.avFix.markList[i] is not None:
                    pos = self.avFix.markList[i]
                    s = '{0:.2f},{1:.2f},{2:.2f}'.format(pos[0], pos[1], pos[2])
                    self.tableWidget.setItem(i, 0, QtWidgets.QTableWidgetItem(s))
            if i in self.avMove.markList:
                if self.avMove.markList[i] is not None:
                    pos = self.avMove.markList[i]
                    s = '{0:.2f},{1:.2f},{2:.2f}'.format(pos[0], pos[1], pos[2])
                    self.tableWidget.setItem(i, 1, QtWidgets.QTableWidgetItem(s))

    def exportPointLists(self, path_fix, path_move):
        l = [[], []]
        t = [None, None]
        for i in range(self.tableWidget.rowCount()):
            if i in self.avFix.markList and i in self.avMove.markList:
                if self.avFix.markList[i] is not None and self.avMove.markList[i] is not None:
                    l[0].append(self.avFix.markList[i])
                    l[1].append(self.avMove.markList[i])
        for i in range(2):
            ls = l[i]
            t[i] = 'point\n'
            t[i] += str(len(ls)) + '\n'
            for pos in ls:
                t[i] += '{0:.2f} {1:.2f} {2:.2f}\n'.format(pos[0], pos[1], pos[2])
        f1 = open(path_fix, 'w')
        f1.write(t[0])
        f2 = open(path_move, 'w')
        f2.write(t[1])
        return len(l[0])

    def importPointList(self, path, fix_or_move):
        d = self.avMove
        if fix_or_move == 'fix':
            d = self.avFix
        f1 = open(path)
        f1.readline()
        ct = int(f1.readline())
        self.tableWidget.setRowCount(max(self.tableWidget.rowCount(), ct))
        for i in range(ct):
            line = f1.readline()
            print(line)
            if len(line) < 3:
                break
            line = line.split(' ')
            d.appendMark([float(line[0]), float(line[1]), float(line[2])])
            d.mark_idx += 1
            d.paintMark()

    def action_importpoints(self):
        d = QtWidgets.QFileDialog()
        d.setFileMode(QtWidgets.QFileDialog.ExistingFile)
        file1 = d.getOpenFileName(self, 'Import points on templete')[0]
        file2 = d.getOpenFileName(self, 'Import points on image')[0]
        self.importPointList(file1, 'fix')
        self.importPointList(file2, 'move')
        self.updatePointList()

    def action_exportpoints(self):
        d = QtWidgets.QFileDialog()
        d.setAcceptMode(QtWidgets.QFileDialog.AcceptSave)
        d.setFileMode(QtWidgets.QFileDialog.AnyFile)
        file1 = d.getSaveFileName(self, 'Export')[0]
        file2 = d.getSaveFileName(self, 'Export')[0]
        self.exportPointLists(file1, file2)

    def runElastix(self):
        self.pbElastix.setEnabled(False)
        path1 = 'interactive_registration/elastix_tmp/fix.txt'
        path2 = 'interactive_registration/elastix_tmp/move.txt'

        if self.exportPointLists(path1, path2) < 1 :#or self.initial_transform is None:
            self.elastix_thread.filter.SetParameterMap(self.elastix_thread.paramMaps_initial)
        else:
            self.elastix_thread.initial = False
            #transfomix = sitk.TransformixImageFilter()
            #transfomix.SetFixedPointSetFileName(path2)
            #transfomix.SetTransformParameter(self.additional_transform)
            #transfomix.SetOutputDirectory('interactive_registration/elastix_tmp')
            #transfomix.Execute()
            self.elastix_thread.filter.SetParameterMap(self.elastix_thread.paramMaps_interactive)
            self.elastix_thread.filter.SetFixedPointSetFileName(path1)
            self.elastix_thread.filter.SetMovingPointSetFileName(path2)
        self.elastix_thread.filter.SetFixedImage(self.avFix.image)
        self.elastix_thread.filter.SetMovingImage(self.original_move_image)
        self.elastix_thread.start()

    def elastixProgress(self, progress):
        self.progressBar.setValue(progress)
        self.statusbar.showMessage('Running elastix...')
        if progress == 100:
            if 0:#self.initial_transform is None:
                self.original_move_image = self.elastix_thread.filter.GetResultImage()
                self.initial_transform = self.elastix_thread.filter.GetTransformParameterMap()
            else:
                self.additional_transform = self.elastix_thread.filter.GetTransformParameterMap()
            self.pbElastix.setEnabled(True)
            self.avMove.loadImage(self.elastix_thread.filter.GetResultImage())
            self.statusbar.showMessage('Elastix finished.')


def execute():
    app = QtWidgets.QApplication(sys.argv)
    window = mainwindow()
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    execute()