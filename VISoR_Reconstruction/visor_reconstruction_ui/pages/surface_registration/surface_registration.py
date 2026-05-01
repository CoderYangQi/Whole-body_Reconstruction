import sys, os

import SimpleITK as sitk
from PyQt5 import QtWidgets

from .open_dialog import OpenDialog
from .surface_registration_ui import Ui_Form


class SurfaceRegistrationPage(QtWidgets.QWidget, Ui_Form):
    def __init__(self, parent=None):
        super(SurfaceRegistrationPage, self).__init__(parent)
        self.setupUi(self)
        #self.avFix.scene.pos_changed_info.connect(self.statusbar.showMessage)
        #self.avMove.scene.pos_changed_info.connect(self.statusbar.showMessage)
        self.pb_open.clicked.connect(self.open_files)
        self.pb_save.clicked.connect(self.save_points)
        self.avFix.scene.pos_changed.connect(self.avMove.paintSyncCursor)
        self.avMove.scene.pos_changed.connect(self.avFix.paintSyncCursor)
        self.mode = 'inactive'
        self.original_move_image = None
        self.initial_transform = None
        self.additional_transform = None
        self.transform_fixed = sitk.Transform()
        self.transform_moving = sitk.Transform()
        self.inverse_transform_fixed = sitk.Transform()
        self.inverse_transform_moving = sitk.Transform()

    def open_files(self):
        dialog = OpenDialog()
        dialog.exec()
        path = {k: dialog.setting_group[k][2] for k in dialog.setting_group}
        if os.path.isfile(path['fimg']):
            fdf = None
            if os.path.isfile(path['fdf']):
                fdf = path['fdf']
            self.load_fixed_image(path['fimg'], fdf)
        if os.path.isfile(path['mimg']):
            mdf = None
            if os.path.isfile(path['mdf']):
                mdf = path['mdf']
            self.load_moving_image(path['mimg'], mdf)
        if os.path.isfile(path['fp']):
            self.importPointList(path['fp'], 'fix')
        if os.path.isfile(path['mp']):
            self.importPointList(path['mp'], 'move')
        self.updatePointList()

    def load_deformed_image(self, file, df_file=None):
        img = sitk.ReadImage(file)
        if img.GetDimension() == 2:
            img = sitk.JoinSeries([img])
        t = sitk.Transform()
        it = sitk.Transform()
        if df_file is not None:
            df = sitk.ReadImage(df_file)
            idf = sitk.InvertDisplacementField(df)
            if df.GetDimension() == 2:
                df = sitk.JoinSeries([sitk.Compose(sitk.VectorIndexSelectionCast(df, 0),
                                                   sitk.VectorIndexSelectionCast(df, 1),
                                                   sitk.Image(df.GetSize(), sitk.sitkFloat64))])
                idf = sitk.JoinSeries([sitk.Compose(sitk.VectorIndexSelectionCast(idf, 0),
                                                   sitk.VectorIndexSelectionCast(idf, 1),
                                                   sitk.Image(idf.GetSize(), sitk.sitkFloat64))])
            t = sitk.DisplacementFieldTransform(sitk.Image(df))
            it = sitk.DisplacementFieldTransform(sitk.Image(idf))
            img = sitk.Resample(img, df, t)
        return img, t, it


    def load_fixed_image(self, file, df_file=None):
        try:
            img, t, it = self.load_deformed_image(file, df_file)
        except Exception as e:
            print(e)
            return
        self.avFix.loadImage(img)
        self.transform_fixed = t
        self.inverse_transform_fixed = it
        if self.avMove.mode != 'inactive':
            self.setMode('view')

    def load_moving_image(self, file, df_file=None):
        try:
            img, t, it = self.load_deformed_image(file, df_file)
        except:
            return
        self.original_move_image = img
        self.avMove.loadImage(img)
        self.transform_moving = t
        self.inverse_transform_moving = it
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
            self.avFix.mark_updated.connect(self.updatePointList)
            self.avMove.mark_updated.connect(self.updatePointList)
            self.pbEdit.setEnabled(True)
            self.pbView.setEnabled(True)
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
        tf = [self.transform_fixed, self.transform_moving]
        text = [None, None]
        for i in range(self.tableWidget.rowCount()):
            if i in self.avFix.markList and i in self.avMove.markList:
                if self.avFix.markList[i] is not None and self.avMove.markList[i] is not None:
                    l[0].append(self.avFix.markList[i])
                    l[1].append(self.avMove.markList[i])
        for i in range(2):
            ls = l[i]
            text[i] = 'point\n'
            text[i] += str(len(ls)) + '\n'
            for pos in ls:
                pos = tf[i].TransformPoint(pos)
                text[i] += '{0:.2f} {1:.2f} {2:.2f}\n'.format(pos[0], pos[1], pos[2])
        f1 = open(path_fix, 'w')
        f1.write(text[0])
        f2 = open(path_move, 'w')
        f2.write(text[1])
        return len(l[0])

    def importPointList(self, path, fix_or_move):
        d = self.avMove
        t = self.inverse_transform_moving
        if fix_or_move == 'fix':
            d = self.avFix
            t = self.inverse_transform_fixed
        f1 = open(path)
        f1.readline()
        ct = int(f1.readline())
        self.tableWidget.setRowCount(max(self.tableWidget.rowCount(), ct))
        for i in range(ct):
            line = f1.readline()
            if len(line) < 3:
                break
            line = line.split(' ')
            pos = [float(line[0]), float(line[1]), float(line[2])]
            pos = list(t.TransformPoint(pos))
            d.appendMark(pos)
            d.mark_idx += 1
        d.paintMark()

    def save_points(self):
        d = QtWidgets.QFileDialog()
        d.setAcceptMode(QtWidgets.QFileDialog.AcceptSave)
        d.setFileMode(QtWidgets.QFileDialog.AnyFile)
        file1 = d.getSaveFileName(self, 'Export')[0]
        if len(file1) < 1:
            return
        file2 = d.getSaveFileName(self, 'Export')[0]
        if len(file2) < 1:
            return
        self.exportPointLists(file1, file2)


def execute():
    app = QtWidgets.QApplication(sys.argv)
    window = SurfaceRegistrationPage()
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    execute()