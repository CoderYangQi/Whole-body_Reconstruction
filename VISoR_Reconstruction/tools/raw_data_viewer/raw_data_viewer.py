from VISoR_Brain.tools.raw_data_viewer.raw_data_viewer_ui import Ui_MainWindow
import sys, os
from VISoR_Brain.positioning.visor_sample import VISoRSample
import SimpleITK as sitk
from PyQt5 import QtWidgets, QtCore
import numpy as np


class mainwindow(QtWidgets.QMainWindow, Ui_MainWindow):
    def __init__(self):
        super(mainwindow, self).__init__()
        self.setupUi(self)
        self.actionOpen.triggered.connect(self.open)
        self.actionSave_records.triggered.connect(self.save_records)
        self.actionLoad_points.triggered.connect(self.load_points)
        self.sample_data = VISoRSample()
        self.widget1.mark_updated.connect(self.change_raw_data_position)
        self.widget2.mark_updated.connect(self.change_sample_position)
        self.widget1.comboBox.setCurrentIndex(2)
        self.widget2.comboBox.setCurrentIndex(2)
        self.pb_record.clicked.connect(self.record_position)
        self.raw_pos = []

        self.pointlist = QtCore.QStringListModel()
        self.listView.setModel(self.pointlist)
        self.listView.clicked.connect(self.navigate)

    def open(self):
        d = QtWidgets.QFileDialog()
        d.setFileMode(QtWidgets.QFileDialog.ExistingFile)
        info_file = d.getOpenFileName(self, 'Select info file')[0]
        if len(info_file) == 0:
            return
        #image_file = d.getOpenFileName(self, 'Select image file')[0]
        #rawdata_file = d.getOpenFileName(self, 'Select flsm raw data file')[0]
        self.sample_data.load(info_file)
        image_file = ''
        if self.sample_data.image_source is not None:
            image_file = self.sample_data.image_source
        else:
            image_file = info_file.replace('.tar', '.mha')
            if os.path.exists(image_file):
                self.sample_data.image_origin = np.array(self.widget1.image.GetOrigin())
                self.sample_data.image_spacing = np.array(self.widget1.image.GetSpacing())
            else:
                image_file = info_file.replace('.tar', '.tif')
                self.sample_data.image_origin = np.array(self.sample_data.sphere[0])
                self.sample_data.image_spacing = np.array([4, 4, 4])
        self.sample_data.load_columns()
        self.raw_data = self.sample_data.raw_data
        # raw_data.RawData(rawdata_file, '../../devices/visor2_lowres.txt')
        self.sample_data.image = sitk.ReadImage(image_file)
        self.widget1.loadImage(self.sample_data.image)
        self.widget1.setMode('mark')
        self.widget2.setMode('mark')

        self.widget2.graphicsView.next_i.disconnect()
        self.widget2.graphicsView.prev_i.disconnect()
        self.widget2.graphicsView.next_i.connect(self.next_raw_image)
        self.widget2.graphicsView.prev_i.connect(self.prev_raw_image)

        initial_pos = (np.array(self.sample_data.image.GetSize()) / 2).tolist()
        self.widget1.addMark(initial_pos)

    def change_raw_data_position(self):
        pos = self.widget1.markList[0]
        pos = self.sample_data.get_sample_position_from_image(pos)
        self.statusbar.showMessage('%2f,%2f,%2f' % (pos[0], pos[1], pos[2]))
        pos = self.sample_data.get_column_position(pos)
        self.set_raw_data_position(pos[0], int(round(pos[1][2])))
        self.widget2.appendMark([pos[1][0], pos[1][1]])
        self.widget2.paintMark()

    def set_sample_position(self, pos):
        self.widget1.data_pos = np.int32(pos).tolist()
        self.widget1.showImage()

    def change_sample_position(self):
        try:
            pos = self.widget2.markList[0]
            pos = self.sample_data.get_slice_position(self.raw_pos[0], [pos[0], pos[1], self.raw_pos[1]])
            print(pos)
            pos = self.sample_data.get_image_position(pos)
            print(pos)
            print(self.sample_data.get_sample_position_from_image(pos))
            #pos = (np.array(pos) - self._origin) / self._spacing + 1
            self.widget1.appendMark(pos)
            self.set_sample_position(pos)
        except Exception as e:
            print(e)

    def record_position(self):
        pos = [self.raw_pos[0],
               self.widget2.markList[0][0],
               self.widget2.markList[0][1],
               self.raw_pos[1]]
        self.textBrowser.append('{0},{1},{2},{3}'.format(*pos))

    def save_records(self):
        text = str(self.textBrowser.toPlainText())
        d = QtWidgets.QFileDialog()
        d.setAcceptMode(QtWidgets.QFileDialog.AcceptSave)
        d.setFileMode(QtWidgets.QFileDialog.AnyFile)
        file = d.getSaveFileName(self, 'save records')[0]
        with open(file, 'w') as f:
            f.write(text)

    def set_raw_data_position(self, index, num):
        sphere = self.sample_data.column_images[index].sphere
        if sphere[0][2] <= num < sphere[1][2]:
            img = self.raw_data.load(index, [num, num + 1])
        else:
            img = sitk.Image(self.widget2.image.GetSize(), self.widget2.image.GetPixelIDValue())
        self.raw_pos = [index, num]
        self.widget2.loadImage(img)

    def next_raw_image(self):
        self.set_raw_data_position(self.raw_pos[0], self.raw_pos[1] + 1)

    def prev_raw_image(self):
        self.set_raw_data_position(self.raw_pos[0], self.raw_pos[1] - 1)

    def load_points(self):
        d = QtWidgets.QFileDialog()
        d.setFileMode(QtWidgets.QFileDialog.ExistingFile)
        point_list_file = d.getOpenFileName(self, 'Select point list file')[0]
        if len(point_list_file) == 0:
            return
        with open(point_list_file) as f:
            lines = f.readlines()
            self.pointlist.setStringList(lines)

    def navigate(self):
        pos = self.pointlist.data(self.listView.selectedIndexes()[0], QtCore.Qt.DisplayRole)
        pos = [float(p) for p in pos.split(',')]
        print(pos)
        self.set_raw_data_position(int(pos[0]), int(pos[3]))
        self.widget2.addMark([pos[1], pos[2]])


if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    window = mainwindow()
    window.show()
    sys.exit(app.exec_())