import SimpleITK as sitk
import numpy as np
from PyQt5 import QtCore, QtWidgets, QtGui

from .annotation_widget_ui import Ui_Form


class VirtualCursor(QtWidgets.QGraphicsItem):
    def __init__(self, *args):
        super(VirtualCursor, self).__init__(*args)
        self.pen = QtGui.QPen()
        self.pen.setColor(QtGui.QColor(255, 0, 0))

    def paint(self, QPainter, QStyleOptionGraphicsItem, widget=None):
        QPainter.setPen(self.pen)
        QPainter.save()
        s = self.scale() / QPainter.transform().m11()
        QPainter.scale(s, s)
        QPainter.drawLine(0, -7, 0, -2)
        QPainter.drawLine(0, 2, 0, 7)
        QPainter.drawLine(-7, 0, -2, 0)
        QPainter.drawLine(2, 0, 7, 0)
        QPainter.restore()

    def boundingRect(self):
        return QtCore.QRectF(-10, -10, 20, 20)


class markItem(QtWidgets.QGraphicsItem):
    def __init__(self, *args):
        super(markItem, self).__init__(*args)
        self.text = ''
        self.idx = None
        self.pen = QtGui.QPen()
        self.pen.setColor(QtGui.QColor(0, 255, 0))

    def paint(self, QPainter, QStyleOptionGraphicsItem, widget=None):
        QPainter.setPen(self.pen)
        QPainter.save()
        s = self.scale() / QPainter.transform().m11()
        QPainter.scale(s, s)
        QPainter.drawLine(0, -5, 0, 5)
        QPainter.drawLine(-5, 0, 5, 0)
        QPainter.drawText(5, 0, self.text)
        QPainter.restore()

    def boundingRect(self):
        return QtCore.QRectF(-5, -15, 20, 20)

class AnnotationScene(QtWidgets.QGraphicsScene):
    mark = QtCore.pyqtSignal(list)
    pos_changed = QtCore.pyqtSignal(list)
    pos_changed_info = QtCore.pyqtSignal(str)
    def __init__(self, *args):
        super(AnnotationScene, self).__init__(*args)

    def mouseReleaseEvent(self, QGraphicsSceneMouseEvent):
        pos = QGraphicsSceneMouseEvent.scenePos()
        pos = [pos.x(), pos.y()]
        self.mark.emit(pos)

    def mouseMoveEvent(self, QGraphicsSceneMouseEvent):
        pos = QGraphicsSceneMouseEvent.scenePos()
        tr = '{0},{1}'.format(pos.x(), pos.y())
        self.pos_changed_info.emit(tr)
        self.pos_changed.emit([pos.x(), pos.y()])


class AnnotationWidget(QtWidgets.QWidget, Ui_Form):
    mark_updated = QtCore.pyqtSignal()
    def __init__(self, *args):
        super(AnnotationWidget, self).__init__(*args)
        self.setupUi(self)
        self.image = None
        self.data_pos = [0, 0, 0]
        self.scene = AnnotationScene()
        self.marks = []
        self.sync_cursor = VirtualCursor()
        self.graphicsView.setScene(self.scene)
        self.comboBox.currentIndexChanged.connect(self.showImage)
        self.scaleLevel = 0
        self.dataMaximium = 255
        self.horizontalSlider.setMaximum(255)
        self.horizontalSlider.valueChanged.connect(self.showImage)
        self.horizontalSlider.valueChanged.connect(self.show_contrast)
        self.scene.pos_changed.connect(self.removeSyncCursor)
        self.checkBox.stateChanged.connect(self.showImage)
        self.graphicsView.zoomin.connect(self.zoomIn)
        self.graphicsView.zoomout.connect(self.zoomOut)
        self.markList = {}
        self.mark_idx = 0
        self.markDict = [{}, {}, {}]
        self.mode = 'inactive'
        self.show_index = True

    def paintSyncCursor(self, pos):
        if self.sync_cursor.scene() != self.scene:
            self.scene.addItem(self.sync_cursor)
        self.sync_cursor.setPos(pos[0], pos[1])

    def removeSyncCursor(self):
        if self.sync_cursor.scene() == self.scene:
            self.scene.removeItem(self.sync_cursor)

    def setMode(self, mode):
        if mode == self.mode:
            return
        if self.mode == 'inactive':
            self.lineEdit_2.editingFinished.connect(self.setPosByText)
            self.graphicsView.next_i.connect(self.nextImage)
            self.graphicsView.prev_i.connect(self.prevImage)
            self.horizontalSlider_2.valueChanged.connect(self.set_image_position)
        if self.mode == 'mark':
            self.scene.mark.disconnect()
            self.unsetCursor()
        if mode == 'view':
            self.mode = mode
            self.graphicsView.setDragMode(QtWidgets.QGraphicsView.ScrollHandDrag)
        if mode == 'mark':
            self.mode = mode
            self.scene.mark.connect(self.addMark)
            self.graphicsView.setDragMode(QtWidgets.QGraphicsView.NoDrag)
            self.graphicsView.setCursor(QtCore.Qt.CrossCursor)

    def show_contrast(self):
        if self.checkBox.isChecked():
            contrast = 255 / np.log(self.dataMaximium) / (1 - float(self.horizontalSlider.value() - 1) / 255)
        else:
            contrast = self.dataMaximium / np.square(255 / self.horizontalSlider.value())
        self.lineEdit.setText(str(int(contrast)))

    def showImage(self):
        if self.image is None:
            return
        idx = self.comboBox.currentIndex()
        plane_pos = [0, 0, 0]
        plane_pos[idx] = self.data_pos[idx]
        plane_size = list(self.image.GetSize())
        plane_size[idx] = 0
        plane = sitk.Extract(self.image, plane_size, plane_pos)
        plane = sitk.GetArrayFromImage(plane)
        if self.checkBox.isChecked():
            plane = np.log(np.float32(plane))
            plane -= np.log(self.dataMaximium) * self.horizontalSlider.value() / 255
            plane *= 255 / np.log(self.dataMaximium) / (1 - float(self.horizontalSlider.value() - 1) / 255)
        else:
            plane = plane * (np.square(255 / self.horizontalSlider.value()) * 255 / self.dataMaximium)
        plane = np.clip(plane, 0, 255)
        plane = np.uint8(plane)
        im = QtGui.QImage(plane.data,
                          plane.shape[1],
                          plane.shape[0],
                          plane.shape[1],
                          QtGui.QImage.Format_Grayscale8).copy()
        im = QtGui.QPixmap.fromImage(im)
        self.label.setText('/' + str(self.image.GetSize()[idx]))
        self.lineEdit_2.setText(str(self.data_pos[idx]))
        self.scene.removeItem(self.sync_cursor)
        self.scene.clear()
        self.marks.clear()
        self.scene.addPixmap(im)
        self.paintMark()
        self.graphicsView.repaint()
        self.refresh_z_slider()

    def loadImage(self, img):
        self.image = img
        if self.image.GetPixelIDValue() == sitk.sitkUInt16:
            self.dataMaximium = 65535
        if self.image.GetPixelIDValue() == sitk.sitkUInt8:
            self.dataMaximium = 255
        self.data_pos[0] = int(self.image.GetSize()[0] / 2)
        self.data_pos[1] = int(self.image.GetSize()[1] / 2)
        self.data_pos[2] = int(self.image.GetSize()[2] / 2)
        self.showImage()
        if self.mode == 'inactive':
            self.setMode('view')

    def refresh_z_slider(self):
        if self.image.GetSize()[self.comboBox.currentIndex()] - 1 != self.horizontalSlider_2.maximum():
            self.horizontalSlider_2.setMaximum(self.image.GetSize()[self.comboBox.currentIndex()] - 1)
        if self.data_pos[self.comboBox.currentIndex()] != self.horizontalSlider_2.value():
            self.horizontalSlider_2.setValue(self.data_pos[self.comboBox.currentIndex()])

    def zoomIn(self):
        if self.scaleLevel < 6:
            self.scaleLevel += 1
            self.graphicsView.scale(1.25, 1.25)
            self.label_2.setText(str(np.power(1.25, self.scaleLevel) * 100) + '%')

    def zoomOut(self):
        if self.scaleLevel > -8:
            self.scaleLevel -= 1
            self.graphicsView.scale(0.8, 0.8)
            self.label_2.setText(str(np.power(1.25, self.scaleLevel) * 100) + '%')

    def setPosByText(self):
        idx = self.comboBox.currentIndex()
        pos = self.lineEdit_2.text()
        if not pos.isdigit():
            self.lineEdit_2.setText(str(self.data_pos[idx]))
            return
        pos = int(pos)
        if pos >= self.image.GetSize()[idx]:
            self.lineEdit_2.setText(str(self.data_pos[idx]))
            return
        self.data_pos[idx] = pos
        self.showImage()

    def nextImage(self):
        idx = self.comboBox.currentIndex()
        if self.data_pos[idx] < self.image.GetSize()[idx] - 1:
            self.data_pos[idx] = self.data_pos[idx] + 1
            self.showImage()

    def prevImage(self):
        idx = self.comboBox.currentIndex()
        if self.data_pos[idx] > 0:
            self.data_pos[idx] = self.data_pos[idx] - 1
            self.showImage()

    def set_image_position(self, i):
        idx = self.comboBox.currentIndex()
        if self.data_pos[idx] < 0:
            i = 0
        if self.data_pos[idx] > self.image.GetSize()[idx] - 1:
            i = self.image.GetSize()[idx] - 1
        self.data_pos[idx] = i
        self.showImage()

    def addMark(self, pos):
        self.appendMark(pos)
        self.paintMark()
        self.mark_updated.emit()

    def appendMark(self, pos):
        idx = self.comboBox.currentIndex()
        if len(pos) == 2:
            pos.insert(idx, float(self.data_pos[idx]))
        self.markList[self.mark_idx] = pos
        for i in range(3):
            if not int(pos[i]) in self.markDict[i]:
                self.markDict[i][int(pos[i])] = {}
            self.markDict[i][int(pos[i])][self.mark_idx] = pos

    def clearMarks(self):
        self.markList = []
        self.markDict = [{}, {}, {}]

    def paintMark(self):
        for item in self.marks:
            self.scene.removeItem(item)
        idx = self.comboBox.currentIndex()
        if not self.data_pos[idx] in self.markDict[idx]:
            return
        for i, p in self.markDict[idx][self.data_pos[idx]].items():
            p = p.copy()
            del p[idx]
            mark_item = markItem()
            mark_item.idx = i
            if self.show_index:
                mark_item.text = str(i)
            mark_item.setPos(p[0], p[1])
            self.marks.append(mark_item)
            self.scene.addItem(mark_item)

    def setMarkIdx(self, idx):
        self.mark_idx = idx

if __name__ == '__main__':
    import sys
    app = QtWidgets.QApplication(sys.argv)
    window = AnnotationWidget()
    window.show()
    sys.exit(app.exec_())