from PyQt5 import QtCore, QtWidgets, QtGui

class AnnotationView(QtWidgets.QGraphicsView):
    zoomin = QtCore.pyqtSignal()
    zoomout = QtCore.pyqtSignal()
    next_i = QtCore.pyqtSignal()
    prev_i = QtCore.pyqtSignal()
    def __init__(self, *args):
        super(AnnotationView, self).__init__(*args)
        #self.shortcut_next = QtWidgets.QShortcut(QtGui.QKeySequence(QtCore.Qt.Key_Right), self)
        #self.shortcut_prev = QtWidgets.QShortcut(QtGui.QKeySequence(QtCore.Qt.Key_Left), self)

    def wheelEvent(self, QWheelEvent):
        super(AnnotationView, self).wheelEvent(QWheelEvent)
        if QWheelEvent.angleDelta().y() > 0:
            self.zoomin.emit()
        else:
            self.zoomout.emit()

    def keyPressEvent(self, QKeyEvent):
        if QKeyEvent.key() == QtCore.Qt.Key_Right:
            self.next_i.emit()
        if QKeyEvent.key() == QtCore.Qt.Key_Left:
            self.prev_i.emit()

