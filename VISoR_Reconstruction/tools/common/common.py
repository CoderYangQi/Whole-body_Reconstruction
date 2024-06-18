from PyQt5 import QtWidgets, QtCore, QtGui
import sys, traceback


class EmittingStream(QtCore.QObject):
    textWritten = QtCore.pyqtSignal(str)

    def write(self, text):
        self.textWritten.emit(text)


class WorkerThread(QtCore.QThread):
    progress = QtCore.pyqtSignal(int)
    status = QtCore.pyqtSignal(str)

    def __init__(self, parent=None):
        super(WorkerThread, self).__init__(parent)
        self.func = None
        self.args = []
        self.kwargs = {}
        self.text_stream = EmittingStream()

    def set_func(self, func, args: list=None, kwargs: dict=None,
                 progress_reporter: QtCore.pyqtSignal=None):
        self.func = func
        if args is not None:
            self.args = args
        if kwargs is not None:
            self.kwargs = kwargs
        if progress_reporter is not None:
            progress_reporter.connect(self.report_progress)

    def report_progress(self, value):
        self.progress.emit(value)

    def run(self):
        sys.stdout = self.text_stream
        self.status.emit('Running')
        #error = None
        try:
            self.func(*self.args, **self.kwargs)
        except Exception as e:
            exc_type, exc_value, exc_traceback = sys.exc_info()
            traceback.print_exception(exc_type, exc_value, exc_traceback)
            self.status.emit('Failed')
        self.status.emit('Finished')
        sys.stdout = sys.__stdout__
        #print(error)