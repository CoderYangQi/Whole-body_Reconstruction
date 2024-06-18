from.whole_brain_reconstruct_ui import Ui_Form
from PyQt5 import QtWidgets
from VISoR_Reconstruction.tools.common.common import WorkerThread
import os, json
from VISoR_Reconstruction.reconstruction_executor.generator import gen_brain_reconstruction_pipeline, default_param
from VISoR_Reconstruction.reconstruction_executor.executor import main
from VISoR_Reconstruction.tools.common.qjsonmodel import QJsonModel
from VISoR_Reconstruction.misc import ROOT_DIR
from multiprocessing import Pipe, Process


class WholeBrainReconstructPage(QtWidgets.QWidget, Ui_Form):
    def __init__(self, pipeline, parent=None):
        super(WholeBrainReconstructPage, self).__init__(parent)
        self.setupUi(self)
        self.pb_save.clicked.connect(self.set_save_path)
        self.pb_start.clicked.connect(self.start_reconstruct)
        self.pb_settings.clicked.connect(self.settings)
        self.pb_stop.clicked.connect(self.stop_reconstruct)
        self.worker_thread = WorkerThread()
        self.worker_thread.text_stream.textWritten.connect(self.textBrowser.append)
        self.worker_thread.finished.connect(self.reconstruct_finished)
        self.param = default_param.copy()

        self.checkBox.toggled.connect(self.line_edit_save.setEnabled)
        self.checkBox.toggled.connect(self.pb_save.setEnabled)
        self.checkBox.toggle()
        self.checkBox.toggle()

        self.pipe = None

        self.pipeline = pipeline

    def set_path(self, line_edit: QtWidgets.QLineEdit):
        d = QtWidgets.QFileDialog()
        d.setFileMode(QtWidgets.QFileDialog.Directory)
        d.setOption(QtWidgets.QFileDialog.ShowDirsOnly, True)
        d.setAcceptMode(QtWidgets.QFileDialog.AcceptOpen)
        path = d.getExistingDirectory(self, 'Select Directory')
        if len(path) == 0:
            return
        line_edit.setText(path)

    def set_save_path(self):
        self.set_path(self.line_edit_save)

    def start_reconstruct(self):
        dst = self.line_edit_save.text()
        if not self.checkBox.isChecked():
            dst = self.pipeline.dataset.path
        self.param['output_path'] = dst
        self.pipe, pipe = Pipe()
        s = gen_brain_reconstruction_pipeline(self.pipeline.dataset, **self.param)
        p = Process(target=main, args=(s, pipe))
        p.start()
        self.worker_thread.set_func(self.listen, [p])
        self.worker_thread.start()
        self.pb_start.setEnabled(False)

    def listen(self, p):
        ct = 0
        while 1:
            if self.pipe.poll(1):
                try:
                    s = self.pipe.recv()
                except EOFError:
                    break
                ct += 1
                if ct == 16:
                    ct = 0
                if 'message' in s:
                    print(s['message'])
                if 'progress' in s:
                    self.progressBar.setValue(int(s['progress'] * self.progressBar.maximum()))
                if 'status' in s:
                    self.label_status.setText(s['status'])
            p.join(0.01)
            if p.exitcode is not None:
                break

    def stop_reconstruct(self):
        self.pipe.send({'stop': None})

    def reconstruct_finished(self):
        self.pb_start.setEnabled(True)

    def settings(self):
        dialog = QtWidgets.QDialog()
        dialog.setWindowTitle('Settings')
        view = QtWidgets.QTreeView(dialog)
        model = QJsonModel()
        model.load(self.param)
        view.setModel(model)

        def save_settings():
            d = QtWidgets.QFileDialog()
            d.setDirectory(os.path.join(ROOT_DIR, 'preset'))
            d.setAcceptMode(QtWidgets.QFileDialog.AcceptSave)
            d.setFileMode(QtWidgets.QFileDialog.AnyFile)
            file = d.getSaveFileName(self, 'Export settings')[0]
            if len(file) == 0:
                return
            with open(file, 'w') as fp:
                json.dump(model.json(), fp, indent=2)

        pb_save = QtWidgets.QPushButton('Save', dialog)
        pb_save.clicked.connect(save_settings)

        def load_settings():
            d = QtWidgets.QFileDialog()
            d.setDirectory(os.path.join(ROOT_DIR, 'preset'))
            d.setFileMode(QtWidgets.QFileDialog.ExistingFile)
            file = d.getOpenFileName(self, 'Import settings')[0]
            if len(file) == 0:
                return
            with open(file) as fp:
                self.param = json.load(fp)
                param = {**default_param, **self.param}
                model.load(param)

        pb_load = QtWidgets.QPushButton('Load', dialog)
        pb_load.clicked.connect(load_settings)

        dialog.resize(600, 800)
        layout = QtWidgets.QVBoxLayout(dialog)
        layout.addWidget(view)
        layout_pb = QtWidgets.QHBoxLayout(dialog)
        layout.addLayout(layout_pb)
        layout_pb.addWidget(pb_load)
        layout_pb.addWidget(pb_save)
        dialog.setLayout(layout)
        dialog.exec()
        self.param = model.json()
        for k, v in self.param.items():
            if isinstance(v, str):
                if len(v) == 0:
                    self.param[k] = None

    def update_dataset(self):
        if 'Parameters' in self.pipeline.dataset.reconstruction_info:
            self.param = {**default_param, **self.pipeline.dataset.reconstruction_info['Parameters']}
