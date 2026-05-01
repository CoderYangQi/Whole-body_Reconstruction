from .roi_reconstruction_ui import Ui_Form
from VISoR_Reconstruction.reconstruction_executor.roi_reconstruction_generator import \
    gen_roi_reconstruction_pipeline, default_param
from VISoR_Reconstruction.tools.common.common import WorkerThread
from VISoR_Reconstruction.reconstruction_executor.executor import main
from VISoR_Reconstruction.tools.common.qjsonmodel import QJsonModel
from VISoR_Brain.positioning.visor_brain import VISoRBrain

from PyQt5 import QtWidgets
from multiprocessing import Pipe, Process
import os, json


class ROIReconsrtuctionPage(QtWidgets.QWidget, Ui_Form):
    def __init__(self, pipeline, parent=None):
        super(ROIReconsrtuctionPage, self).__init__(parent)
        self.setupUi(self)
        self.worker_thread = WorkerThread()
        self.worker_thread.finished.connect(self.reconstruct_finished)
        self.pb_start.clicked.connect(self.start_reconstruct)
        self.pb_settings.clicked.connect(self.settings)
        self.param = default_param.copy()
        self.pipeline = pipeline
        self.l_tlx.textChanged.connect(self.set_roi)
        self.l_tly.textChanged.connect(self.set_roi)
        self.l_tlz.textChanged.connect(self.set_roi)
        self.l_brx.textChanged.connect(self.set_roi)
        self.l_bry.textChanged.connect(self.set_roi)
        self.l_brz.textChanged.connect(self.set_roi)

        self.pipe = None
        
    def set_path(self):
        d = QtWidgets.QFileDialog()
        d.setFileMode(QtWidgets.QFileDialog.Directory)
        d.setOption(QtWidgets.QFileDialog.ShowDirsOnly, True)
        d.setAcceptMode(QtWidgets.QFileDialog.AcceptOpen)
        path = d.getExistingDirectory(self, 'Select Directory')
        if len(path) == 0:
            return
        self.l_path.setText(path)
        
    def start_reconstruct(self):
        dst = self.l_path.text()
        self.param['output_path'] = dst
        self.pipe, pipe = Pipe()
        s = gen_roi_reconstruction_pipeline(self.pipeline.dataset, **self.param)
        p = Process(target=main, args=(s, pipe))
        p.start()
        self.worker_thread.set_func(self.listen, [p])
        self.worker_thread.start()
        self.pb_start.setEnabled(False)

    def listen(self, p):
        while 1:
            if self.pipe.poll(1):
                try:
                    s = self.pipe.recv()
                except EOFError:
                    break
                if 'message' in s:
                    print(s['message'])
                if 'progress' in s:
                    self.progressBar.setValue(int(s['progress'] * self.progressBar.maximum()))
                if 'status' in s:
                    self.label_status.setText(s['status'])
            p.join(0.001)
            if p.exitcode is not None:
                break

    def stop_reconstruct(self):
        self.pipe.send({'stop': None})

    def reconstruct_finished(self):
        self.pb_start.setEnabled(True)

    def set_roi(self):
        try:
            roi = [[float(self.l_tlx.text()), float(self.l_tly.text()), float(self.l_tlz.text())],
                   [float(self.l_brx.text()), float(self.l_bry.text()), float(self.l_brz.text())]]
            self.param['roi'] = roi
        except ValueError:
            pass

    def show_roi(self):
        roi = self.param['roi']
        self.l_tlx.setText(str(roi[0][0]))
        self.l_tly.setText(str(roi[0][1]))
        self.l_tlz.setText(str(roi[0][2]))
        self.l_brx.setText(str(roi[1][0]))
        self.l_bry.setText(str(roi[1][1]))
        self.l_brz.setText(str(roi[1][2]))

    def settings(self):
        dialog = QtWidgets.QDialog()
        dialog.setWindowTitle('Settings')
        view = QtWidgets.QTreeView(dialog)
        model = QJsonModel()
        model.load(self.param)
        view.setModel(model)

        def save_settings():
            d = QtWidgets.QFileDialog()
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
        self.show_roi()
        for k, v in self.param.items():
            if isinstance(v, str):
                if len(v) == 0:
                    self.param[k] = None

    def update_dataset(self):
        b = VISoRBrain(self.pipeline.dataset.brain_transform)
        self.param['roi'] = b.sphere
        self.show_roi()
        path = os.path.join(self.pipeline.dataset.path, 'Analysis')
        ct = 1
        while os.path.exists(os.path.join(path, self.param['name'])):
            self.param['name'] = 'ROIReconstruction_{}'.format(ct)
            ct += 1
        self.l_path.setText(path)

