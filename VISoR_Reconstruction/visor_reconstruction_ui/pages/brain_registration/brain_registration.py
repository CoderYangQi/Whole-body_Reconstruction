from.brain_registration_ui import Ui_Form
from PyQt5 import QtWidgets
from VISoR_Reconstruction.tools.common.common import WorkerThread
from VISoR_Reconstruction.reconstruction.brain_registration import generate_freesia_input, register_brain
from VISoR_Brain.misc import ROOT_DIR
import os, json


class BrainRegistrationPage(QtWidgets.QWidget, Ui_Form):
    def __init__(self, pipeline, parent=None):
        super(BrainRegistrationPage, self).__init__(parent)
        self.setupUi(self)

        self.worker_thread = WorkerThread()
        self.worker_thread.status.connect(self.label_status.setText)
        self.pipeline = pipeline

        self.templates = {}
        self.channels = {}
        self.label_status.setText('No dataset')
        self.pb_start.setEnabled(False)
        self.pb_start.clicked.connect(self.run_brain_registration)
        for f in os.listdir(os.path.join(ROOT_DIR, 'data')):
            if f.split('.')[-1] == 'json':
                with open(os.path.join(ROOT_DIR, 'data', f)) as file:
                    d = json.load(file)
                d['file'] = f
                self.templates[d['name']] = d
                self.cb_template.addItem(d['name'])

    def update_dataset(self):
        for c in self.pipeline.dataset.channels:
            self.cb_channel.addItem(self.pipeline.dataset.channels[c]['ChannelName'])
            self.channels[self.pipeline.dataset.channels[c]['ChannelName']] = c
        if 'Reconstruction' in self.pipeline.dataset.misc:
            if 'BrainImage' in self.pipeline.dataset.misc['Reconstruction']:
                if 'BrainImage' in self.pipeline.dataset.reconstruction_info['BrainImage']:
                    self.label_status.setText('Ready')
                    self.pb_start.setEnabled(True)

    def run_brain_registration(self):
        transform_path = os.path.join(os.path.dirname(os.path.join(self.pipeline.dataset.path,
                                                      self.pipeline.dataset.misc['Reconstruction']['BrainTransform'])),
                                      'visor_brain.txt')
        output_path = os.path.join(self.pipeline.dataset.path, 'Reconstruction/BrainRegistration')
        if not os.path.exists(output_path):
            os.mkdir(output_path)
        current_template = self.templates[self.cb_template.currentText()]
        template_path = os.path.join(ROOT_DIR, 'data', current_template['file'])
        d = self.pipeline.dataset.reconstruction_info['BrainImage']['BrainImage']
        pixel_size = list(d.values())[0]['PixelSize']
        image_path = os.path.join(os.path.dirname(os.path.join(self.pipeline.dataset.path,
                                                  self.pipeline.dataset.misc['Reconstruction']['BrainImage'])),
                                  'freesia_{}_C{}_{}.json'.format(str(pixel_size), self.channels[self.cb_channel.currentText()],
                                                                  self.cb_channel.currentText()))
        self.worker_thread.set_func(register_brain,
                                    [image_path, output_path, template_path, self.pipeline.dataset.name,
                                     transform_path])
        self.worker_thread.finished.connect(self.registration_finished)
        self.worker_thread.start()
        self.pb_start.setEnabled(False)

    def registration_finished(self):
        self.pb_start.setEnabled(True)
