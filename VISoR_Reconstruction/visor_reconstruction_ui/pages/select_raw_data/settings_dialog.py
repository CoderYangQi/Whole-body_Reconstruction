from PyQt5 import QtWidgets
from .settings_dialog_ui import Ui_Dialog
from VISoR_Reconstruction.reconstruction.sample_reconstruct import get_all_methods
import json


class SettingDialog(QtWidgets.QDialog, Ui_Dialog):
    def __init__(self, parent=None):
        super(SettingDialog, self).__init__(parent)
        self.setupUi(self)
        self.settings = {}
        self.methods = [*get_all_methods()['stitch'].keys()]
        self.align_channel_methods = [*get_all_methods()['align_channels'].keys()]
        self.channels = []

        self.pb_export.clicked.connect(self.export_settings)
        self.pb_import.clicked.connect(self.import_settings)

        for s in self.methods:
            try:
                self.combo_box_method.addItem(s)
            except Exception as e:
                print(e)

        for s in self.align_channel_methods:
            try:
                self.combo_box_channel_method.addItem(s)
            except Exception as e:
                print(e)

        self.check_box_channel.stateChanged.connect(self.set_channels_enabled)

    def set_channels_enabled(self, state):
        state = state != 0
        self.combo_box_channel_method.setEnabled(state)
        self.combo_box_reference.setEnabled(state)

    def set_current(self):

        for s in self.channels:
            try:
                self.combo_box_reference.addItem(s)
            except Exception as e:
                print(e)

        for i in range(len(self.methods)):
            if self.settings['realign_method'] == self.methods[i]:
                self.combo_box_method.setCurrentIndex(i)
        for i in range(len(self.align_channel_methods)):
            if self.settings['align_channel_method'] == self.align_channel_methods[i]:
                self.combo_box_channel_method.setCurrentIndex(i)
        for i in range(len(self.channels)):
            if self.settings['reference_channel'] in self.channels:
                self.combo_box_reference.setCurrentIndex(i)
        self.line_edit_voxel_size.setText(str(self.settings['voxel_size']))
        self.check_box_thumb.setCheckState(self.settings['use_thumbnail'] * 2)
        self.check_box_channel.setCheckState(self.settings['align_channel'] * 2)
        self.set_channels_enabled(self.check_box_channel.checkState())

    def accept(self):
        self.settings['voxel_size'] = float(self.line_edit_voxel_size.text())
        self.settings['realign_method'] = self.combo_box_method.currentText()
        self.settings['use_thumbnail'] = self.check_box_thumb.checkState() != 0
        self.settings['align_channel'] = self.check_box_channel.checkState() != 0
        self.settings['align_channel_method'] = self.combo_box_channel_method.currentText()
        self.settings['reference_channel'] = self.combo_box_reference.currentText()
        super(SettingDialog, self).accept()

    def export_settings(self):
        d = QtWidgets.QFileDialog()
        d.setAcceptMode(QtWidgets.QFileDialog.AcceptSave)
        d.setFileMode(QtWidgets.QFileDialog.AnyFile)
        file = d.getSaveFileName(self, 'Export settings')[0]
        if len(file) == 0:
            return
        with open(file, 'w') as f:
            json.dump(self.settings, f)

    def import_settings(self):
        d = QtWidgets.QFileDialog()
        d.setFileMode(QtWidgets.QFileDialog.ExistingFile)
        file = d.getOpenFileName(self, 'Import settings')[0]
        if len(file) == 0:
            return
        with open(file) as f:
            try:
                s = json.load(f)
            except json.JSONDecodeError as e:
                t = e.msg
                print('Error when parsing setting file:\n' + t)
                msg = QtWidgets.QMessageBox(text='Error when parsing setting file:\n' + t)
                msg.exec()
                return
            for k in s:
                self.settings[k] = s[k]
        self.set_current()
