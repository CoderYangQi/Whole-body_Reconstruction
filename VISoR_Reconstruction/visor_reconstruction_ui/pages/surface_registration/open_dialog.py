from PyQt5 import QtWidgets
from .open_dialog_ui import Ui_Dialog
import os

class OpenDialog(QtWidgets.QDialog, Ui_Dialog):
    def __init__(self, parent=None):
        super(OpenDialog, self).__init__(parent)
        self.setupUi(self)
        self.setting_group = {'fimg': [self.line_edit_fimg, self.pb_fimg, str()],
                              'mimg': [self.line_edit_mimg, self.pb_mimg, str()],
                              'fdf': [self.line_edit_fdf, self.pb_fdf, str()],
                              'mdf': [self.line_edit_mdf, self.pb_mdf, str()],
                              'fp': [self.line_edit_fp, self.pb_fp, str()],
                              'mp': [self.line_edit_mp, self.pb_mp, str()]}
        d = lambda v: lambda: self.open_file(v)
        for k, v in self.setting_group.items():
            v[1].clicked.connect(d(v[0]))

    def open_file(self, target: QtWidgets.QLineEdit):
        d = QtWidgets.QFileDialog()
        d.setFileMode(QtWidgets.QFileDialog.ExistingFile)
        file = d.getOpenFileName(self, 'Import settings')[0]
        if len(file) == 0:
            return
        target.setText(file)

    def accept(self):
        for v in self.setting_group.values():
            v[2] = v[0].text()
        super(OpenDialog, self).accept()
