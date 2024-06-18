from .select_raw_data_ui import Ui_widget
from PyQt5 import QtWidgets, QtCore, QtGui
import os, traceback
from VISoR_Brain.format.raw_data import RawData
from VISoR_Reconstruction.tools.common.common import WorkerThread
from VISoR_Brain.format.visor_data import VISoRData
from .settings_dialog import SettingDialog


class RawDataListModel(QtCore.QAbstractTableModel):

    def __init__(self, parent=None):
        super(RawDataListModel, self).__init__(parent)
        self.data_sheet = []
        self.colomns = ['Name', 'Index', 'Enabled', 'Status', 'Channel', 'Path']
        self.available_channels = set()
        self.raw_data_list = {}

    def load_raw_data(self, source: list):
        self.reset_data()
        for r in source:
            self.data_sheet.append([r.name, r.z_index, True, 0, r.wave_length, r.file])
            self.raw_data_list[r.file] = r
        self.data_sheet.sort(key=lambda x: int(x[1]))
        self.modelReset.emit()

    def load_visor_data(self, visor_data: VISoRData, validate: bool=True):
        self.reset_data()
        valid = True
        for c in visor_data.acquisition_results:
            for i in visor_data.acquisition_results[c]:
                r = visor_data.acquisition_results[c][i]
                status = 0
                if validate and not os.path.isfile(r):
                    status = 1
                    valid = False
                self.data_sheet.append([os.path.split(r)[1], i, True, status, visor_data.channels[c]['LaserWavelength'], r])
        self.data_sheet.sort(key=lambda x: int(x[1]))
        self.modelReset.emit()
        return valid

    def reset_data(self):
        self.data_sheet = []
        self.raw_data_list = {}

    def get_selection(self):
        return {l[5] for l in self.data_sheet if l[2]}

    def get_prop_list(self):
        return {os.path.normpath(l[5]): l for l in self.data_sheet}

    def rowCount(self, parent: QtCore.QModelIndex = ...):
        return len(self.data_sheet)

    def columnCount(self, parent: QtCore.QModelIndex = ...):
        return len(self.colomns)

    def headerData(self, section: int, orientation: QtCore.Qt.Orientation, role: int = ...):
        if role != QtCore.Qt.DisplayRole:
            return QtCore.QVariant()
        if orientation == QtCore.Qt.Horizontal:
            return self.colomns[section]
        else:
            return str(section)

    def data(self, index: QtCore.QModelIndex, role: int = ...):
        if not index.isValid():
            return QtCore.QVariant()
        value = self.data_sheet[index.row()][index.column()]
        if role == QtCore.Qt.DisplayRole and index.column() == 3:
            if value == 0:
                return 'ready'
            elif value == 1:
                return 'not found'
            else:
                return 'invalid'
        if role == QtCore.Qt.DisplayRole and index.column() != 2:
            return QtCore.QVariant(value)
        if role == QtCore.Qt.CheckStateRole and index.column() == 2:
            return int(value) * 2
        if role == QtCore.Qt.ForegroundRole:
            if index.column() == 3 and value == 1:
                return QtGui.QBrush(QtGui.QColor(255, 255, 255))
        if role == QtCore.Qt.BackgroundRole:
            if index.column() == 3 and value == 1:
                return QtGui.QBrush(QtGui.QColor(255, 0, 0))
        return QtCore.QVariant()

    def flags(self, index: QtCore.QModelIndex):
        if index.column() == 1:
            return QtCore.Qt.ItemIsEditable | QtCore.Qt.ItemIsEnabled | QtCore.Qt.ItemIsSelectable
        if index.column() == 2:
            return QtCore.Qt.ItemIsUserCheckable | QtCore.Qt.ItemIsEnabled | QtCore.Qt.ItemIsSelectable
        else:
            return QtCore.Qt.ItemIsEnabled | QtCore.Qt.ItemIsSelectable

    def setData(self, index: QtCore.QModelIndex, value, role: int = ...):
        if role == QtCore.Qt.CheckStateRole and index.column() == 2:
            if self.data_sheet[index.row()][index.column()] == True:
                self.data_sheet[index.row()][index.column()] = False
            else:
                self.data_sheet[index.row()][index.column()] = True
        if role == QtCore.Qt.EditRole and index.column() == 1:
            self.data_sheet[index.row()][index.column()] = value
            r = self.data_sheet[index.row()][5]
            if r in self.raw_data_list:
                self.raw_data_list[r].z_index = value
        return True

    def get_enabled_raw_data_list(self):
        raw_data_list = {}
        for i in range(len(self.path)):
            path = self.path[i]
            if self.enabled[i] == True:
                raw_data_list[i] = self.data_sheet[path]
        return raw_data_list

    def update_status(self, result):
        for k in result:
            self.status[k] = result[k]
        self.dataChanged.emit(self.index(0, 3), self.index(self.rowCount() - 1, 3))

    def enable_all(self):
        for i in self.data_sheet:
            i[2] = True
        self.dataChanged.emit(self.index(0, 2), self.index(self.rowCount() - 1, 2))

    def disable_all(self):
        for i in self.data_sheet:
            i[2] = False
        self.dataChanged.emit(self.index(0, 2), self.index(self.rowCount() - 1, 2))

    def check_missing_slices(self):
        sl = dict()
        out = dict()
        try:
            for i in self.data_sheet:
                if not i[4] in sl:
                    sl[i[4]] = set()
                sl[i[4]].add(int(i[1]))
            imin, imax = min([min(sl[i]) for i in sl]), max([max(sl[i]) for i in sl])
            for c in sl:
                for i in range(imin, imax + 1):
                    if i not in sl[c]:
                        if c not in out:
                            out[c] = set()
                        out[c].add(i)
        except Exception:
            pass
        return out


class RawDataLoader(QtCore.QObject):

    progress = QtCore.pyqtSignal(int)
    status = QtCore.pyqtSignal(str)

    def load_raw_data(self, path):
        ct = 0
        self.progress.emit(0)
        self.raw_data_list = []
        file_list = []
        for root, dirs, files in os.walk(path):
            for f in files:
                if f.split('.')[-1] == 'flsm' and os.path.isfile(os.path.join(root, f)):
                    file_list.append(os.path.join(root, f))
        for file in file_list:
            try:
                r = RawData(file)
                self.raw_data_list.append(r)
                print('pass:{0}'.format(file))
            except Exception as e:
                print(e)
                print('fail:{0}'.format(file))
            finally:
                ct += 1
                self.progress.emit(100 * (ct / len(file_list)))


class SelectRawDataPage(QtWidgets.QWidget, Ui_widget):
    def __init__(self, pipeline, parent=None):
        super(SelectRawDataPage, self).__init__(parent)
        self.setupUi(self)
        self.pb_1.clicked.connect(self.set_dataset_path)
        self.pb_2.clicked.connect(self.set_folder_path)
        self.pb_load.clicked.connect(self.load_dataset)
        self.pb_settings.clicked.connect(self.open_setting)
        self.pb_create.clicked.connect(self.create_dataset)
        self.pb_create.setEnabled(False)
        self.pb_use.clicked.connect(self.use_dataset)
        self.pb_use.setEnabled(False)
        self.pb_checkmissing.clicked.connect(self.check_missing_slices)
        self.pb_checkmissing.setEnabled(False)
        self.loader = RawDataLoader()
        self.data_list = RawDataListModel(self.tableView)
        self.tableView.setModel(self.data_list)

        self.radioButton.toggled.connect(self.line_edit_1.setEnabled)
        self.radioButton.toggled.connect(self.pb_1.setEnabled)
        self.radioButton_2.toggled.connect(self.line_edit_2.setEnabled)
        self.radioButton_2.toggled.connect(self.pb_2.setEnabled)

        self.radioButton_2.toggle()
        self.radioButton.toggle()

        self.pb_select_all.clicked.connect(self.data_list.enable_all)
        self.pb_select_none.clicked.connect(self.data_list.disable_all)

        self.worker_thread = WorkerThread()
        self.worker_thread.progress.connect(self.progressBar.setValue)
        self.worker_thread.text_stream.textWritten.connect(self.textBrowser.append)

        self.pipeline = pipeline
        self.dataset = None

        self.settings = {'voxel_size': 4.0,
                         'realign_method': 'elastix_align',
                         'reconstruct_image': True,
                         'save_image_as_blocks': False,
                         'block_size': 2048,
                         'use_thumbnail': True,
                         'save_to_raw_data': False,
                         'align_channel': False,
                         'align_channel_method': 'channel_elastix_align',
                         'reference_channel': '488'}

    def set_dataset_path(self):
        d = QtWidgets.QFileDialog()
        d.setFileMode(QtWidgets.QFileDialog.ExistingFile)
        d.setAcceptMode(QtWidgets.QFileDialog.AcceptOpen)
        path = d.getOpenFileName(self, 'Select File')
        if len(path[0]) == 0:
            return
        self.line_edit_1.setText(path[0])

    def set_folder_path(self):
        d = QtWidgets.QFileDialog()
        d.setFileMode(QtWidgets.QFileDialog.Directory)
        d.setOption(QtWidgets.QFileDialog.ShowDirsOnly, True)
        d.setAcceptMode(QtWidgets.QFileDialog.AcceptOpen)
        path = d.getExistingDirectory(self, 'Select Directory')
        if len(path) == 0:
            return
        self.line_edit_2.setText(path)

    def load_dataset(self):
        if self.radioButton_2.isChecked():
            self.worker_thread.set_func(self.loader.load_raw_data,
                                        [self.line_edit_2.text()], progress_reporter=self.loader.progress)

            self.worker_thread.finished.connect(self.finish_load_raw_data)
            self.worker_thread.start()
            self.label_status.setText('Loading...')
            self.pb_load.setEnabled(False)
            self.pb_create.setEnabled(False)
            self.pb_checkmissing.setEnabled(False)
        else:
            try:
                self.dataset = VISoRData(self.line_edit_1.text())
            except Exception as e:
                traceback.print_exception(type(e), e, e.__traceback__)
                self.label_status.setText('Invalid dataset file')
                return
            if not self.data_list.load_visor_data(self.dataset):
                b = QtWidgets.QMessageBox(QtWidgets.QMessageBox.Warning, 'Warning', 'Cannot find some of raw data')
                b.exec()
            self.pb_create.setEnabled(True)
            self.pb_use.setEnabled(True)
            self.pb_checkmissing.setEnabled(True)

    def finish_load_raw_data(self):
        self.data_list.load_raw_data(self.loader.raw_data_list)
        self.label_status.setText('Ready')
        self.pb_load.setEnabled(True)
        if self.data_list.rowCount() > 0:
            self.pb_create.setEnabled(True)

    def create_dataset(self):
        d = QtWidgets.QFileDialog()
        if self.radioButton_2.isChecked():
            d.setDirectory(self.line_edit_2.text())
        else:
            d.setDirectory(self.dataset.path)
        d.setAcceptMode(QtWidgets.QFileDialog.AcceptSave)
        d.setFileMode(QtWidgets.QFileDialog.AnyFile)
        file = d.getSaveFileName(self, 'Export settings')[0]
        if len(file) == 0:
            return

        selection, prop_list = self.data_list.get_selection(), self.data_list.get_prop_list()
        if len(self.data_list.raw_data_list) == 0:
            for c in self.dataset.acquisition_results:
                pop_list = []
                for i, r in self.dataset.acquisition_results[c].items():
                    if r not in selection:
                        pop_list.append(i)
                for i in pop_list:
                    self.dataset.acquisition_results[c].pop(i)
        else:
            self.dataset = VISoRData(raw_data_list=[self.data_list.raw_data_list[p] for p in selection])
        aq = {}
        for c in self.dataset.acquisition_results:
            aq[c] = {}
            for i, r in self.dataset.acquisition_results[c].items():
                print(prop_list.keys())
                aq[c][prop_list[os.path.normpath(r)][1]] = r
        self.dataset.acquisition_results = aq
        self.data_list.load_visor_data(self.dataset)
        self.dataset.save(file)
        self.pb_use.setEnabled(True)

    def use_dataset(self):
        self.pipeline.set_dataset(self.dataset)

    def open_setting(self):
        dialog = SettingDialog()
        dialog.settings = self.settings
        dialog.channels = self.data_list.available_channels
        dialog.set_current()
        dialog.exec()

    def check_missing_slices(self):
        sl = self.data_list.check_missing_slices()
        for c in sl:
            self.textBrowser.append('{}:\n'.format(c))
            for i in sl[c]:
                self.textBrowser.append(str(i) + '\n')
