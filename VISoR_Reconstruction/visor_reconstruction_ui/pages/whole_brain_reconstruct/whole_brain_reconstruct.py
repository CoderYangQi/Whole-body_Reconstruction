from.whole_brain_reconstruct_ui import Ui_Form
from PyQt5 import QtWidgets, QtCore, QtGui
from VISoR_Reconstruction.tools.common.common import WorkerThread
import os, json
from VISoR_Reconstruction.reconstruction_executor.generator import gen_brain_reconstruction_pipeline, default_param
from VISoR_Reconstruction.reconstruction_executor.executor import main
from VISoR_Reconstruction.reconstruction_executor.b85_runner import (
    B85_STEP_NAMES,
    infer_b85_config,
    main as b85_main,
)
from VISoR_Reconstruction.tools.common.qjsonmodel import QJsonModel
from VISoR_Reconstruction.misc import ROOT_DIR
from multiprocessing import Pipe, Process


class WholeBrainReconstructPage(QtWidgets.QWidget, Ui_Form):
    pipe_message_received = QtCore.pyqtSignal(dict)

    def __init__(self, pipeline, parent=None):
        super(WholeBrainReconstructPage, self).__init__(parent)
        self.setupUi(self)
        self.pb_save.clicked.connect(self.set_save_path)
        self.pb_start.clicked.connect(self.start_reconstruct)
        self.pb_settings.clicked.connect(self.settings)
        self.pb_stop.clicked.connect(self.stop_reconstruct)
        self.worker_thread = WorkerThread()
        self.worker_thread.text_stream.textWritten.connect(self._append_log)
        self.worker_thread.finished.connect(self.reconstruct_finished)
        self.param = default_param.copy()

        self.checkBox.toggled.connect(self.line_edit_save.setEnabled)
        self.checkBox.toggled.connect(self.pb_save.setEnabled)
        self.checkBox.toggle()
        self.checkBox.toggle()

        self.pipe = None
        self.process = None
        self.process_exit_code = None
        self.latest_b85_result = None
        self.pipe_message_received.connect(self._handle_pipe_message)

        self.pipeline = pipeline
        self._setup_page_chrome()
        self._setup_b85_controls()
        self._set_b85_controls_enabled(False)

    def _append_log(self, text):
        if text is None:
            return
        text = str(text)
        if len(text) == 0:
            return
        text = text.replace('\r\n', '\n').replace('\r', '\n')
        lines = text.split('\n')
        cursor = self.textBrowser.textCursor()
        cursor.movePosition(QtGui.QTextCursor.End)
        for line in lines:
            if len(line) == 0:
                continue
            if not self.textBrowser.document().isEmpty():
                cursor.insertBlock()
            cursor.insertText(line)
        self.textBrowser.setTextCursor(cursor)
        self.textBrowser.ensureCursorVisible()

    def _setup_page_chrome(self):
        self.setObjectName('wholeBrainReconstructPage')
        self.verticalLayout.setContentsMargins(16, 16, 16, 16)
        self.verticalLayout.setSpacing(14)
        self.gridLayout.setHorizontalSpacing(10)
        self.gridLayout.setVerticalSpacing(10)
        self.horizontalLayout.setSpacing(10)

        fixed_font = QtGui.QFontDatabase.systemFont(QtGui.QFontDatabase.FixedFont)
        fixed_font.setPointSize(10)
        self.textBrowser.setObjectName('logBrowser')
        self.textBrowser.setFont(fixed_font)
        self.textBrowser.setLineWrapMode(QtWidgets.QTextEdit.NoWrap)
        self.textBrowser.setMinimumHeight(260)

        self.label_status.setObjectName('statusLabel')
        self.label_status.setText('Idle')
        self.label_status.setAlignment(QtCore.Qt.AlignCenter)
        self.label_status.setMinimumWidth(175)
        self.progressBar.setFixedHeight(22)
        self.progressBar.setTextVisible(True)
        self.progressBar.setFormat('%p%')

        style = self.style()
        self.pb_start.setIcon(style.standardIcon(QtWidgets.QStyle.SP_MediaPlay))
        self.pb_stop.setIcon(style.standardIcon(QtWidgets.QStyle.SP_MediaStop))
        self.pb_settings.setIcon(style.standardIcon(QtWidgets.QStyle.SP_FileDialogDetailedView))
        self.pb_save.setIcon(style.standardIcon(QtWidgets.QStyle.SP_DirOpenIcon))
        self.pb_start.setProperty('role', 'primary')
        self.pb_stop.setProperty('role', 'danger')
        self.pb_settings.setProperty('role', 'secondary')
        for button in [self.pb_start, self.pb_stop, self.pb_settings, self.pb_save]:
            button.setCursor(QtCore.Qt.PointingHandCursor)
            button.setMinimumHeight(36)
        self.pb_stop.setEnabled(False)
        self.line_edit_save.setPlaceholderText('Dataset directory')

        self.setStyleSheet("""
            QWidget#wholeBrainReconstructPage {
                background: #f5f7fb;
                color: #1f2937;
                font-family: "Microsoft YaHei UI", "Segoe UI";
                font-size: 13px;
            }
            QLabel {
                color: #344054;
            }
            QLabel#sectionLabel {
                color: #475467;
                font-size: 12px;
                font-weight: 600;
                padding-top: 8px;
                padding-bottom: 2px;
            }
            QLabel#statusLabel {
                background: #eef4ff;
                border: 1px solid #c7d7fe;
                border-radius: 9px;
                color: #1d4ed8;
                font-weight: 600;
                padding: 4px 12px;
            }
            QFrame#modeFrame {
                background: #ffffff;
                border: 1px solid #d9e2ec;
                border-radius: 6px;
            }
            QGroupBox {
                background: #ffffff;
                border: 1px solid #d9e2ec;
                border-radius: 6px;
                margin-top: 16px;
                padding: 16px 12px 12px 12px;
                font-weight: 600;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 4px;
                color: #243b53;
                font-size: 13px;
            }
            QLineEdit, QComboBox, QSpinBox, QDoubleSpinBox {
                background: #ffffff;
                border: 1px solid #c8d3df;
                border-radius: 5px;
                min-height: 32px;
                padding: 5px 8px;
                selection-background-color: #2563eb;
            }
            QListWidget {
                background: #ffffff;
                border: 1px solid #c8d3df;
                border-radius: 5px;
                padding: 5px 8px;
                selection-background-color: #2563eb;
            }
            QListWidget::item {
                min-height: 26px;
                padding: 3px 4px;
            }
            QLineEdit:focus, QComboBox:focus, QSpinBox:focus, QDoubleSpinBox:focus, QListWidget:focus {
                border-color: #2563eb;
            }
            QPushButton {
                background: #ffffff;
                border: 1px solid #c8d3df;
                border-radius: 5px;
                min-height: 34px;
                padding: 7px 14px;
                color: #243b53;
            }
            QPushButton:hover {
                background: #f0f4f8;
            }
            QPushButton:disabled {
                background: #eef2f6;
                color: #9aa5b1;
            }
            QPushButton[role="primary"] {
                background: #2563eb;
                border-color: #1d4ed8;
                color: #ffffff;
                font-weight: 600;
            }
            QPushButton[role="primary"]:hover {
                background: #1d4ed8;
            }
            QPushButton[role="primary"]:disabled {
                background: #b8c7e0;
                border-color: #b8c7e0;
                color: #f8fafc;
            }
            QPushButton[role="danger"] {
                background: #fff5f5;
                border-color: #f1b9b4;
                color: #b42318;
                font-weight: 600;
            }
            QPushButton[role="danger"]:disabled {
                background: #f3f6fa;
                border-color: #d8e0ea;
                color: #9aa5b1;
            }
            QPushButton[role="secondary"] {
                background: #f8fafc;
            }
            QProgressBar {
                background: #e5eaf0;
                border: 1px solid #c8d3df;
                border-radius: 11px;
                color: #243b53;
                text-align: center;
            }
            QProgressBar::chunk {
                background: #2f9e66;
                border-radius: 10px;
            }
            QTextBrowser#logBrowser {
                background: #101828;
                border: 1px solid #1d2939;
                border-radius: 6px;
                color: #d0d5dd;
                padding: 8px;
            }
            QScrollArea#refinementScroll {
                background: transparent;
                border: none;
            }
        """)

    def _section_label(self, text):
        label = QtWidgets.QLabel(text, self.b85_group)
        label.setObjectName('sectionLabel')
        return label

    def _path_row_widget(self, line_edit, button):
        widget = QtWidgets.QWidget(self.b85_group)
        layout = QtWidgets.QHBoxLayout(widget)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)
        line_edit.setMinimumWidth(360)
        button.setMinimumWidth(112)
        layout.addWidget(line_edit, 1)
        layout.addWidget(button)
        return widget

    def _slice_range_widget(self):
        widget = QtWidgets.QWidget(self.b85_group)
        layout = QtWidgets.QHBoxLayout(widget)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)
        start_label = QtWidgets.QLabel('Start', self.b85_group)
        end_label = QtWidgets.QLabel('End exclusive', self.b85_group)
        self.b85_start_slice.setMinimumWidth(130)
        self.b85_end_slice.setMinimumWidth(130)
        layout.addWidget(start_label)
        layout.addWidget(self.b85_start_slice)
        layout.addSpacing(10)
        layout.addWidget(end_label)
        layout.addWidget(self.b85_end_slice)
        layout.addStretch(1)
        return widget

    def _form_row_widget(self, label_text, widget, label_width=150):
        row = QtWidgets.QWidget(self.b85_group)
        layout = QtWidgets.QHBoxLayout(row)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(10)
        label = QtWidgets.QLabel(label_text, self.b85_group)
        label.setMinimumWidth(label_width)
        label.setAlignment(QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter)
        widget.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Fixed)
        layout.addWidget(label)
        layout.addWidget(widget, 1)
        return row

    def _numeric_parameters_widget(self):
        widget = QtWidgets.QWidget(self.b85_group)
        layout = QtWidgets.QGridLayout(widget)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setHorizontalSpacing(12)
        layout.setVerticalSpacing(10)
        controls = [
            ('Pixel size', self.b85_pixel_size),
            ('Block size', self.b85_block_size),
            ('Ref size gap', self.b85_gap),
        ]
        for row, (label_text, control) in enumerate(controls):
            label = QtWidgets.QLabel(label_text, self.b85_group)
            label.setMinimumWidth(150)
            label.setAlignment(QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter)
            control.setMinimumWidth(160)
            control.setMaximumWidth(220)
            layout.addWidget(label, row, 0)
            layout.addWidget(control, row, 1)
            layout.setRowMinimumHeight(row, 38)
        layout.setColumnStretch(2, 1)
        return widget

    def _setup_b85_controls(self):
        mode_frame = QtWidgets.QFrame(self)
        mode_frame.setObjectName('modeFrame')
        mode_layout = QtWidgets.QHBoxLayout(mode_frame)
        mode_layout.setContentsMargins(12, 10, 12, 10)
        mode_layout.setSpacing(10)
        mode_label = QtWidgets.QLabel('Mode', mode_frame)
        self.cb_mode = QtWidgets.QComboBox(self)
        self.cb_mode.addItem('Standard VISoR', 'standard')
        self.cb_mode.addItem('Refinement', 'b85')
        mode_layout.addWidget(mode_label)
        mode_layout.addWidget(self.cb_mode)
        mode_layout.addStretch(1)
        self.verticalLayout.insertWidget(0, mode_frame)

        self.b85_scroll = QtWidgets.QScrollArea(self)
        self.b85_scroll.setObjectName('refinementScroll')
        self.b85_scroll.setWidgetResizable(True)
        self.b85_scroll.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAsNeeded)
        self.b85_scroll.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAsNeeded)
        self.b85_group = QtWidgets.QGroupBox('Refinement', self.b85_scroll)
        b85_layout = QtWidgets.QVBoxLayout(self.b85_group)
        b85_layout.setContentsMargins(14, 16, 14, 14)
        b85_layout.setSpacing(10)

        self.b85_output_root = QtWidgets.QLineEdit(self.b85_group)
        self.b85_temp_root = QtWidgets.QLineEdit(self.b85_group)
        self.b85_pb_output = QtWidgets.QPushButton('Browse...', self.b85_group)
        self.b85_pb_temp = QtWidgets.QPushButton('Browse...', self.b85_group)
        self.b85_pb_output.setIcon(self.style().standardIcon(QtWidgets.QStyle.SP_DirOpenIcon))
        self.b85_pb_temp.setIcon(self.style().standardIcon(QtWidgets.QStyle.SP_DirOpenIcon))
        self.b85_pb_output.setCursor(QtCore.Qt.PointingHandCursor)
        self.b85_pb_temp.setCursor(QtCore.Qt.PointingHandCursor)
        self.b85_pb_output.setMinimumHeight(36)
        self.b85_pb_temp.setMinimumHeight(36)
        self.b85_reference_channel = QtWidgets.QComboBox(self.b85_group)
        self.b85_reference_channel.setMinimumWidth(260)
        self.b85_output_channels = QtWidgets.QListWidget(self.b85_group)
        self.b85_output_channels.setMinimumHeight(132)
        self.b85_output_channels.setMaximumHeight(190)

        self.b85_start_slice = QtWidgets.QSpinBox(self.b85_group)
        self.b85_start_slice.setMaximum(99999)
        self.b85_end_slice = QtWidgets.QSpinBox(self.b85_group)
        self.b85_end_slice.setMaximum(99999)
        self.b85_pixel_size = QtWidgets.QDoubleSpinBox(self.b85_group)
        self.b85_pixel_size.setRange(0.1, 100.0)
        self.b85_pixel_size.setDecimals(2)
        self.b85_pixel_size.setValue(4.0)
        self.b85_block_size = QtWidgets.QSpinBox(self.b85_group)
        self.b85_block_size.setRange(1, 10000)
        self.b85_block_size.setValue(250)
        self.b85_gap = QtWidgets.QSpinBox(self.b85_group)
        self.b85_gap.setRange(0, 20000)
        self.b85_gap.setValue(500)
        self.b85_overwrite_existing = QtWidgets.QCheckBox('Overwrite existing outputs', self.b85_group)
        self.b85_overwrite_existing.setChecked(False)

        self.b85_steps = QtWidgets.QListWidget(self.b85_group)
        self.b85_steps.setMinimumHeight(210)
        self.b85_steps.setMaximumHeight(280)
        for step in B85_STEP_NAMES:
            item = QtWidgets.QListWidgetItem(step)
            item.setFlags(item.flags() | QtCore.Qt.ItemIsUserCheckable)
            item.setCheckState(QtCore.Qt.Checked)
            self.b85_steps.addItem(item)

        b85_layout.addWidget(self._section_label('Paths'))
        b85_layout.addWidget(self._form_row_widget('Output root', self._path_row_widget(self.b85_output_root, self.b85_pb_output)))
        b85_layout.addWidget(self._form_row_widget('Temp root', self._path_row_widget(self.b85_temp_root, self.b85_pb_temp)))

        b85_layout.addWidget(self._section_label('Dataset'))
        b85_layout.addWidget(self._form_row_widget('Reference channel', self.b85_reference_channel))
        b85_layout.addWidget(self._form_row_widget('Slice range', self._slice_range_widget()))
        b85_layout.addWidget(self._form_row_widget('Output channels', self.b85_output_channels))

        b85_layout.addWidget(self._section_label('Parameters'))
        b85_layout.addWidget(self._numeric_parameters_widget())
        b85_layout.addWidget(self.b85_overwrite_existing)

        b85_layout.addWidget(self._section_label('Steps'))
        b85_layout.addWidget(self.b85_steps)
        b85_layout.addStretch(1)

        self.b85_group.setMinimumWidth(680)
        self.b85_scroll.setWidget(self.b85_group)
        self.verticalLayout.insertWidget(2, self.b85_scroll)
        self.cb_mode.currentIndexChanged.connect(self._mode_changed)
        self.b85_pb_output.clicked.connect(lambda: self._set_b85_path(self.b85_output_root))
        self.b85_pb_temp.clicked.connect(lambda: self._set_b85_path(self.b85_temp_root))
        self._mode_changed()

    def _mode_changed(self):
        is_b85 = self.cb_mode.currentData() == 'b85'
        self.b85_scroll.setVisible(is_b85)

    def _set_running_state(self, running):
        self.pb_start.setEnabled(not running)
        self.pb_stop.setEnabled(running)
        self.pb_settings.setEnabled(not running)
        self.cb_mode.setEnabled(not running)
        self.checkBox.setEnabled(not running)
        self.pb_save.setEnabled((not running) and self.checkBox.isChecked())

    def _set_b85_controls_enabled(self, enabled):
        self.b85_group.setEnabled(enabled)

    def _set_b85_path(self, line_edit):
        d = QtWidgets.QFileDialog()
        d.setFileMode(QtWidgets.QFileDialog.Directory)
        d.setOption(QtWidgets.QFileDialog.ShowDirsOnly, True)
        d.setAcceptMode(QtWidgets.QFileDialog.AcceptOpen)
        path = d.getExistingDirectory(self, 'Select Directory')
        if len(path) == 0:
            return
        line_edit.setText(path)

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
        if self.cb_mode.currentData() == 'b85':
            self.start_b85_reconstruct()
            return
        dst = self.line_edit_save.text()
        if not self.checkBox.isChecked():
            dst = self.pipeline.dataset.path
        self.param['output_path'] = dst
        self.latest_b85_result = None
        self.process_exit_code = None
        self.pipe, pipe = Pipe()
        s = gen_brain_reconstruction_pipeline(self.pipeline.dataset, **self.param)
        p = Process(target=main, args=(s, pipe))
        self.process = p
        p.start()
        self.worker_thread.set_func(self.listen, [p])
        self.worker_thread.start()
        self._set_running_state(True)

    def start_b85_reconstruct(self):
        try:
            config = self._build_b85_config(validate=True)
        except Exception as e:
            self._append_log(str(e))
            self.label_status.setText('Refinement config error')
            return
        self.latest_b85_result = None
        self.process_exit_code = None
        self.pipe, pipe = Pipe()
        p = Process(target=b85_main, args=(config.to_json(), pipe))
        self.process = p
        p.start()
        self.worker_thread.set_func(self.listen, [p])
        self.worker_thread.start()
        self._set_running_state(True)

    def listen(self, p):
        pipe = self.pipe
        while 1:
            try:
                if pipe is not None:
                    if pipe.poll(0.2):
                        self.pipe_message_received.emit(pipe.recv())
                else:
                    p.join(0.2)
            except (EOFError, BrokenPipeError, OSError) as e:
                self.pipe_message_received.emit({
                    'status': 'Pipe closed',
                    'message': 'Process pipe closed: {}'.format(e),
                })
                pipe = None

            p.join(0.01)
            if p.exitcode is not None:
                self.process_exit_code = p.exitcode
                while pipe is not None:
                    try:
                        if not pipe.poll():
                            break
                        self.pipe_message_received.emit(pipe.recv())
                    except (EOFError, BrokenPipeError, OSError):
                        break
                break

    def _handle_pipe_message(self, s):
        if not isinstance(s, dict):
            return
        if 'message' in s:
            self._append_log(s['message'])
        if 'progress' in s:
            try:
                value = int(float(s['progress']) * self.progressBar.maximum())
                value = max(0, min(self.progressBar.maximum(), value))
                self.progressBar.setValue(value)
            except (TypeError, ValueError):
                pass
        if 'status' in s:
            self.label_status.setText(str(s['status']))
        if 'result' in s:
            self.latest_b85_result = s['result']

    def _safe_send_stop(self):
        if self.pipe is None:
            return
        try:
            self.pipe.send({'stop': None})
        except (BrokenPipeError, EOFError, OSError) as e:
            self._append_log('Stop signal failed: {}'.format(e))

    def _close_process_handles(self):
        if self.pipe is not None:
            try:
                self.pipe.close()
            except (BrokenPipeError, EOFError, OSError):
                pass
            self.pipe = None
        if self.process is not None and self.process.exitcode is not None:
            try:
                self.process.close()
            except (AttributeError, ValueError, OSError):
                pass
            self.process = None

    def _apply_b85_result_to_dataset(self):
        if self.latest_b85_result is None or self.pipeline.dataset is None:
            return
        brain_transform = self.latest_b85_result.get('brain_transform')
        if brain_transform is None:
            return

        dataset = self.pipeline.dataset
        dataset.brain_transform = brain_transform
        dataset.reconstruction_info.setdefault('Refinement', {})
        dataset.reconstruction_info['Refinement']['RunSummary'] = self.latest_b85_result
        dataset.misc.setdefault('Reconstruction', {})
        for key, summary_key in [
            ('BrainTransform', 'brain_transform_metadata'),
            ('BrainImage', 'brain_image_metadata'),
        ]:
            metadata_path = self.latest_b85_result.get(summary_key)
            if metadata_path is None:
                continue
            try:
                with open(metadata_path) as fp:
                    dataset.reconstruction_info[key] = json.load(fp)
                try:
                    metadata_ref = os.path.relpath(metadata_path, dataset.path)
                except ValueError:
                    metadata_ref = metadata_path
                dataset.misc['Reconstruction'][key] = metadata_ref
            except Exception as e:
                self._append_log('Load Refinement {} metadata failed: {}'.format(key, e))
        self._append_log('Refinement brain transform: {}'.format(brain_transform))

        try:
            self.pipeline.pages[3]['main_page'].update_dataset()
            self.pipeline.pages[4]['main_page'].update_dataset()
            self.pipeline.toggle_page.emit(4)
        except Exception as e:
            self._append_log('Refresh downstream pages failed: {}'.format(e))

    def stop_reconstruct(self):
        if self.pipe is not None:
            self.label_status.setText('Stopping')
            self._safe_send_stop()

    def reconstruct_finished(self):
        self._set_running_state(False)
        if self.process_exit_code not in (None, 0):
            self.label_status.setText('Failed')
            self._append_log('Reconstruction process exited with code {}'.format(self.process_exit_code))
        self._close_process_handles()
        if self.process_exit_code in (None, 0):
            self._apply_b85_result_to_dataset()

    def settings(self):
        if self.cb_mode.currentData() == 'b85':
            try:
                config = self._build_b85_config(validate=False)
                self._append_log(config.to_json())
            except Exception as e:
                self._append_log(str(e))
            return
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
        self._populate_b85_controls()

    def _populate_b85_controls(self):
        dataset = self.pipeline.dataset
        if dataset is None:
            self._set_b85_controls_enabled(False)
            return
        self._set_b85_controls_enabled(True)
        try:
            config = infer_b85_config(dataset, validate=False)
        except Exception as e:
            self._append_log('Refinement defaults failed: {}'.format(e))
            return

        self.b85_output_root.setText(config.output_root)
        self.b85_temp_root.setText(config.temp_root)
        self.b85_start_slice.setValue(config.start_slice)
        self.b85_end_slice.setValue(config.end_slice_exclusive)
        self.b85_pixel_size.setValue(config.pixel_size)
        self.b85_block_size.setValue(config.block_size)
        self.b85_gap.setValue(config.gap)
        self.b85_overwrite_existing.setChecked(config.overwrite_existing)

        self.b85_reference_channel.clear()
        self.b85_output_channels.clear()
        for channel_id, channel in dataset.channels.items():
            channel_id = str(channel.get('ChannelId', channel_id))
            label = '{} ({}, {})'.format(channel_id, channel.get('ChannelName', ''), channel.get('LaserWavelength', ''))
            self.b85_reference_channel.addItem(label, channel_id)
            item = QtWidgets.QListWidgetItem(label)
            item.setData(QtCore.Qt.UserRole, channel_id)
            item.setFlags(item.flags() | QtCore.Qt.ItemIsUserCheckable)
            checked = channel_id in config.output_channel_ids
            item.setCheckState(QtCore.Qt.Checked if checked else QtCore.Qt.Unchecked)
            self.b85_output_channels.addItem(item)
        ref_index = self.b85_reference_channel.findData(config.reference_channel_id)
        if ref_index >= 0:
            self.b85_reference_channel.setCurrentIndex(ref_index)

    def _selected_b85_steps(self):
        steps = []
        for row in range(self.b85_steps.count()):
            item = self.b85_steps.item(row)
            if item.checkState() == QtCore.Qt.Checked:
                steps.append(item.text())
        return steps

    def _selected_b85_output_channels(self):
        channels = []
        for row in range(self.b85_output_channels.count()):
            item = self.b85_output_channels.item(row)
            if item.checkState() == QtCore.Qt.Checked:
                channels.append(item.data(QtCore.Qt.UserRole))
        return channels

    def _build_b85_config(self, validate):
        return infer_b85_config(
            self.pipeline.dataset,
            output_root=self.b85_output_root.text(),
            temp_root=self.b85_temp_root.text(),
            reference_channel_id=self.b85_reference_channel.currentData(),
            output_channel_ids=self._selected_b85_output_channels(),
            start_slice=self.b85_start_slice.value(),
            end_slice_exclusive=self.b85_end_slice.value(),
            pixel_size=self.b85_pixel_size.value(),
            block_size=self.b85_block_size.value(),
            gap=self.b85_gap.value(),
            overwrite_existing=self.b85_overwrite_existing.isChecked(),
            selected_steps=self._selected_b85_steps(),
            validate=validate,
        )
