from PyQt5 import QtWidgets, QtGui, QtCore
import os, sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from VISoR_Reconstruction.misc import VERSION, ROOT_DIR

if __name__ == '__main__':
    #os.environ["QT_SCALE_FACTOR"] = '1.5'
    QtWidgets.QApplication.setAttribute(QtCore.Qt.AA_EnableHighDpiScaling)
    app = QtWidgets.QApplication(sys.argv)
    app.setStyle('Fusion')
    ui_font = QtGui.QFont('Microsoft YaHei UI')
    ui_font.setPointSize(10)
    ui_font.setStyleStrategy(QtGui.QFont.PreferAntialias)
    app.setFont(ui_font)
    splash_image = QtGui.QPixmap(os.path.join(os.path.dirname(__file__), 'splash.png'))
    splash = QtWidgets.QSplashScreen(splash_image)
    splash.showMessage('VISoR Reconstruction {}'.format(VERSION), color=QtGui.QColor(255, 255, 255),
                       alignment=QtCore.Qt.AlignBottom)
    splash.show()
    app.processEvents()

    from VISoR_Reconstruction.visor_reconstruction_ui.brain_reconstruction_ui import Ui_MainWindow
    import pathlib
    from VISoR_Reconstruction.visor_reconstruction_ui.pipelines.reconstruction_pipeline import ReconstructionPipeline

    class mainwindow(QtWidgets.QMainWindow, Ui_MainWindow):
        def __init__(self):
            super(mainwindow, self).__init__()
            self.setupUi(self)
            self._setup_main_chrome()
            self.workflowList.currentRowChanged.connect(self.stackedWidget.setCurrentIndex)
            self.stackedWidget.currentChanged.connect(self._set_current_nav_row)
            self.stackedWidget.currentChanged.connect(self._update_page_header)
            #self.page.line_edit_save.textChanged.connect(self.page_2.line_edit_save.setText)
            #self.page_2.line_edit_save.textChanged.connect(self.page.line_edit_save.setText)
            #self.page.line_edit_load.textChanged.connect(self.page_2.line_edit_load.setText)
            #self.page_2.line_edit_load.textChanged.connect(self.page.line_edit_load.setText)
            self.actionUser_Guide.triggered.connect(self.show_user_guide)
            self.setWindowTitle('VISoR Reconstruction {}'.format(VERSION))

            self.pipeline = ReconstructionPipeline(self)
            self.toolBox.removeItem(0)
            for p in self.pipeline.pages:
                self.stackedWidget.addWidget(p['main_page'])
                self.toolBox.addItem(p['toolbox_page'], self._page_icon(p['name']), p['name'])
                self._add_nav_item(p['name'])
            for i in range(len(self.pipeline.pages)):
                self._set_page_enabled(i, self.pipeline.pages[i]['enabled'])
            self.pipeline.toggle_page.connect(lambda x: self._set_page_enabled(x, True))
            self.workflowList.setCurrentRow(0)
            self._update_page_header(0)

        def _setup_main_chrome(self):
            self.setObjectName('mainWindow')
            self.centralwidget.setObjectName('mainCentral')
            self.stackedWidget.setObjectName('mainStack')
            self.toolBox.setObjectName('workflowToolBox')
            self.toolBox.hide()
            self.dockWidget.setObjectName('workflowDock')
            self.dockWidgetContents.setObjectName('workflowDockContents')
            self.statusbar.setObjectName('mainStatusBar')
            self.menubar.setObjectName('mainMenuBar')

            self.resize(1360, 860)
            self.setMinimumSize(1120, 720)
            self.setWindowIcon(QtGui.QIcon(os.path.join(os.path.dirname(__file__), 'splash.png')))
            self.dockWidget.setMinimumWidth(320)
            self.dockWidget.setMaximumWidth(390)
            self.dockWidget.setTitleBarWidget(QtWidgets.QWidget(self.dockWidget))
            self.verticalLayout.setContentsMargins(14, 14, 14, 14)
            self.verticalLayout.setSpacing(14)
            self.verticalLayout_2.setContentsMargins(16, 16, 16, 16)
            self.verticalLayout_2.setSpacing(14)

            self.sidebar_header = QtWidgets.QFrame(self.dockWidgetContents)
            self.sidebar_header.setObjectName('sidebarHeader')
            sidebar_layout = QtWidgets.QVBoxLayout(self.sidebar_header)
            sidebar_layout.setContentsMargins(14, 12, 14, 12)
            sidebar_layout.setSpacing(3)
            self.sidebar_title = QtWidgets.QLabel('VISoR', self.sidebar_header)
            self.sidebar_title.setObjectName('sidebarTitle')
            self.sidebar_subtitle = QtWidgets.QLabel('Reconstruction Workflow', self.sidebar_header)
            self.sidebar_subtitle.setObjectName('sidebarSubtitle')
            sidebar_layout.addWidget(self.sidebar_title)
            sidebar_layout.addWidget(self.sidebar_subtitle)
            self.verticalLayout.insertWidget(0, self.sidebar_header)

            self.workflowList = QtWidgets.QListWidget(self.dockWidgetContents)
            self.workflowList.setObjectName('workflowList')
            self.workflowList.setFrameShape(QtWidgets.QFrame.NoFrame)
            self.workflowList.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
            self.workflowList.setVerticalScrollMode(QtWidgets.QAbstractItemView.ScrollPerPixel)
            self.workflowList.setSelectionMode(QtWidgets.QAbstractItemView.SingleSelection)
            self.workflowList.setWordWrap(True)
            self.workflowList.setSpacing(6)
            self.workflowList.setIconSize(QtCore.QSize(22, 22))
            self.workflowList.setUniformItemSizes(False)
            self.workflowList.setTextElideMode(QtCore.Qt.ElideNone)
            self.verticalLayout.insertWidget(1, self.workflowList, 1)

            self.content_header = QtWidgets.QFrame(self.centralwidget)
            self.content_header.setObjectName('contentHeader')
            header_layout = QtWidgets.QVBoxLayout(self.content_header)
            header_layout.setContentsMargins(16, 12, 16, 12)
            header_layout.setSpacing(4)
            self.page_title = QtWidgets.QLabel('VISoR Reconstruction', self.content_header)
            self.page_title.setObjectName('pageTitle')
            self.page_subtitle = QtWidgets.QLabel('Load data, reconstruct whole brain images, register, and export ROI data.', self.content_header)
            self.page_subtitle.setObjectName('pageSubtitle')
            header_layout.addWidget(self.page_title)
            header_layout.addWidget(self.page_subtitle)
            self.verticalLayout_2.insertWidget(0, self.content_header)

            self.statusbar.showMessage('Ready')
            self._apply_main_style()

        def _add_nav_item(self, name):
            item = QtWidgets.QListWidgetItem(self._page_icon(name), name)
            item.setSizeHint(QtCore.QSize(300, 60))
            item.setToolTip(self._nav_tooltip(name))
            self.workflowList.addItem(item)

        def _set_current_nav_row(self, index):
            if index < 0 or index >= self.workflowList.count():
                return
            if self.workflowList.currentRow() != index:
                self.workflowList.setCurrentRow(index)

        def _set_page_enabled(self, index, enabled):
            self.toolBox.setItemEnabled(index, enabled)
            item = self.workflowList.item(index)
            if item is None:
                return
            flags = item.flags()
            if enabled:
                item.setFlags(flags | QtCore.Qt.ItemIsEnabled | QtCore.Qt.ItemIsSelectable)
                item.setForeground(QtGui.QBrush(QtGui.QColor('#344054')))
            else:
                item.setFlags(flags & ~QtCore.Qt.ItemIsEnabled & ~QtCore.Qt.ItemIsSelectable)
                item.setForeground(QtGui.QBrush(QtGui.QColor('#98a2b3')))

        def _nav_tooltip(self, name):
            tooltip_map = {
                'Data': 'Data: select raw VISoR acquisition data.',
                'Reconstruction': 'Reconstruction: run Standard VISoR or Refinement.',
                'Manual Surface Alignment': 'Manual Surface Alignment: inspect and adjust slice surface alignment.',
                'Brain Registration': 'Brain Registration: register outputs to reference space.',
                'ROI Reconstruction': 'ROI Reconstruction: generate region-focused reconstruction outputs.',
            }
            return tooltip_map.get(name, name)

        def _page_icon(self, name):
            style = self.style()
            icon_map = {
                'Data': QtWidgets.QStyle.SP_DirOpenIcon,
                'Reconstruction': QtWidgets.QStyle.SP_ComputerIcon,
                'Manual Surface Alignment': QtWidgets.QStyle.SP_FileDialogDetailedView,
                'Brain Registration': QtWidgets.QStyle.SP_FileDialogContentsView,
                'ROI Reconstruction': QtWidgets.QStyle.SP_DialogApplyButton,
            }
            return style.standardIcon(icon_map.get(name, QtWidgets.QStyle.SP_FileIcon))

        def _update_page_header(self, index):
            if not hasattr(self, 'pipeline') or index < 0 or index >= len(self.pipeline.pages):
                return
            name = self.pipeline.pages[index]['name']
            subtitle_map = {
                'Data': 'Select raw VISoR acquisition data and prepare the dataset.',
                'Reconstruction': 'Run Standard VISoR reconstruction or the integrated Refinement workflow.',
                'Manual Surface Alignment': 'Inspect and adjust slice surface alignment results.',
                'Brain Registration': 'Register reconstructed brain outputs to reference space.',
                'ROI Reconstruction': 'Generate region-focused reconstruction outputs from the prepared dataset.',
            }
            self.page_title.setText(name)
            self.page_subtitle.setText(subtitle_map.get(name, 'VISoR Reconstruction workflow step.'))
            self.statusbar.showMessage(name)

        def _apply_main_style(self):
            self.setStyleSheet("""
                QMainWindow#mainWindow {
                    background: #eef2f6;
                }
                QWidget {
                    font-family: "Microsoft YaHei UI", "Segoe UI";
                    font-size: 13px;
                }
                QWidget#mainCentral {
                    background: #eef2f6;
                }
                QFrame#contentHeader {
                    background: #ffffff;
                    border: 1px solid #d8e0ea;
                    border-radius: 7px;
                }
                QLabel#pageTitle {
                    color: #1f2937;
                    font-size: 22px;
                    font-weight: 700;
                }
                QLabel#pageSubtitle {
                    color: #667085;
                    font-size: 14px;
                }
                QDockWidget#workflowDock {
                    background: #f8fafc;
                    border-right: 1px solid #d8e0ea;
                    titlebar-close-icon: none;
                    titlebar-normal-icon: none;
                }
                QWidget#workflowDockContents {
                    background: #f8fafc;
                    border-right: 1px solid #d8e0ea;
                }
                QFrame#sidebarHeader {
                    background: #ffffff;
                    border: 1px solid #d8e0ea;
                    border-radius: 7px;
                }
                QLabel#sidebarTitle {
                    color: #1f2937;
                    font-size: 26px;
                    font-weight: 800;
                }
                QLabel#sidebarSubtitle {
                    color: #667085;
                    font-size: 13px;
                    font-weight: 500;
                }
                QToolBox#workflowToolBox {
                    background: transparent;
                    border: none;
                    spacing: 6px;
                }
                QListWidget#workflowList {
                    background: transparent;
                    border: none;
                    outline: 0;
                }
                QListWidget#workflowList::item {
                    background: #ffffff;
                    border: 1px solid #d8e0ea;
                    border-radius: 6px;
                    color: #344054;
                    font-size: 14px;
                    min-height: 42px;
                    padding: 9px 12px;
                    margin: 1px 0;
                    font-weight: 600;
                }
                QListWidget#workflowList::item:selected {
                    background: #eaf3ff;
                    border-color: #84b6f4;
                    color: #1d4ed8;
                }
                QListWidget#workflowList::item:hover {
                    background: #f0f4f8;
                }
                QListWidget#workflowList::item:disabled {
                    background: #edf1f5;
                    border-color: #d8e0ea;
                    color: #98a2b3;
                }
                QStackedWidget#mainStack {
                    background: transparent;
                    border: none;
                }
                QMenuBar#mainMenuBar {
                    background: #ffffff;
                    border-bottom: 1px solid #d8e0ea;
                    color: #344054;
                    padding: 4px 10px;
                }
                QMenuBar#mainMenuBar::item {
                    background: transparent;
                    padding: 7px 12px;
                    border-radius: 5px;
                }
                QMenuBar#mainMenuBar::item:selected {
                    background: #eef4ff;
                    color: #1d4ed8;
                }
                QMenu {
                    background: #ffffff;
                    border: 1px solid #d8e0ea;
                    color: #344054;
                    padding: 4px;
                }
                QMenu::item {
                    padding: 8px 28px 8px 14px;
                    border-radius: 4px;
                }
                QMenu::item:selected {
                    background: #eef4ff;
                    color: #1d4ed8;
                }
                QStatusBar#mainStatusBar {
                    background: #ffffff;
                    border-top: 1px solid #d8e0ea;
                    color: #667085;
                }
                QPushButton {
                    min-height: 34px;
                    padding: 6px 12px;
                }
                QLineEdit, QComboBox, QSpinBox, QDoubleSpinBox {
                    min-height: 32px;
                    padding: 4px 8px;
                }
                QCheckBox {
                    spacing: 8px;
                }
                QScrollBar:vertical {
                    background: transparent;
                    width: 10px;
                    margin: 0;
                }
                QScrollBar::handle:vertical {
                    background: #c4ced8;
                    border-radius: 5px;
                    min-height: 24px;
                }
                QScrollBar::handle:vertical:hover {
                    background: #98a2b3;
                }
                QScrollBar::add-line:vertical,
                QScrollBar::sub-line:vertical {
                    height: 0;
                }
            """)

        def show_user_guide(self):
            try:
                s = QtGui.QDesktopServices()
                #print(os.path.join(ROOT_DIR, 'doc', 'user_guide.pdf'))
                c = s.openUrl(QtCore.QUrl().fromLocalFile(os.path.join(ROOT_DIR, 'doc', 'user_guide.pdf')))
                #print(c)
            except:
                print('Failed to open user guide.')


    window = mainwindow()
    window.show()
    splash.finish(window)
    smoke_exit_ms = os.environ.get('VISOR_UI_SMOKE_EXIT_MS')
    if smoke_exit_ms:
        QtCore.QTimer.singleShot(int(smoke_exit_ms), app.quit)
    sys.exit(app.exec_())
