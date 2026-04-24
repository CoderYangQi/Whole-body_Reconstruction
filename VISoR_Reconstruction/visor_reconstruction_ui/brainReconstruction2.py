from PyQt5 import QtWidgets, QtGui, QtCore
import os, sys
from VISoR_Reconstruction.misc import VERSION, ROOT_DIR

if __name__ == '__main__':
    #os.environ["QT_SCALE_FACTOR"] = '1.5'
    app = QtWidgets.QApplication(sys.argv)
    app.setAttribute(QtCore.Qt.AA_EnableHighDpiScaling)
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
            self.toolBox.currentChanged.connect(self.stackedWidget.setCurrentIndex)
            self.stackedWidget.currentChanged.connect(self.toolBox.setCurrentIndex)
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
                self.toolBox.addItem(p['toolbox_page'],  p['name'])
            for i in range(len(self.pipeline.pages)):
                self.toolBox.setItemEnabled(i, self.pipeline.pages[i]['enabled'])
            self.pipeline.toggle_page.connect(lambda x: self.toolBox.setItemEnabled(x, True))

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
    sys.exit(app.exec_())