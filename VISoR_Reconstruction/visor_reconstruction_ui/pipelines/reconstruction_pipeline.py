from VISoR_Reconstruction.visor_reconstruction_ui.pages.select_raw_data.select_raw_data import SelectRawDataPage
from VISoR_Reconstruction.visor_reconstruction_ui.pages.surface_registration.surface_registration import SurfaceRegistrationPage
from VISoR_Reconstruction.visor_reconstruction_ui.pages.whole_brain_reconstruct.whole_brain_reconstruct import WholeBrainReconstructPage
from VISoR_Reconstruction.visor_reconstruction_ui.pages.brain_registration.brain_registration import BrainRegistrationPage
from VISoR_Reconstruction.visor_reconstruction_ui.pages.roi_reconstruction.roi_reconstruction import ROIReconsrtuctionPage
from VISoR_Brain.format.visor_data import VISoRData
from PyQt5 import QtWidgets, QtCore


class ReconstructionPipeline(QtCore.QObject):

    toggle_page = QtCore.pyqtSignal(int)

    def __init__(self, parent=None):
        super(ReconstructionPipeline, self).__init__(parent)
        self.dataset = None

        self.pages = [{'name': 'Data',
                       'main_page': SelectRawDataPage(pipeline=self),
                       'toolbox_page': QtWidgets.QWidget(),
                       'enabled': True},
                      {'name': 'Reconstruction',
                       'main_page': WholeBrainReconstructPage(pipeline=self),
                       'toolbox_page': QtWidgets.QWidget(),
                       'enabled': False},
                      {'name': 'Manual Surface Alignment',
                       'main_page': SurfaceRegistrationPage(),
                       'toolbox_page': QtWidgets.QWidget(),
                       'enabled': True},
                      {'name': 'Brain Registration',
                       'main_page': BrainRegistrationPage(pipeline=self),
                       'toolbox_page': QtWidgets.QWidget(),
                       'enabled': True},
                      {'name': 'ROI Reconstruction',
                       'main_page': ROIReconsrtuctionPage(pipeline=self),
                       'toolbox_page': QtWidgets.QWidget(),
                       'enabled': False}
                      ]

    def set_dataset(self, dataset: VISoRData):
        self.dataset = dataset
        self.pages[1]['main_page'].update_dataset()
        self.pages[3]['main_page'].update_dataset()
        self.toggle_page.emit(1)
        if dataset.brain_transform is not None:
            self.pages[4]['main_page'].update_dataset()
            self.toggle_page.emit(4)
