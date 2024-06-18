# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'whole_brain_reconstruct_ui.ui'
#
# Created by: PyQt5 UI code generator 5.9.2
#
# WARNING! All changes made in this file will be lost!

from PyQt5 import QtCore, QtGui, QtWidgets

class Ui_Form(object):
    def setupUi(self, Form):
        Form.setObjectName("Form")
        Form.resize(901, 735)
        self.verticalLayout = QtWidgets.QVBoxLayout(Form)
        self.verticalLayout.setObjectName("verticalLayout")
        self.gridLayout = QtWidgets.QGridLayout()
        self.gridLayout.setObjectName("gridLayout")
        self.pb_save = QtWidgets.QPushButton(Form)
        self.pb_save.setObjectName("pb_save")
        self.gridLayout.addWidget(self.pb_save, 0, 2, 1, 1)
        self.line_edit_save = QtWidgets.QLineEdit(Form)
        self.line_edit_save.setObjectName("line_edit_save")
        self.gridLayout.addWidget(self.line_edit_save, 0, 1, 1, 1)
        self.checkBox = QtWidgets.QCheckBox(Form)
        self.checkBox.setObjectName("checkBox")
        self.gridLayout.addWidget(self.checkBox, 0, 0, 1, 1)
        self.verticalLayout.addLayout(self.gridLayout)
        self.horizontalLayout = QtWidgets.QHBoxLayout()
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.pb_start = QtWidgets.QPushButton(Form)
        self.pb_start.setObjectName("pb_start")
        self.horizontalLayout.addWidget(self.pb_start)
        self.pb_stop = QtWidgets.QPushButton(Form)
        self.pb_stop.setObjectName("pb_stop")
        self.horizontalLayout.addWidget(self.pb_stop)
        self.label_status = QtWidgets.QLabel(Form)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.label_status.sizePolicy().hasHeightForWidth())
        self.label_status.setSizePolicy(sizePolicy)
        self.label_status.setText("")
        self.label_status.setObjectName("label_status")
        self.horizontalLayout.addWidget(self.label_status)
        self.progressBar = QtWidgets.QProgressBar(Form)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.progressBar.sizePolicy().hasHeightForWidth())
        self.progressBar.setSizePolicy(sizePolicy)
        self.progressBar.setMaximum(1000)
        self.progressBar.setProperty("value", 0)
        self.progressBar.setObjectName("progressBar")
        self.horizontalLayout.addWidget(self.progressBar)
        self.pb_settings = QtWidgets.QPushButton(Form)
        self.pb_settings.setObjectName("pb_settings")
        self.horizontalLayout.addWidget(self.pb_settings)
        self.verticalLayout.addLayout(self.horizontalLayout)
        self.textBrowser = ConsoleOutputTextBrowser(Form)
        self.textBrowser.setObjectName("textBrowser")
        self.verticalLayout.addWidget(self.textBrowser)

        self.retranslateUi(Form)
        QtCore.QMetaObject.connectSlotsByName(Form)

    def retranslateUi(self, Form):
        _translate = QtCore.QCoreApplication.translate
        Form.setWindowTitle(_translate("Form", "Form"))
        self.pb_save.setText(_translate("Form", "Browse..."))
        self.checkBox.setText(_translate("Form", "Save to different directory"))
        self.pb_start.setText(_translate("Form", "Start"))
        self.pb_stop.setText(_translate("Form", "Stop"))
        self.pb_settings.setText(_translate("Form", "Settings"))

from VISoR_Reconstruction.tools.common.console_output_text_browser import ConsoleOutputTextBrowser
