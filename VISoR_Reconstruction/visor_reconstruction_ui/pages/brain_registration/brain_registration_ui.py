# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'brain_registration_ui.ui'
#
# Created by: PyQt5 UI code generator 5.9.2
#
# WARNING! All changes made in this file will be lost!

from PyQt5 import QtCore, QtGui, QtWidgets

class Ui_Form(object):
    def setupUi(self, Form):
        Form.setObjectName("Form")
        Form.resize(1087, 864)
        self.verticalLayout_2 = QtWidgets.QVBoxLayout(Form)
        self.verticalLayout_2.setObjectName("verticalLayout_2")
        self.verticalLayout = QtWidgets.QVBoxLayout()
        self.verticalLayout.setObjectName("verticalLayout")
        self.horizontalLayout_2 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_2.setObjectName("horizontalLayout_2")
        self.checkBox = QtWidgets.QCheckBox(Form)
        self.checkBox.setObjectName("checkBox")
        self.horizontalLayout_2.addWidget(self.checkBox)
        self.lineEdit = QtWidgets.QLineEdit(Form)
        self.lineEdit.setObjectName("lineEdit")
        self.horizontalLayout_2.addWidget(self.lineEdit)
        self.pb_browse = QtWidgets.QPushButton(Form)
        self.pb_browse.setObjectName("pb_browse")
        self.horizontalLayout_2.addWidget(self.pb_browse)
        self.verticalLayout.addLayout(self.horizontalLayout_2)
        self.horizontalLayout = QtWidgets.QHBoxLayout()
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.label = QtWidgets.QLabel(Form)
        self.label.setObjectName("label")
        self.horizontalLayout.addWidget(self.label)
        self.cb_template = QtWidgets.QComboBox(Form)
        self.cb_template.setObjectName("cb_template")
        self.horizontalLayout.addWidget(self.cb_template)
        self.label_2 = QtWidgets.QLabel(Form)
        self.label_2.setObjectName("label_2")
        self.horizontalLayout.addWidget(self.label_2)
        self.cb_channel = QtWidgets.QComboBox(Form)
        self.cb_channel.setObjectName("cb_channel")
        self.horizontalLayout.addWidget(self.cb_channel)
        self.pb_start = QtWidgets.QPushButton(Form)
        self.pb_start.setObjectName("pb_start")
        self.horizontalLayout.addWidget(self.pb_start)
        spacerItem = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout.addItem(spacerItem)
        self.label_status = QtWidgets.QLabel(Form)
        self.label_status.setObjectName("label_status")
        self.horizontalLayout.addWidget(self.label_status)
        self.pb_settings = QtWidgets.QPushButton(Form)
        self.pb_settings.setObjectName("pb_settings")
        self.horizontalLayout.addWidget(self.pb_settings)
        self.verticalLayout.addLayout(self.horizontalLayout)
        self.verticalLayout_2.addLayout(self.verticalLayout)
        self.widget = QtWidgets.QWidget(Form)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.widget.sizePolicy().hasHeightForWidth())
        self.widget.setSizePolicy(sizePolicy)
        self.widget.setObjectName("widget")
        self.verticalLayout_2.addWidget(self.widget)

        self.retranslateUi(Form)
        QtCore.QMetaObject.connectSlotsByName(Form)

    def retranslateUi(self, Form):
        _translate = QtCore.QCoreApplication.translate
        Form.setWindowTitle(_translate("Form", "Form"))
        self.checkBox.setText(_translate("Form", "Use other dataset"))
        self.pb_browse.setText(_translate("Form", "Browse..."))
        self.label.setText(_translate("Form", "Template"))
        self.label_2.setText(_translate("Form", "Channel"))
        self.pb_start.setText(_translate("Form", "Start Registration"))
        self.label_status.setText(_translate("Form", "Ready"))
        self.pb_settings.setText(_translate("Form", "Settings"))

