# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'settings_dialog_ui.ui'
#
# Created by: PyQt5 UI code generator 5.9.2
#
# WARNING! All changes made in this file will be lost!

from PyQt5 import QtCore, QtGui, QtWidgets

class Ui_Dialog(object):
    def setupUi(self, Dialog):
        Dialog.setObjectName("Dialog")
        Dialog.resize(381, 288)
        self.formLayout = QtWidgets.QFormLayout(Dialog)
        self.formLayout.setObjectName("formLayout")
        self.label = QtWidgets.QLabel(Dialog)
        self.label.setObjectName("label")
        self.formLayout.setWidget(0, QtWidgets.QFormLayout.LabelRole, self.label)
        self.line_edit_voxel_size = QtWidgets.QLineEdit(Dialog)
        self.line_edit_voxel_size.setObjectName("line_edit_voxel_size")
        self.formLayout.setWidget(0, QtWidgets.QFormLayout.FieldRole, self.line_edit_voxel_size)
        self.label_2 = QtWidgets.QLabel(Dialog)
        self.label_2.setObjectName("label_2")
        self.formLayout.setWidget(1, QtWidgets.QFormLayout.LabelRole, self.label_2)
        self.combo_box_method = QtWidgets.QComboBox(Dialog)
        self.combo_box_method.setObjectName("combo_box_method")
        self.formLayout.setWidget(1, QtWidgets.QFormLayout.FieldRole, self.combo_box_method)
        self.label_3 = QtWidgets.QLabel(Dialog)
        self.label_3.setObjectName("label_3")
        self.formLayout.setWidget(2, QtWidgets.QFormLayout.LabelRole, self.label_3)
        self.check_box_thumb = QtWidgets.QCheckBox(Dialog)
        self.check_box_thumb.setText("")
        self.check_box_thumb.setObjectName("check_box_thumb")
        self.formLayout.setWidget(2, QtWidgets.QFormLayout.FieldRole, self.check_box_thumb)
        self.label_4 = QtWidgets.QLabel(Dialog)
        self.label_4.setObjectName("label_4")
        self.formLayout.setWidget(3, QtWidgets.QFormLayout.LabelRole, self.label_4)
        self.check_box_channel = QtWidgets.QCheckBox(Dialog)
        self.check_box_channel.setText("")
        self.check_box_channel.setObjectName("check_box_channel")
        self.formLayout.setWidget(3, QtWidgets.QFormLayout.FieldRole, self.check_box_channel)
        self.label_5 = QtWidgets.QLabel(Dialog)
        self.label_5.setObjectName("label_5")
        self.formLayout.setWidget(4, QtWidgets.QFormLayout.LabelRole, self.label_5)
        self.combo_box_channel_method = QtWidgets.QComboBox(Dialog)
        self.combo_box_channel_method.setObjectName("combo_box_channel_method")
        self.formLayout.setWidget(4, QtWidgets.QFormLayout.FieldRole, self.combo_box_channel_method)
        self.label_6 = QtWidgets.QLabel(Dialog)
        self.label_6.setObjectName("label_6")
        self.formLayout.setWidget(5, QtWidgets.QFormLayout.LabelRole, self.label_6)
        self.combo_box_reference = QtWidgets.QComboBox(Dialog)
        self.combo_box_reference.setObjectName("combo_box_reference")
        self.formLayout.setWidget(5, QtWidgets.QFormLayout.FieldRole, self.combo_box_reference)
        self.pb_import = QtWidgets.QPushButton(Dialog)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.pb_import.sizePolicy().hasHeightForWidth())
        self.pb_import.setSizePolicy(sizePolicy)
        self.pb_import.setObjectName("pb_import")
        self.formLayout.setWidget(6, QtWidgets.QFormLayout.LabelRole, self.pb_import)
        self.pb_export = QtWidgets.QPushButton(Dialog)
        self.pb_export.setObjectName("pb_export")
        self.formLayout.setWidget(6, QtWidgets.QFormLayout.FieldRole, self.pb_export)
        self.buttonBox = QtWidgets.QDialogButtonBox(Dialog)
        self.buttonBox.setOrientation(QtCore.Qt.Horizontal)
        self.buttonBox.setStandardButtons(QtWidgets.QDialogButtonBox.Cancel|QtWidgets.QDialogButtonBox.Ok)
        self.buttonBox.setObjectName("buttonBox")
        self.formLayout.setWidget(7, QtWidgets.QFormLayout.SpanningRole, self.buttonBox)

        self.retranslateUi(Dialog)
        self.buttonBox.accepted.connect(Dialog.accept)
        self.buttonBox.rejected.connect(Dialog.reject)
        QtCore.QMetaObject.connectSlotsByName(Dialog)

    def retranslateUi(self, Dialog):
        _translate = QtCore.QCoreApplication.translate
        Dialog.setWindowTitle(_translate("Dialog", "Settings..."))
        self.label.setText(_translate("Dialog", "Result voxel size / um"))
        self.label_2.setText(_translate("Dialog", "Realign method"))
        self.label_3.setText(_translate("Dialog", "Use thumbnail"))
        self.label_4.setText(_translate("Dialog", "Align different channels"))
        self.label_5.setText(_translate("Dialog", "Channel alignment method"))
        self.label_6.setText(_translate("Dialog", "Reference channel"))
        self.pb_import.setText(_translate("Dialog", "Import settings"))
        self.pb_export.setText(_translate("Dialog", "Export settings"))

