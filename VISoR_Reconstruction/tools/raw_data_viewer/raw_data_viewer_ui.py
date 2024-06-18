# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'raw_data_viewer_ui.ui'
#
# Created by: PyQt5 UI code generator 5.6
#
# WARNING! All changes made in this file will be lost!

from PyQt5 import QtCore, QtGui, QtWidgets

class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(1074, 600)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.horizontalLayout = QtWidgets.QHBoxLayout(self.centralwidget)
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.widget1 = AnnotationWidget(self.centralwidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.widget1.sizePolicy().hasHeightForWidth())
        self.widget1.setSizePolicy(sizePolicy)
        self.widget1.setObjectName("widget1")
        self.horizontalLayout.addWidget(self.widget1)
        self.widget2 = AnnotationWidget(self.centralwidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.widget2.sizePolicy().hasHeightForWidth())
        self.widget2.setSizePolicy(sizePolicy)
        self.widget2.setObjectName("widget2")
        self.horizontalLayout.addWidget(self.widget2)
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 1074, 31))
        self.menubar.setObjectName("menubar")
        self.menuFile = QtWidgets.QMenu(self.menubar)
        self.menuFile.setObjectName("menuFile")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)
        self.recorder = QtWidgets.QDockWidget(MainWindow)
        self.recorder.setFeatures(QtWidgets.QDockWidget.DockWidgetFloatable|QtWidgets.QDockWidget.DockWidgetMovable)
        self.recorder.setObjectName("recorder")
        self.dockWidgetContents_4 = QtWidgets.QWidget()
        self.dockWidgetContents_4.setObjectName("dockWidgetContents_4")
        self.verticalLayout = QtWidgets.QVBoxLayout(self.dockWidgetContents_4)
        self.verticalLayout.setContentsMargins(0, 0, 0, 0)
        self.verticalLayout.setObjectName("verticalLayout")
        self.textBrowser = QtWidgets.QTextBrowser(self.dockWidgetContents_4)
        self.textBrowser.setUndoRedoEnabled(True)
        self.textBrowser.setReadOnly(False)
        self.textBrowser.setAcceptRichText(False)
        self.textBrowser.setObjectName("textBrowser")
        self.verticalLayout.addWidget(self.textBrowser)
        self.pb_record = QtWidgets.QPushButton(self.dockWidgetContents_4)
        self.pb_record.setObjectName("pb_record")
        self.verticalLayout.addWidget(self.pb_record)
        self.recorder.setWidget(self.dockWidgetContents_4)
        MainWindow.addDockWidget(QtCore.Qt.DockWidgetArea(1), self.recorder)
        self.navigator = QtWidgets.QDockWidget(MainWindow)
        self.navigator.setFeatures(QtWidgets.QDockWidget.DockWidgetFloatable|QtWidgets.QDockWidget.DockWidgetMovable)
        self.navigator.setObjectName("navigator")
        self.dockWidgetContents = QtWidgets.QWidget()
        self.dockWidgetContents.setObjectName("dockWidgetContents")
        self.verticalLayout_2 = QtWidgets.QVBoxLayout(self.dockWidgetContents)
        self.verticalLayout_2.setContentsMargins(0, 0, 0, 0)
        self.verticalLayout_2.setObjectName("verticalLayout_2")
        self.listView = QtWidgets.QListView(self.dockWidgetContents)
        self.listView.setObjectName("listView")
        self.verticalLayout_2.addWidget(self.listView)
        self.horizontalLayout_2 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_2.setObjectName("horizontalLayout_2")
        self.pb_navi_prev = QtWidgets.QPushButton(self.dockWidgetContents)
        self.pb_navi_prev.setObjectName("pb_navi_prev")
        self.horizontalLayout_2.addWidget(self.pb_navi_prev)
        self.pb_navi_next = QtWidgets.QPushButton(self.dockWidgetContents)
        self.pb_navi_next.setObjectName("pb_navi_next")
        self.horizontalLayout_2.addWidget(self.pb_navi_next)
        self.verticalLayout_2.addLayout(self.horizontalLayout_2)
        self.navigator.setWidget(self.dockWidgetContents)
        MainWindow.addDockWidget(QtCore.Qt.DockWidgetArea(1), self.navigator)
        self.actionOpen = QtWidgets.QAction(MainWindow)
        self.actionOpen.setObjectName("actionOpen")
        self.actionSave_records = QtWidgets.QAction(MainWindow)
        self.actionSave_records.setObjectName("actionSave_records")
        self.actionLoad_points = QtWidgets.QAction(MainWindow)
        self.actionLoad_points.setObjectName("actionLoad_points")
        self.menuFile.addAction(self.actionOpen)
        self.menuFile.addAction(self.actionSave_records)
        self.menuFile.addAction(self.actionLoad_points)
        self.menubar.addAction(self.menuFile.menuAction())

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "Slice Viewer"))
        self.menuFile.setTitle(_translate("MainWindow", "File"))
        self.recorder.setWindowTitle(_translate("MainWindow", "Record"))
        self.pb_record.setText(_translate("MainWindow", "record position"))
        self.navigator.setWindowTitle(_translate("MainWindow", "Navigate"))
        self.pb_navi_prev.setText(_translate("MainWindow", "prev"))
        self.pb_navi_next.setText(_translate("MainWindow", "next"))
        self.actionOpen.setText(_translate("MainWindow", "open"))
        self.actionSave_records.setText(_translate("MainWindow", "save records"))
        self.actionLoad_points.setText(_translate("MainWindow", "load points"))

from VISoR_Brain.tools.common.annotation_widget.annotation_widget import AnnotationWidget
