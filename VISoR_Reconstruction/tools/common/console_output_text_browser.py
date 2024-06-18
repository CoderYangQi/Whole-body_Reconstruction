from PyQt5 import QtWidgets, QtGui


class ConsoleOutputTextBrowser(QtWidgets.QTextBrowser):

    def append(self, text: str):
        pos = self.textCursor().position()
        self.moveCursor(QtGui.QTextCursor.End)
        self.insertPlainText(text)
        self.textCursor().setPosition(pos)