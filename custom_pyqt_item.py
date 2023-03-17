import pyqtgraph as pg
from pyqtgraph import QtCore
from pyqtgraph import GraphicsObject
import PyQt5.QtGui as QtGui


class ColoredBarGraphItem(GraphicsObject):
    def __init__(self, height, x, width):
        super().__init__()
        self.height = height
        self.x = x
        self.width = width

    def setOpts(self, height):
        self.height = height
        self.prepareGeometryChange()

    def boundingRect(self):
        return pg.QtCore.QRectF(self.x[0] - self.width / 2, 0, len(self.x) * self.width, 1)

    def paint(self, p, *args):
        p.setRenderHint(p.Antialiasing, False)
        p.setRenderHint(p.TextAntialiasing, True)
        no_pen = QtGui.QPen(QtCore.Qt.NoPen)

        for i in range(len(self.x)):
            height = self.height[i]
            green_height = min(height, 0.5)
            yellow_height = min(max(0, height - green_height), 0.3)
            red_height = max(0, height - green_height - yellow_height)

            p.setBrush(pg.mkBrush((0, 255, 0, 255)))  # Green
            p.setPen(no_pen)
            p.drawRect(QtCore.QRectF(self.x[i] - self.width / 2, 0, self.width, green_height))

            p.setBrush(pg.mkBrush((255, 255, 0, 255)))  # Yellow
            p.setPen(no_pen)
            p.drawRect(QtCore.QRectF(self.x[i] - self.width / 2, green_height, self.width, yellow_height))

            p.setBrush(pg.mkBrush((255, 0, 0, 255)))  # Red
            p.setPen(no_pen)
            p.drawRect(QtCore.QRectF(self.x[i] - self.width / 2, green_height + yellow_height, self.width, red_height))
