import sys
import numpy as np
from PyQt5.QtWidgets import QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QLabel, QStatusBar
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Interactive Endoscopy/IVUS UI")
        self.setStyleSheet("background-color: #222; color: #eee;")
        self.resize(1200, 600)

        # Central widget and layout
        central = QWidget()
        layout = QHBoxLayout()
        central.setLayout(layout)
        self.setCentralWidget(central)

        # Main live feed (large)
        self.main_canvas = FigureCanvas(Figure(figsize=(5, 5)))
        self.main_ax = self.main_canvas.figure.subplots()
        self.main_ax.set_title("Live Feed", color='w')
        self.main_ax.axis('off')
        layout.addWidget(self.main_canvas, stretch=2)

        # Side panel (vertical)
        side_panel = QVBoxLayout()
        layout.addLayout(side_panel, stretch=1)

        # Coronal view
        self.coronal_canvas = FigureCanvas(Figure(figsize=(3, 3)))
        self.coronal_ax = self.coronal_canvas.figure.subplots()
        self.coronal_ax.set_title("Coronal View", color='w')
        self.coronal_ax.axis('off')
        side_panel.addWidget(self.coronal_canvas)

        # Transverse view
        self.trans_canvas = FigureCanvas(Figure(figsize=(3, 3)))
        self.trans_ax = self.trans_canvas.figure.subplots()
        self.trans_ax.set_title("Transverse View", color='w')
        self.trans_ax.axis('off')
        side_panel.addWidget(self.trans_canvas)

        # Status bar
        self.status = QStatusBar()
        self.setStatusBar(self.status)
        self.status.showMessage("Patient: John Doe | Procedure: Demo | System Ready")

        # Demo: draw placeholder images and marker
        self.draw_demo()

    def draw_demo(self):
        # Main view: pinkish tissue
        img = np.random.normal(0.5, 0.1, (256, 256)) + 0.3 * np.exp(-((np.indices((256,256))[0]-128)**2 + (np.indices((256,256))[1]-128)**2)/8000)
        self.main_ax.imshow(img, cmap='pink')
        self.main_ax.annotate("Verumontanum", xy=(128, 220), xytext=(128, 250), color='yellow',
                              ha='center', arrowprops=dict(facecolor='yellow', shrink=0.05))
        self.main_canvas.draw()

        # Coronal and transverse: blue cross-sections with yellow marker
        for ax in [self.coronal_ax, self.trans_ax]:
            ax.clear()
            circ = np.linspace(0, 2*np.pi, 100)
            ax.fill(64+40*np.cos(circ), 64+40*np.sin(circ), color='#00aaff', alpha=0.5)
            ax.plot(64, 90, 'yo', markersize=14)  # marker
            ax.set_xlim(0, 128)
            ax.set_ylim(0, 128)
            ax.axis('off')
        self.coronal_canvas.draw()
        self.trans_canvas.draw()

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())