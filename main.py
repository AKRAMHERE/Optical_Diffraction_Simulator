"""
main.py — Application Entry Point
===================================
Run this file to start the Optical Diffraction Simulator.

Usage:
    python main.py

Dependencies:
    pip install numpy matplotlib PyQt5
"""

import sys
import os

# Make sure our modules are importable regardless of cwd
sys.path.insert(0, os.path.dirname(__file__))

from PyQt5.QtWidgets import QApplication
from PyQt5.QtGui import QIcon
from gui import MainWindow


def main():
    app = QApplication(sys.argv)
    app.setApplicationName("Optical Diffraction Simulator")

    # Use a clean fusion style as a base, then our dark theme overrides
    app.setStyle("Fusion")

    window = MainWindow()
    window.show()

    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
