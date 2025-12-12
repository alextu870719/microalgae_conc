import sys
import os
import cv2
import numpy as np
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
                             QPushButton, QLabel, QFileDialog, QListWidget, QSplitter, 
                             QProgressBar, QMessageBox, QGraphicsView, QGraphicsScene, 
                             QGraphicsPixmapItem, QGraphicsPolygonItem)
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QPointF
from PyQt6.QtGui import QPixmap, QImage, QPolygonF, QPen, QColor, QBrush, QAction

from inference import MicroalgaeDetector

class InferenceWorker(QThread):
    finished = pyqtSignal(object, int, str) # result_image, count, error_message
    progress = pyqtSignal(str)

    def __init__(self, model_path, image_path, roi_points):
        super().__init__()
        self.model_path = model_path
        self.image_path = image_path
        self.roi_points = roi_points

    def run(self):
        try:
            self.progress.emit("Loading model...")
            detector = MicroalgaeDetector(self.model_path)
            
            self.progress.emit("Running inference...")
            detections, original_img = detector.predict(self.image_path, self.roi_points)
            
            self.progress.emit("Drawing results...")
            result_img = detector.draw_results(original_img, detections)
            
            # Convert BGR to RGB for Qt
            result_img = cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB)
            
            self.finished.emit(result_img, len(detections), "")
        except Exception as e:
            self.finished.emit(None, 0, str(e))

class ImageScene(QGraphicsScene):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.image_item = None
        self.polygon_item = None
        self.points = []
        self.drawing = False
        self.temp_line = None

    def set_image(self, cv_img):
        self.clear()
        self.points = []
        self.drawing = False
        
        # Convert cv image to QPixmap
        if len(cv_img.shape) == 3:
            h, w, ch = cv_img.shape
            bytes_per_line = ch * w
            q_img = QImage(cv_img.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
        else:
            h, w = cv_img.shape
            q_img = QImage(cv_img.data, w, h, w, QImage.Format.Format_Grayscale8)
            
        pixmap = QPixmap.fromImage(q_img)
        self.image_item = QGraphicsPixmapItem(pixmap)
        self.addItem(self.image_item)
        self.setSceneRect(0, 0, w, h)

    def start_drawing(self):
        self.drawing = True
        self.points = []
        if self.polygon_item:
            self.removeItem(self.polygon_item)
            self.polygon_item = None

    def mousePressEvent(self, event):
        if self.drawing and self.image_item:
            pos = event.scenePos()
            # Check if click is within image bounds
            if 0 <= pos.x() <= self.width() and 0 <= pos.y() <= self.height():
                self.points.append(pos)
                self.update_polygon()
        super().mousePressEvent(event)

    def mouseDoubleClickEvent(self, event):
        if self.drawing and len(self.points) > 2:
            self.drawing = False
            # Close the polygon
            self.update_polygon(closed=True)
        super().mouseDoubleClickEvent(event)

    def update_polygon(self, closed=False):
        if not self.points:
            return
            
        poly = QPolygonF(self.points)
        if closed:
            poly.append(self.points[0])
            
        if self.polygon_item:
            self.removeItem(self.polygon_item)
            
        self.polygon_item = QGraphicsPolygonItem(poly)
        pen = QPen(QColor(255, 0, 0), 3)
        brush = QBrush(QColor(255, 0, 0, 50))
        self.polygon_item.setPen(pen)
        self.polygon_item.setBrush(brush)
        self.addItem(self.polygon_item)

    def get_roi_points(self):
        if not self.points or len(self.points) < 3:
            return None
        return [(p.x(), p.y()) for p in self.points]

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Microalgae Counter")
        self.resize(1200, 800)
        
        self.current_image_path = None
        
        # Determine the path to the model file
        if getattr(sys, 'frozen', False):
            # If running as compiled exe
            application_path = os.path.dirname(sys.executable)
        else:
            # If running as script
            application_path = os.path.dirname(os.path.abspath(__file__))
            
        self.model_path = os.path.join(application_path, "best.pt")
        
        self.init_ui()
        
    def init_ui(self):
        # Main Layout
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        layout = QHBoxLayout(main_widget)
        
        # Left Panel: File List
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        
        self.btn_load_folder = QPushButton("Load Folder")
        self.btn_load_folder.clicked.connect(self.load_folder)
        left_layout.addWidget(self.btn_load_folder)
        
        self.file_list = QListWidget()
        self.file_list.itemClicked.connect(self.load_image)
        left_layout.addWidget(self.file_list)
        
        # Center Panel: Image View
        self.scene = ImageScene()
        self.view = QGraphicsView(self.scene)
        self.view.setRenderHint(Qt.RenderHintType.Antialiasing)
        self.view.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)
        
        # Right Panel: Controls
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        
        self.btn_draw_roi = QPushButton("Draw ROI (Polygon)")
        self.btn_draw_roi.setCheckable(True)
        self.btn_draw_roi.clicked.connect(self.toggle_drawing)
        right_layout.addWidget(self.btn_draw_roi)
        
        self.btn_reset_roi = QPushButton("Reset ROI")
        self.btn_reset_roi.clicked.connect(self.reset_roi)
        right_layout.addWidget(self.btn_reset_roi)
        
        right_layout.addStretch()
        
        self.btn_load_model = QPushButton("Load Model (.pt)")
        self.btn_load_model.clicked.connect(self.load_model_file)
        right_layout.addWidget(self.btn_load_model)
        
        self.lbl_model = QLabel("Model: best.pt")
        right_layout.addWidget(self.lbl_model)
        
        self.btn_run = QPushButton("Run Counting")
        self.btn_run.setStyleSheet("background-color: #4CAF50; color: white; font-weight: bold; padding: 10px;")
        self.btn_run.clicked.connect(self.run_inference)
        right_layout.addWidget(self.btn_run)
        
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        right_layout.addWidget(self.progress_bar)
        
        self.lbl_result = QLabel("Count: -")
        self.lbl_result.setStyleSheet("font-size: 24px; font-weight: bold;")
        right_layout.addWidget(self.lbl_result)
        
        right_layout.addStretch()

        # Splitter
        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.addWidget(left_panel)
        splitter.addWidget(self.view)
        splitter.addWidget(right_panel)
        splitter.setStretchFactor(1, 4) # Image view gets most space
        
        layout.addWidget(splitter)

    def load_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "Select Folder")
        if folder:
            self.file_list.clear()
            extensions = ['.jpg', '.jpeg', '.png', '.bmp']
            for f in os.listdir(folder):
                if any(f.lower().endswith(ext) for ext in extensions):
                    self.file_list.addItem(os.path.join(folder, f))

    def load_image(self, item):
        path = item.text()
        self.current_image_path = path
        
        # Read with opencv to ensure we have data for scene
        img = cv2.imread(path)
        if img is not None:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            self.scene.set_image(img)
            self.view.fitInView(self.scene.sceneRect(), Qt.AspectRatioMode.KeepAspectRatio)
            self.lbl_result.setText("Count: -")
            self.btn_draw_roi.setChecked(False)

    def toggle_drawing(self):
        if self.btn_draw_roi.isChecked():
            self.view.setDragMode(QGraphicsView.DragMode.NoDrag)
            self.scene.start_drawing()
            self.btn_draw_roi.setText("Finish Drawing (Double Click)")
        else:
            self.view.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)
            self.btn_draw_roi.setText("Draw ROI (Polygon)")

    def reset_roi(self):
        self.scene.points = []
        if self.scene.polygon_item:
            self.scene.removeItem(self.scene.polygon_item)
            self.scene.polygon_item = None
        self.btn_draw_roi.setChecked(False)
        self.toggle_drawing()

    def load_model_file(self):
        path, _ = QFileDialog.getOpenFileName(self, "Select Model", "", "Model Files (*.pt)")
        if path:
            self.model_path = path
            self.lbl_model.setText(f"Model: {os.path.basename(path)}")

    def run_inference(self):
        if not self.current_image_path:
            QMessageBox.warning(self, "Warning", "Please select an image first.")
            return
            
        if not os.path.exists(self.model_path):
            QMessageBox.warning(self, "Warning", f"Model file not found: {self.model_path}")
            return

        roi = self.scene.get_roi_points()
        
        self.worker = InferenceWorker(self.model_path, self.current_image_path, roi)
        self.worker.progress.connect(lambda s: self.progress_bar.setFormat(s))
        self.worker.finished.connect(self.on_inference_finished)
        
        self.progress_bar.setVisible(True)
        self.progress_bar.setRange(0, 0) # Indeterminate
        self.btn_run.setEnabled(False)
        self.worker.start()

    def on_inference_finished(self, result_img, count, error):
        self.progress_bar.setVisible(False)
        self.btn_run.setEnabled(True)
        
        if error:
            QMessageBox.critical(self, "Error", error)
            return
            
        self.lbl_result.setText(f"Count: {count}")
        self.scene.set_image(result_img)
        
        # Re-draw ROI if it existed, just for visual confirmation
        # (Optional, might obscure results)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())
