import sys
import os
import cv2
import numpy as np
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
                             QPushButton, QLabel, QFileDialog, QListWidget, QSplitter, 
                             QProgressBar, QMessageBox, QGraphicsView, QGraphicsScene, 
                             QGraphicsPixmapItem, QGraphicsPolygonItem, QDialog, QFormLayout, 
                             QSpinBox, QDoubleSpinBox, QCheckBox)
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QPointF
from PyQt6.QtGui import QPixmap, QImage, QPolygonF, QPen, QColor, QBrush, QAction, QPainter

from inference import MicroalgaeDetector

class InferenceWorker(QThread):
    finished = pyqtSignal(object, object, object, int, str) # result_image, original_img, detections, count, error_message
    progress = pyqtSignal(str)

    def __init__(self, model_path, image_path, roi_points, conf_thres):
        super().__init__()
        self.model_path = model_path
        self.image_path = image_path
        self.roi_points = roi_points
        self.conf_thres = conf_thres

    def run(self):
        try:
            self.progress.emit("Loading model...")
            detector = MicroalgaeDetector(self.model_path)
            
            self.progress.emit("Running inference...")
            detections, original_img = detector.predict(self.image_path, self.roi_points, self.conf_thres)
            
            self.progress.emit("Drawing results...")
            result_img = detector.draw_results(original_img, detections)
            
            # Convert BGR to RGB for Qt
            result_img = cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB)
            original_img_rgb = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)
            
            self.finished.emit(result_img, original_img_rgb, detections, len(detections), "")
        except Exception as e:
            self.finished.emit(None, None, None, 0, str(e))

class ImageScene(QGraphicsScene):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.image_item = None
        self.polygon_item = None
        self.points = []
        self.drawing = False
        self.temp_line = None

    def set_image(self, cv_img):
        # Save current polygon points before clearing
        current_points = self.points
        
        self.clear()
        self.polygon_item = None 
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
        
        # Restore polygon if it exists
        if current_points:
            self.points = current_points
            self.update_polygon(closed=True)

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

class HemocytometerDialog(QDialog):
    def __init__(self, counts, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Hemocytometer Calculator")
        self.resize(400, 300)
        
        layout = QFormLayout(self)
        
        # Handle single count or list of counts
        if isinstance(counts, list):
            self.counts = counts
            avg_count = sum(counts) / len(counts) if counts else 0
            count_text = f"{avg_count:.1f} (Avg of {len(counts)} images)"
            self.base_count = avg_count
        else:
            self.counts = [counts]
            self.base_count = counts
            count_text = str(counts)
            
        self.lbl_count_display = QLabel(count_text)
        layout.addRow("Cell Count:", self.lbl_count_display)
        
        self.input_squares = QDoubleSpinBox()
        self.input_squares.setRange(0.0001, 100)
        self.input_squares.setDecimals(4)
        self.input_squares.setValue(1.0)
        self.input_squares.setToolTip("Number of 1mm x 1mm squares counted")
        layout.addRow("Squares (1mm²):", self.input_squares)
        
        self.input_dilution = QDoubleSpinBox()
        self.input_dilution.setRange(0.0001, 10000)
        self.input_dilution.setDecimals(4)
        self.input_dilution.setValue(1.0)
        layout.addRow("Dilution Factor:", self.input_dilution)
        
        self.btn_calc = QPushButton("Calculate")
        self.btn_calc.clicked.connect(self.calculate)
        layout.addRow(self.btn_calc)
        
        self.lbl_result = QLabel("Result: - cells/mL")
        self.lbl_result.setStyleSheet("font-weight: bold; font-size: 16px; color: #2196F3;")
        layout.addRow(self.lbl_result)
        
        self.lbl_info = QLabel("Formula: (Count / Squares) * Dilution * 10,000")
        self.lbl_info.setStyleSheet("color: gray; font-size: 10px;")
        layout.addRow(self.lbl_info)
        
    def calculate(self):
        squares = self.input_squares.value()
        dilution = self.input_dilution.value()
        
        if squares <= 0:
            return
            
        # Standard formula for Neubauer chamber (depth 0.1mm)
        conc = (self.base_count / squares) * dilution * 10000
        self.lbl_result.setText(f"{conc:,.0f} cells/mL")

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Microalgae Counter")
        self.resize(1200, 800)
        
        self.current_image_path = None
        self.current_detections = None
        self.current_original_img = None
        
        # Determine the path to the model file
        if getattr(sys, 'frozen', False):
            # If running as compiled exe
            if hasattr(sys, '_MEIPASS'):
                # PyInstaller --onefile mode: content is in temp dir
                application_path = sys._MEIPASS
            else:
                # PyInstaller --onedir mode: content is in exe dir
                application_path = os.path.dirname(sys.executable)
        else:
            # If running as script
            application_path = os.path.dirname(os.path.abspath(__file__))
            
        # Try best.onnx first, then search for any .onnx file
        self.model_path = os.path.join(application_path, "best.onnx")
        
        # Fallback: Check next to executable if not found in bundle (useful if user wants to override)
        if not os.path.exists(self.model_path) and getattr(sys, 'frozen', False):
             exe_dir = os.path.dirname(sys.executable)
             external_model = os.path.join(exe_dir, "best.onnx")
             if os.path.exists(external_model):
                 self.model_path = external_model

        if not os.path.exists(self.model_path):
            import glob
            pt_files = glob.glob(os.path.join(application_path, "*.onnx"))
            if pt_files:
                self.model_path = pt_files[0]
        
        self.init_ui()
        self.lbl_model.setText(f"Model: {os.path.basename(self.model_path)}")
        
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
        self.view.setRenderHint(QPainter.RenderHint.Antialiasing)
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
        
        # Zoom Controls
        zoom_layout = QHBoxLayout()
        self.btn_zoom_in = QPushButton("Zoom In (+)")
        self.btn_zoom_in.clicked.connect(self.zoom_in)
        self.btn_zoom_out = QPushButton("Zoom Out (-)")
        self.btn_zoom_out.clicked.connect(self.zoom_out)
        zoom_layout.addWidget(self.btn_zoom_in)
        zoom_layout.addWidget(self.btn_zoom_out)
        right_layout.addLayout(zoom_layout)
        
        right_layout.addStretch()
        
        self.btn_load_model = QPushButton("Load Model (.pt)")
        self.btn_load_model.clicked.connect(self.load_model_file)
        right_layout.addWidget(self.btn_load_model)
        
        self.lbl_model = QLabel("Model: best.pt")
        right_layout.addWidget(self.lbl_model)
        
        # Confidence Threshold
        conf_layout = QHBoxLayout()
        conf_layout.addWidget(QLabel("Confidence:"))
        self.spin_conf = QDoubleSpinBox()
        self.spin_conf.setRange(0.01, 1.0)
        self.spin_conf.setSingleStep(0.05)
        self.spin_conf.setValue(0.05) # Default from user request
        conf_layout.addWidget(self.spin_conf)
        right_layout.addLayout(conf_layout)
        
        # Show Labels Toggle
        self.chk_show_labels = QCheckBox("Show Labels")
        self.chk_show_labels.setChecked(True)
        self.chk_show_labels.stateChanged.connect(self.toggle_labels)
        right_layout.addWidget(self.chk_show_labels)
        
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
        
        # Calculator Buttons
        calc_layout = QVBoxLayout()
        
        self.btn_calc = QPushButton("Hemocytometer Calculator")
        self.btn_calc.setStyleSheet("background-color: #2196F3; color: white; padding: 8px;")
        self.btn_calc.clicked.connect(self.open_calculator)
        calc_layout.addWidget(self.btn_calc)
        
        self.btn_avg_calc = QPushButton("Average & Calculate (Selected Files)")
        self.btn_avg_calc.setStyleSheet("background-color: #9C27B0; color: white; padding: 8px;")
        self.btn_avg_calc.clicked.connect(self.calculate_average)
        self.btn_avg_calc.setToolTip("Select multiple files in the list to calculate average count")
        calc_layout.addWidget(self.btn_avg_calc)
        
        right_layout.addLayout(calc_layout)
        
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
            self.file_list.setSelectionMode(QListWidget.SelectionMode.ExtendedSelection) # Enable multi-selection
            extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff', '.webp']
            for f in os.listdir(folder):
                if any(f.lower().endswith(ext) for ext in extensions):
                    self.file_list.addItem(os.path.join(folder, f))

    def load_image(self, item):
        # Reset drawing state first to prevent crash
        if self.btn_draw_roi.isChecked():
            self.btn_draw_roi.setChecked(False)
            self.toggle_drawing()
        else:
            # Ensure scene drawing state is reset even if button wasn't checked
            self.scene.drawing = False
            self.view.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)
            
        path = item.text()
        self.current_image_path = path
        
        # Clear previous results
        self.current_detections = None
        self.current_original_img = None
        self.lbl_result.setText("Count: -")
        
        # Read with opencv to ensure we have data for scene
        img = cv2.imread(path)
        if img is not None:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            self.scene.set_image(img)
            self.view.fitInView(self.scene.sceneRect(), Qt.AspectRatioMode.KeepAspectRatio)
            self.btn_draw_roi.setChecked(False)
            self.scene.drawing = False # Explicitly stop drawing

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

    def zoom_in(self):
        self.view.scale(1.2, 1.2)

    def zoom_out(self):
        self.view.scale(1/1.2, 1/1.2)

    def open_calculator(self):
        # Get current count from label if possible
        text = self.lbl_result.text()
        count = 0
        if "Count:" in text:
            try:
                count = int(text.split(":")[1].strip())
            except:
                pass
        
        dlg = HemocytometerDialog(count, self)
        dlg.exec()
        
    def calculate_average(self):
        selected_items = self.file_list.selectedItems()
        if not selected_items:
            QMessageBox.warning(self, "Warning", "Please select multiple images from the list first.")
            return
            
        if not os.path.exists(self.model_path):
            QMessageBox.warning(self, "Warning", f"Model file not found: {self.model_path}")
            return
            
        # Create a progress dialog since this might take a while
        progress = QProgressBar(self)
        progress.setRange(0, len(selected_items))
        progress.setValue(0)
        progress.setWindowTitle("Processing Batch...")
        progress.show()
        
        counts = []
        detector = MicroalgaeDetector(self.model_path)
        conf = self.spin_conf.value()
        roi = self.scene.get_roi_points() # Use current ROI for all images
        
        # Note: This runs in main thread for simplicity, but could freeze UI for large batches
        # For better UX, this should be moved to a worker thread, but keeping it simple for now
        try:
            for i, item in enumerate(selected_items):
                path = item.text()
                detections, _ = detector.predict(path, roi, conf)
                counts.append(len(detections))
                progress.setValue(i + 1)
                QApplication.processEvents() # Keep UI responsive
                
            progress.close()
            
            dlg = HemocytometerDialog(counts, self)
            dlg.exec()
            
        except Exception as e:
            progress.close()
            QMessageBox.critical(self, "Error", str(e))

    def load_model_file(self):
        path, _ = QFileDialog.getOpenFileName(self, "Select Model", "", "Model Files (*.onnx)")
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
        conf = self.spin_conf.value()
        
        self.worker = InferenceWorker(self.model_path, self.current_image_path, roi, conf)
        self.worker.progress.connect(lambda s: self.progress_bar.setFormat(s))
        self.worker.finished.connect(self.on_inference_finished)
        
        self.progress_bar.setVisible(True)
        self.progress_bar.setRange(0, 0) # Indeterminate
        self.btn_run.setEnabled(False)
        self.worker.start()

    def on_inference_finished(self, result_img, original_img, detections, count, error):
        self.progress_bar.setVisible(False)
        self.btn_run.setEnabled(True)
        
        if error:
            QMessageBox.critical(self, "Error", error)
            return
            
        self.current_original_img = original_img
        self.current_detections = detections
        self.lbl_result.setText(f"Count: {count}")
        
        # Apply current toggle state
        self.redraw_results()
        
    def toggle_labels(self):
        if self.current_original_img is not None and self.current_detections is not None:
            self.redraw_results()
            
    def redraw_results(self):
        if self.current_original_img is None:
            return
            
        # We need to use the detector's draw function, but we don't want to reload the model
        # So we'll just use a static method or instantiate a lightweight version if needed
        # Or better, just import the class and use it if the method was static, but it's not.
        # However, draw_results doesn't use self.model, so we can just create an instance cheaply
        # or refactor. For now, creating an instance is cheap enough as model load is lazy in YOLO? 
        # No, YOLO loads on init. 
        # Let's just manually call the method if we can, or move draw_results to be static.
        # Actually, we can just use the worker's logic or similar.
        # Simplest: Just re-implement drawing here or make draw_results static.
        # Let's make a temporary detector instance just for drawing? No, that loads the model.
        # Let's just use cv2 here directly since we have the logic.
        
        # Wait, I can just make draw_results static in inference.py? 
        # Let's just copy the drawing logic here to avoid reloading model.
        
        img = self.current_original_img.copy()
        # Convert RGB to BGR for OpenCV drawing if needed, but we have RGB here.
        # The original_img from worker is RGB.
        # OpenCV drawing functions work on any array, but colors are BGR usually.
        # If our image is RGB, (0, 255, 0) is Green.
        
        draw_labels = self.chk_show_labels.isChecked()
        
        if draw_labels and self.current_detections:
            for i, box in enumerate(self.current_detections):
                x1, y1, x2, y2, conf = box
                # Draw rectangle (Green)
                cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                
                # Draw label number
                label = str(i + 1)
                font_scale = 0.6
                thickness = 2
                (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
                
                # Draw background
                cv2.rectangle(img, (int(x1), int(y1) - 20), (int(x1) + w, int(y1)), (0, 255, 0), -1)
                # Text (Black)
                cv2.putText(img, label, (int(x1), int(y1) - 5), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), thickness)
        
        self.scene.set_image(img)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())
