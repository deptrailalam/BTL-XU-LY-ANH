import sys
import numpy as np
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QLabel, QPushButton, QFileDialog, QMessageBox, QHBoxLayout, QFrame
from PyQt5.QtGui import QPixmap, QFont
from PyQt5.QtCore import Qt
from PIL import Image
from keras.models import load_model


class ImageClassifier(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()
        try:
            self.model = load_model('mobilenetv2_animals.h5')  # Load model
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Không tải được mô hình: {e}")
            sys.exit(1)

    def initUI(self):
        self.setWindowTitle('Trình phân loại hình ảnh động vật')
        self.setGeometry(100, 100, 900, 700)

        # Main Layout
        layout = QVBoxLayout()
        layout.setAlignment(Qt.AlignCenter)

        # Title
        title = QLabel("Trình phân loại hình ảnh động vật", self)
        title.setAlignment(Qt.AlignCenter)
        title.setStyleSheet("""
            font-size: 28px; font-weight: bold; 
            color: white; padding: 15px; 
            background: #3A6EA5; border-radius: 10px;
        """)
        layout.addWidget(title)

        # Instruction
        guide = QLabel("Step 1: Chọn ảnh |  Step 2: Nhấn 'Dự đoán'", self)
        guide.setAlignment(Qt.AlignCenter)
        guide.setStyleSheet("font-size: 16px; color: #444; margin: 10px 0;")
        layout.addWidget(guide)

        # Image Preview
        self.image_label = QLabel("Không có ảnh nào", self)
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setStyleSheet("""
            border: 3px dashed #999; 
            padding: 20px; color: #666; 
            background: #F5F5F5; border-radius: 10px;
        """)
        self.image_label.setFixedSize(350, 350)
        layout.addWidget(self.image_label, alignment=Qt.AlignCenter)

        # Image Details
        self.image_info = QLabel("", self)
        self.image_info.setAlignment(Qt.AlignCenter)
        self.image_info.setStyleSheet("color: #333; font-size: 14px; margin: 10px;")
        layout.addWidget(self.image_info, alignment=Qt.AlignCenter)

        # Buttons
        btn_layout = QHBoxLayout()

        self.btn_load = QPushButton("Chọn ảnh", self)
        self.btn_load.setStyleSheet("""
            background: #4CAF50; color: white; 
            font-size: 16px; padding: 10px; border-radius: 10px;
        """)
        self.btn_load.clicked.connect(self.load_image)
        btn_layout.addWidget(self.btn_load)

        self.btn_predict = QPushButton("Nhận diện", self)
        self.btn_predict.setStyleSheet("""
            background: #2196F3; color: white; 
            font-size: 16px; padding: 10px; border-radius: 10px;
        """)
        self.btn_predict.clicked.connect(self.predict_image)
        btn_layout.addWidget(self.btn_predict)

        layout.addLayout(btn_layout)

        # Prediction Result
        self.result_label = QLabel("Kết quả dự đoán sẽ được hiển thị ở đây", self)
        self.result_label.setAlignment(Qt.AlignCenter)
        self.result_label.setStyleSheet("""
            font-size: 18px; color: white; 
            background: #FF7043; padding: 10px; 
            border-radius: 10px; margin-top: 20px;
        """)
        self.result_label.setFixedHeight(100)
        layout.addWidget(self.result_label, alignment=Qt.AlignCenter)

        self.setLayout(layout)

        # Background Gradient
        self.setStyleSheet("""
            QWidget {
                background: qlineargradient(
                    x1:0, y1:0, x2:1, y2:1, 
                    stop:0 #6A85B6, stop:1 #BAC8E0
                );
            }
        """)

    def load_image(self):
        options = QFileDialog.Options()
        file_name, _ = QFileDialog.getOpenFileName(self, "Choose Image", "", "Images (*.png *.jpg *.jpeg *.bmp)", options=options)
        if file_name:
            self.display_image(file_name)

    def display_image(self, file_name):
        self.image_label.setPixmap(QPixmap(file_name).scaled(350, 350, Qt.KeepAspectRatio))
        self.image_path = file_name
        img = Image.open(file_name)
        self.image_info.setText(f"Image: {file_name.split('/')[-1]} | Size: {img.size} | Format: {img.format}")

    def predict_image(self):
        if not hasattr(self, 'image_path'):
            QMessageBox.warning(self, "Warning", "Please select an image first!")
            return

        try:
            # Labels
            labels = ["Bird", "Cat", "Chicken", "Dog", "Elephant", "Butterfly", "Horse", "Spider", "Turtle"]

            # Process image
            img = Image.open(self.image_path).resize((224, 224))
            img_array = np.expand_dims(np.array(img) / 255.0, axis=0)

            # Prediction
            predictions = self.model.predict(img_array)
            pred = np.argmax(predictions, axis=-1)[0]
            confidence = predictions[0][pred] * 100

            # Display result
            result = labels[pred] if pred < len(labels) else "Unknown"
            detailed_result = "\n".join(
                [f"{labels[i]}: {predictions[0][i] * 100:.2f}%" for i in range(len(labels))]
            )
            self.result_label.setText(f"Prediction: {result}\nConfidence: {confidence:.2f}%\n\n{detailed_result}")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Đã xảy ra lỗi trong quá trình dự đoán: {e}")


if __name__ == '__main__':
    app = QApplication(sys.argv)
    classifier = ImageClassifier()
    classifier.show()
    sys.exit(app.exec_())
