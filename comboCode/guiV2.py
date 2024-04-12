import sys
import os
import platform
from PyQt5.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QLabel, QPushButton, QWidget, QListWidget
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import QTimer, Qt
from PyQt5.QtMultimedia import *
from PyQt5.QtWidgets import *
import cv2
import qr
from designerUI import *

class MainWindow(QMainWindow, Ui_MainWindow):
    def __init__(self):
        super().__init__()
        self.setupUi(self)
        self.cmtx, self.dist = qr.read_camera_parameters()
        self.camera_index = self.find_realsense_camera()
        self.return_feature_on = False
        self.initUI()

    def find_realsense_camera(self):
        available_cameras = QCameraInfo.availableCameras()
        if platform.machine() == "aarch64":
            return 4
        
        for i, camera in enumerate(available_cameras):
            if "Intel(R) RealSense(TM) Depth Camera 435 with RGB Module RGB" in camera.description():
                print("CAMERA FOUND: ", i)
                return i
        return -1
    
    def initUI(self):
        #creating zoom factor for camera feed
        self.zoom_factor = 1.0
        # Main vertical layout for the central widget
        mainLayout = QVBoxLayout()

        # Add stretch to center the camera feed label vertically
        mainLayout.addStretch()
        
        # Camera feed label
        mainLayout.addWidget(self.cameraFeedLabel)
        self.cameraFeedLabel.setAlignment(QtCore.Qt.AlignCenter)
        # Add stretch to center the camera feed label vertically
        #mainLayout.addStretch()

        # Image modifier button layout
        zoomLayout = QHBoxLayout()

        zoomLayout.addWidget(self.incZoomBtn)
        zoomLayout.addWidget(self.decZoomBtn)

        mainLayout.addLayout(zoomLayout)

        # Horizontal layout for the buttons
        buttonsLayout = QHBoxLayout()

        # Add the buttons to the horizontal layout
        buttonsLayout.addWidget(self.saveBtn)
        buttonsLayout.addWidget(self.returnBtn)
        buttonsLayout.addWidget(self.overlayBtn)

        # Add the buttons layout to the main layout
        mainLayout.addLayout(buttonsLayout)

        # Horizontal layout for status and indication of tracking information
        indicatorLayout = QHBoxLayout()

        indicatorLayout.addWidget(self.currVectorsLabel)
        indicatorLayout.addWidget(self.savedVectorsLabel)

        # Add the buttons layout to the main layout
        mainLayout.addLayout(indicatorLayout)

        rvecLayout = QHBoxLayout()

        rvecLayout.addWidget(self.currRvecs)
        rvecLayout.addWidget(self.savedRvecs)

        # Add the rvec layout to the main layout
        mainLayout.addLayout(rvecLayout)

        tvecLayout = QHBoxLayout()

        tvecLayout.addWidget(self.currTvecs)
        tvecLayout.addWidget(self.savedTvecs)

        # Add the tvec layout to the main layout
        mainLayout.addLayout(tvecLayout)

        # Status label
        mainLayout.addWidget(self.statusLabel)
        mainLayout.addWidget(self.percentProgressBar)

        # Set the main layout as the layout for the central widget
        self.centralwidget.setLayout(mainLayout)

    #------------------ Connecting funtionality of button---------------------
        self.saveBtn.clicked.connect(self.save_image)
        self.returnBtn.clicked.connect(self.toggle_return_feature)
        self.incZoomBtn.clicked.connect(self.increase_zoom)
        self.decZoomBtn.clicked.connect(self.decrease_zoom)

        # Set selectability if using a comboBox
        #self.listWidget.currentIndexChanged.connect(self.set_saved_vectors)

        #set selectability if using a list tree
        self.listWidget2.clicked.connect(self.set_saved_vectors)

    #------------------ Staring camera ---------------------
        # Begin camera feed
        if self.camera_index != -1:
            self.cap = cv2.VideoCapture(self.camera_index)
            self.timer = QTimer(self)
            self.timer.timeout.connect(self.update_frame)
            self.timer.start(30)
        else:
            print("Intel RealSense camera not found.")

        self.display_saved_images()

    '''def update_frame(self):
        ret, frame = self.cap.read()
        if ret:
            # Process frame with camera parameters
            processed_frame = qr.process_frame(frame, self.cmtx, self.dist)

            # Apply the return feature if it's active
            if self.return_feature_on:
                matched, percentage = qr.returnToLoc(self)
                self.percentProgressBar = percentage

        self.display_image(processed_frame)'''

    def update_frame(self):
        ret, frame = self.cap.read()
        if ret:
            # Process frame with camera parameters
            processed_frame = qr.process_frame(frame, self.cmtx, self.dist, zoom_factor=self.zoom_factor)

            # Update current rvec and tvec labels
            #self.currRvecs.setText(f"Rotation: [{qr.current_rvec[0][0]:.2f}, {qr.current_rvec[1][0]:.2f}, {qr.current_rvec[2][0]:.2f}]")
            #self.currTvecs.setText(f"Translation: [{qr.current_tvec[0][0]:.2f}, {qr.current_tvec[1][0]:.2f}, {qr.current_tvec[2][0]:.2f}]")

            # Apply the return feature if it's active
            if self.return_feature_on:
                matched, percentage = qr.returnToLoc(self)
                self.percentProgressBar.setValue(percentage)
                if matched:
                    self.statusLabel.setText("Status: Matched")
                    self.statusLabel.setStyleSheet("color: green;")
                else:
                    self.statusLabel.setText("Status: Not Matched")
                    self.statusLabel.setStyleSheet("color: red;")

        self.display_image(processed_frame)

    def display_image(self, frame):
        qformat = QImage.Format_RGB888
        out_image = QImage(frame, frame.shape[1], frame.shape[0], frame.strides[0], qformat)
        out_image = out_image.rgbSwapped()
        self.cameraFeedLabel.setPixmap(QPixmap.fromImage(out_image))

    def display_saved_images(self):
        #self.listWidget2.clear()
        image_files = [f for f in os.listdir('savedImages/images') if f.endswith('.png')]

        # Sort the image files in increaseing order
        image_files_sorted = sorted(image_files, key=lambda x: int(x.split('guiSave')[1].split('.')[0]))

        for file in image_files_sorted:
            self.listWidget2.addItem(file)

    def save_image(self):
        ret, frame = self.cap.read()
        if ret:
            qr.saveImageNLoc(qr.current_rvec, qr.current_tvec, frame)

    def toggle_return_feature(self):
        self.return_feature_on = not self.return_feature_on
        self.display_saved_images()
        if self.return_feature_on:
            self.returnBtn.setText("Return: On")
        else:
            self.returnBtn.setText("Return: Off")
            self.savedRvecs.setText("Rvec: N/A")
            self.savedTvecs.setText("Tvec: N/A")

    def decrease_zoom(self):
        # Increase zoom factor and update camera feed
        self.zoom_factor = max(self.zoom_factor - 0.1, 1.0)  
        print(f"Zoom factor: {self.zoom_factor}")
        pass

    def increase_zoom(self):
        # Decrease zoom factor and update camera feed
        self.zoom_factor = min(self.zoom_factor + 0.1, 5.0)  
        print(f"Zoom factor: {self.zoom_factor}")
        pass
    
    #set display rvec and tvec of saved qr-code if using a comboBox to display
    '''def set_saved_vectors(self, index):
        img_name = self.listWidget.itemText(index)
        qr.saved_tvec, qr.saved_rvec = qr.getInitialPoints(img_name)
        if qr.saved_tvec is not None and qr.saved_rvec is not None:
            self.savedRvecs.setText(f"Rotation: [{qr.saved_rvec[0][0]:.2f}, {qr.saved_rvec[1][0]:.2f}, {qr.saved_rvec[2][0]:.2f}]")
            self.savedTvecs.setText(f"Translation: [{qr.saved_tvec[0][0]:.2f}, {qr.saved_tvec[1][0]:.2f}, {qr.saved_tvec[2][0]:.2f}]")
        else:
            print("Error: Invalid saved vectors for the selected image.")'''
    #
    #set display rvec and tvec of saved qr-code if using a list tree to display
    def set_saved_vectors(self, item):
        img_name = self.itemText(item)
        qr.saved_tvec, qr.saved_rvec = qr.getInitialPoints(img_name)
        if qr.saved_tvec is not None and qr.saved_rvec is not None:
            self.savedRvecs.setText(f"Rotation: [{qr.saved_rvec[0][0]:.2f}, {qr.saved_rvec[1][0]:.2f}, {qr.saved_rvec[2][0]:.2f}]")
            self.savedTvecs.setText(f"Translation: [{qr.saved_tvec[0][0]:.2f}, {qr.saved_tvec[1][0]:.2f}, {qr.saved_tvec[2][0]:.2f}]")
        else:
            print("Error: Invalid saved vectors for the selected image.")

if __name__ == '__main__':
    app = QApplication(sys.argv)
    mainWindow = MainWindow()
    mainWindow.showMaximized()
    mainWindow.show()
    sys.exit(app.exec_())