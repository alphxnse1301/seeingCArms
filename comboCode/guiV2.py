import sys
import os
import platform
import numpy as np
from PyQt5.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QLabel, QPushButton, QWidget, QListWidget, QAction
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import QTimer, Qt, QSize, pyqtSignal
from PyQt5.QtMultimedia import *
from PyQt5.QtWidgets import *
import cv2
from PIL import Image, ImageQt
from designerUI import *
import qr



class ImageListItem(QWidget):
    show_image_signal = pyqtSignal(str)

    def __init__(self, img_name, parent=None):
        super().__init__(parent)
        self.layout = QHBoxLayout(self)
        
        self.img_name = img_name
        self.img_label = QLabel(img_name)
        self.delete_btn = QPushButton("Delete")
        self.show_btn = QPushButton("Show")
        
        self.layout.addWidget(self.img_label)
        self.layout.addWidget(self.show_btn)
        self.layout.addWidget(self.delete_btn)
        
        self.delete_btn.clicked.connect(self.delete_image)
        self.show_btn.clicked.connect(lambda: self.show_image_signal.emit(self.img_name))

    def delete_image(self):
        try:
            # Delete the image file
            deletePath = qr.getContinousPath() / 'Images'
            (deletePath / self.img_name).unlink()

            # Remove the image's data from Data.txt
            deleteDataPath = qr.getContinousPath() / 'imgData' / 'Data.txt'
            with open(deleteDataPath, 'r') as file:
                lines = file.readlines()
            with open(deleteDataPath, 'w') as file:
                for line in lines:
                    if not line.startswith(self.img_name + ':'):
                        file.write(line)

            mainWindow = self.get_main_window()
            if mainWindow:
                mainWindow.display_saved_images()
        except Exception as e:
            print(f"Error deleting image or updating list: {e}")

    
#helper function to call the functions from the main window class 
    def get_main_window(self):
        current_widget = self
        while current_widget is not None:
            if isinstance(current_widget, MainWindow):
                return current_widget
            current_widget = current_widget.parent()
        return None

    def show_image(self):
        previewPath = qr.getContinousPath() / 'images'
        img_path = os.path.join(previewPath, self.img_name)
        img = Image.open(img_path)
        cv2.imshow(f'{self.img_name}', img)  # This uses the default image viewer of the system


class MainWindow(QMainWindow, Ui_MainWindow):
    def __init__(self):
        super().__init__()
        self.setupUi(self)
        self.cmtx, self.dist = qr.read_camera_parameters()
        self.camera_index = self.find_realsense_camera()
        self.return_feature_on = False
        self.overlay_on = False
        self.saved_img_name = None
        self.displayVectors = True
        #self.returning = False
        self.initUI()
        

    def find_realsense_camera(self):
        available_cameras = QCameraInfo.availableCameras()
        if platform.machine() == "aarch64":
            return 4
        
        for i, camera in enumerate(available_cameras):
            if "Intel(R) RealSense(TM) Depth Camera 435 with RGB Module RGB" in camera.description():
                print("CAMERA FOUND: ", i)
                return i
            else:
                bundle_dir = os.path.abspath(os.path.dirname('media')) 
                adjustLblPath = os.path.join(bundle_dir, 'test.mp4')
                
                return adjustLblPath
        return -1
    
    def initUI(self):
        #creating zoom factor for camera feed
        self.zoom_factor = 1.0

        # Main vertical layout for the central widget
        '''mainLayout = QVBoxLayout()

        # Add stretch to center the camera feed label vertically
        #mainLayout.addStretch()
        
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
        '''

        #Creation of layout fro frame 1 of horizontal layout
        self.frameLayout = QVBoxLayout(self.frame) 

        #setting camera feeds size
        #self.cameraFeedLabel.setFixedSize(640,480)

        # Add camera feed to the frame's layout
        self.frameLayout.addWidget(self.cameraFeedLabel)
        self.cameraFeedLabel.setAlignment(QtCore.Qt.AlignCenter)

        #setting button sizes and fonts
        button_size = QSize(200, 100)  # Width = 100, Height = 50

        self.saveBtn.setFixedSize(button_size)
        self.returnBtn.setFixedSize(button_size)
        self.overlayBtn.setFixedSize(button_size)
        self.incZoomBtn.setFixedSize(button_size)
        self.decZoomBtn.setFixedSize(button_size)

        # Set a larger font for the buttons
        button_font = self.saveBtn.font() 
        button_font.setPointSize(16)

        self.saveBtn.setFont(button_font)
        self.returnBtn.setFont(button_font)
        self.overlayBtn.setFont(button_font)
        self.incZoomBtn.setFont(button_font)
        self.decZoomBtn.setFont(button_font)

        #create text to display current zoom factor
        self.zoomlabel = QLabel('Zoom: 1.0', self)
        self.zoomlabel.setFont(self.saveBtn.font() )

        #create text to display saved zoom factor
        self.savedZoomLabel = QLabel('Saved Zoom: N/A', self)
        self.savedZoomLabel.setFont(self.saveBtn.font())
        

        #setting saved and curr qr code text size
        text_font = self.saveBtn.font() 
        text_font.setPointSize(16)

        remove_font = self.saveBtn.font() 
        remove_font.setPointSize(1)

        self.currVectorsLabel.setFont(text_font)
        

        if self.displayVectors:
            self.savedVectorsLabel.setFont(text_font)
            self.savedRvecs.setFont(text_font) 
            self.savedTvecs.setFont(text_font)
        else:
            self.savedVectorsLabel.setFont(remove_font)
            self.savedRvecs.setFont(remove_font)
            self.savedTvecs.setFont(remove_font)

        self.currRvecs.setFont(text_font)
        
        self.currTvecs.setFont(text_font)

        self.statusLabel.setFont(text_font)

        imgList_font = self.saveBtn.font() 
        imgList_font.setPointSize(14)

        self.imgListWidget.setFont(imgList_font)

        self.imgListWidget.setMinimumWidth(300)
        self.imgListWidget.setMinimumHeight(750)

    # --- frame_2: Adjusting QListWidget to fill 1/2 ---
        self.frame2Layout = QVBoxLayout(self.frame_2)
        self.imgListWidget.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Expanding)  # Adjusted for expanding
        self.frame2Layout.addWidget(self.imgListWidget)

        self.image_display_label = QLabel()  # Label to display the image
        self.frame2Layout.addWidget(self.image_display_label)

        # This stretch will push the QListWidget to the top
        self.frame2Layout.addStretch(1)

        # --- For frame_1: Centering items ---

        # horizontal layout w/ stretches for centering horizontally
        hCameraLayout = QHBoxLayout()
        hCameraLayout.addStretch(1)
        hCameraLayout.addWidget(self.cameraFeedLabel)
        hCameraLayout.addStretch(1)

        #vertical layout w/ stretches for centering vertically
        vCameraLayout = QVBoxLayout()
        vCameraLayout.addStretch(1)
        vCameraLayout.addLayout(hCameraLayout)
        vCameraLayout.addStretch(1)

        self.frameLayout.addLayout(vCameraLayout)

        # Set the main layout as the layout for the central widget
        self.centralwidget.setLayout(self.horizontalLayout_2)

        zoomLayout0 = QHBoxLayout()
        zoomLayout0.addWidget(self.zoomlabel)
        zoomLayout0.addWidget(self.savedZoomLabel)
        self.frameLayout.addLayout(zoomLayout0)

        # Image zoom button layout
        zoomLayout = QHBoxLayout()
        zoomLayout.addWidget(self.incZoomBtn)
        zoomLayout.addWidget(self.decZoomBtn)
        self.frameLayout.addLayout(zoomLayout)

        # Horizontal layout for the buttons
        buttonsLayout = QHBoxLayout()
        buttonsLayout.addWidget(self.saveBtn)

        buttonsLayout.addWidget(self.returnBtn)
        buttonsLayout.addWidget(self.overlayBtn)
        self.frameLayout.addLayout(buttonsLayout)

        # Adding Status label and progress bar
        self.frameLayout.addWidget(self.statusLabel)
        self.statusLabel.setAlignment(QtCore.Qt.AlignCenter)

        self.frameLayout.addWidget(self.percentProgressBar)

        # Horizontal layout for tracking information labels   
        indicatorLayout = QHBoxLayout()
        
        indicatorLayout.addWidget(self.currVectorsLabel)

        if not self.displayVectors:
            self.currVectorsLabel.setText(f"Camera Adjustments:")

        if self.displayVectors:
            indicatorLayout.addWidget(self.savedVectorsLabel)

        self.frameLayout.addLayout(indicatorLayout)

        # Layout for Rvecs and Tvecs labels
        rvecLayout = QHBoxLayout()
        rvecLayout.addWidget(self.currRvecs)

        if self.displayVectors:
            rvecLayout.addWidget(self.savedRvecs)

        self.frameLayout.addLayout(rvecLayout)

        tvecLayout = QHBoxLayout()
        tvecLayout.addWidget(self.currTvecs)

        if self.displayVectors:
            tvecLayout.addWidget(self.savedTvecs)

        self.frameLayout.addLayout(tvecLayout)

        # Add a stretch to push everything up
        self.frameLayout.addStretch()


    #------------------ Connecting funtionality of button---------------------
        self.saveBtn.clicked.connect(self.save_image)
        self.returnBtn.clicked.connect(self.toggle_return_feature)
        self.incZoomBtn.clicked.connect(self.increase_zoom)
        self.decZoomBtn.clicked.connect(self.decrease_zoom)

        self.overlayBtn.clicked.connect(self.toggle_overlay)

        # Set selectability if using a comboBox
        #self.listWidget.currentIndexChanged.connect(self.set_saved_vectors)

        #set selectability if using a list tree
        self.imgListWidget.itemClicked.connect(self.set_saved_vectors)
        

        # Makes a button in the upper toolbar to delete all of the saved images and image data
        clear_all_Images = QAction("Clear All Data", self)
        clear_all_Images.triggered.connect(self.clearAllImages)
        self.menubar.addAction(clear_all_Images)

        #Makes a button in the upper tool bar for turning on the return guidance features or to displayt the tvecs and rvecs of the home and saved images
        showVectors = QAction("Show Vectors", self)
        showVectors.triggered.connect(self.toggle_vectors)
        #commented out as it doesn't work as expected currently
        #self.menubar.addAction(showVectors)

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

    def update_frame(self):
        ret, frame = self.cap.read()
        if ret:
            # Process frame with camera parameters
            processed_frame = qr.process_frame(frame, self.cmtx, self.dist, zoom_factor=self.zoom_factor, return_tvec=qr.saved_tvec, return_rvec=qr.saved_rvec)

            # Update current rvec and tvec labels

            if self.displayVectors:
                if qr.current_rvec is not None and qr.current_tvec is not None:
                    self.currRvecs.setText(f"Rotation: [{qr.current_rvec[0][0]:.2f}, {qr.current_rvec[1][0]:.2f}, {qr.current_rvec[2][0]:.2f}]")
                    self.currTvecs.setText(f"Translation: [{qr.current_tvec[0][0]:.2f}, {qr.current_tvec[1][0]:.2f}, {qr.current_tvec[2][0]:.2f}]")
                else:
                    self.currRvecs.setText("Rotation: N/A")
                    self.currTvecs.setText("Translation: N/A")

            # Apply the return feature if it's active
            if self.return_feature_on and self.overlay_on:
                try:
                    img_path = qr.getContinousPath() / 'Images'
                    full_path = os.path.join(img_path, self.saved_img_name)
                    # Only overlay if both return_feature and overlay are active
                    overlay_image = Image.open(full_path)  # Load the saved QR code image
                    overlay_image = overlay_image.convert("RGBA") 

                    overlay_image = overlay_image.resize((frame.shape[1], frame.shape[0]), resample=Image.LANCZOS)

                    # Create a Pillow image from the current frame
                    current_frame_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)).convert("RGBA")

                    # Blend the current frame with the overlay image
                    blended_image = Image.blend(current_frame_image, overlay_image, .25)

                    # Convert the blended image back to a format suitable for OpenCV
                    processed_frame = cv2.cvtColor(np.array(blended_image), cv2.COLOR_RGBA2BGR)
                    
                    matched, percentage = qr.returnToLoc(self)
                    self.percentProgressBar.setValue(int(percentage)) 
                    if matched:
                        self.statusLabel.setText("Status: Matched")
                        self.statusLabel.setStyleSheet("color: green;")

                        #Placing text in regards to the images being matched and no adjustment needed               !!
                        self.currRvecs.setText("Rotations adjust: None")
                        self.currTvecs.setText("Translation adjust: None")

                    else:
                        self.statusLabel.setText("Status: Not Matched")
                        self.statusLabel.setStyleSheet("color: red;")

                        '''if not self.displayVectors:
                            #Placing text in regards to the images being matched and no adjustment needed               !!
                            rGuide,tGuide = qr.returnGuidance(qr.current_rvec, qr.saved_rvec, qr.current_tvec, qr.saved_tvec)
                        
                            # Update the adjust text with the guidance
                            self.currRvecs.setText(f"Rotation adjust: {rGuide}")
                            self.currTvecs.setText(f"Translation adjust: {tGuide}")'''

                except FileNotFoundError:
                    print("ERROR: File not found")
                    self.overlay_on = False
                except TypeError:
                    print("ERROR:  join() type error")
                    self.overlay_on = False
            elif self.return_feature_on:
                matched, percentage = qr.returnToLoc(self)
                self.percentProgressBar.setValue(int(percentage)) 
                if matched:
                    self.statusLabel.setText("Status: Matched")
                    self.statusLabel.setStyleSheet("color: green;")

                    #Placing text in regards to the images being matched and no adjustment needed               !!
                    #self.currRvecs.setText("Rotations adjust: None")
                    #self.currTvecs.setText("Translation adjust: None")

                else:
                    self.statusLabel.setText("Status: Not Matched")
                    self.statusLabel.setStyleSheet("color: red;")

                    #Placing text in regards to the images being matched and no adjustment needed               !!
                    '''if qr.saved_rvec is not None and qr.saved_tvec is not None and not self.displayVectors:
                        rGuide,tGuide = qr.returnGuidance(qr.current_rvec, qr.saved_rvec, qr.current_tvec, qr.saved_tvec)
                        # Update the adjust text with the guidance
                        self.currRvecs.setText(f"Rotation adjust: {rGuide}")
                        self.currTvecs.setText(f"Translation adjust: {tGuide}")'''


        self.display_image(processed_frame)

    def display_image(self, frame):
        qformat = QImage.Format_RGB888
        out_image = QImage(frame, frame.shape[1], frame.shape[0], frame.strides[0], qformat)
        out_image = out_image.rgbSwapped()
        self.cameraFeedLabel.setPixmap(QPixmap.fromImage(out_image))

    def display_saved_images(self):
        self.imgListWidget.clear()
        displaySavedPath = qr.getContinousPath() / 'Images'
        #displayPath = qr.getPath(displaySavedPath)
        image_files = [f for f in os.listdir(displaySavedPath) if f.endswith('.png') ]
        # Sort the image files in increaseing order
        image_files_sorted = sorted(image_files, key=lambda x: int(x.split('Image_')[1].split('.')[0]))

        for file in image_files_sorted:
            item = QListWidgetItem(self.imgListWidget)
            item.setData(Qt.UserRole, file)
            custom_widget = ImageListItem(file, self.imgListWidget)
            custom_widget.show_image_signal.connect(self.display_image_in_label)
            item.setSizeHint(custom_widget.sizeHint())
            self.imgListWidget.addItem(item)
            self.imgListWidget.setItemWidget(item, custom_widget)

    def save_image(self):
        self.imgListWidget.clear()
        ret, frame = self.cap.read()
        if ret:
            qr.saveImageNLoc(qr.current_rvec, qr.current_tvec, frame, self.zoom_factor)
            
        self.display_saved_images()

    def toggle_return_feature(self):
        self.return_feature_on = not self.return_feature_on
        #self.display_saved_images()
        if self.return_feature_on:
            #self.returning = True
            qr.returning = True
            self.returnBtn.setText("Return: On")
            if self.saved_img_name is None:
            # No image is selected, select the latest image
                displaySavedPath = qr.getContinousPath() / 'Images'
                image_files = [f for f in os.listdir(displaySavedPath) if f.endswith('.png')]
                if image_files:
                    latest_image = sorted(image_files, key=lambda x: int(x.split('Image_')[1].split('.')[0]))[-1]
                    self.saved_img_name = latest_image
                    qr.saved_tvec, qr.saved_rvec, savedZoom = qr.getInitialPoints(latest_image)
                    # Update UI text for saved vectors
                    if qr.saved_tvec is not None and qr.saved_rvec is not None and self.displayVectors :
                        self.savedRvecs.setText(f"Rotation: [{qr.saved_rvec[0][0]:.2f}, {qr.saved_rvec[1][0]:.2f}, {qr.saved_rvec[2][0]:.2f}]")
                        self.savedTvecs.setText(f"Translation: [{qr.saved_tvec[0][0]:.2f}, {qr.saved_tvec[1][0]:.2f}, {qr.saved_tvec[2][0]:.2f}]")
                        self.savedZoomLabel.setText(f"Saved Zoom: {savedZoom:.1f}")
        else:
            #self.returning = False
            qr.returning = False
            self.returnBtn.setText("Return: Off")
            self.currRvecs.setText("Rotation adjust: None")
            self.currTvecs.setText("Translation adjust: None")
            if self.displayVectors:
                self.savedRvecs.setText("Rvec: N/A")                  
                self.savedTvecs.setText("Tvec: N/A") 
                #self.savedZoomLabel.setText(f"Saved Zoom: {savedZoom:.1f}")                  

    def toggle_overlay(self):
        self.overlay_on = not self.overlay_on
    
    def toggle_vectors(self):
        self.displayVectors = not self.displayVectors

    def decrease_zoom(self):
        # Increase zoom factor and update camera feed
        self.zoom_factor = max(self.zoom_factor - 0.1, 1.0) 
        self.zoomlabel.setText(f"Zoom: {self.zoom_factor:.2f}") 
        print(f"Zoom factor: {self.zoom_factor}")
        pass

    def increase_zoom(self):
        # Decrease zoom factor and update camera feed
        self.zoom_factor = min(self.zoom_factor + 0.1, 5.0)  
        self.zoomlabel.setText(f"Zoom: {self.zoom_factor:.2f}.") 
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
        img_name = item.data(Qt.UserRole)
        self.saved_img_name = img_name
        print(img_name)
        qr.saved_tvec, qr.saved_rvec, savedZoom = qr.getInitialPoints(img_name)
        if qr.saved_tvec is not None and qr.saved_rvec is not None:
            self.savedRvecs.setText(f"Rotation: [{qr.saved_rvec[0][0]:.2f}, {qr.saved_rvec[1][0]:.2f}, {qr.saved_rvec[2][0]:.2f}]")
            self.savedTvecs.setText(f"Translation: [{qr.saved_tvec[0][0]:.2f}, {qr.saved_tvec[1][0]:.2f}, {qr.saved_tvec[2][0]:.2f}]")
            self.savedZoomLabel.setText(f"Saved Zoom: {savedZoom:.1f}")
        else:
            print("Error: Invalid saved vectors for the selected image.")
            self.savedZoomLabel.setText("Saved Zoom: N/A")
    
    # Whne the clear all data menu action is pressed it will remove all the images and data from the savedImages folder then refresh the image list
    def clearAllImages(self):
        qr.clearAllData()
        self.imgListWidget.clear()
        self.display_saved_images()

    # Display the specified image in the QLabel (lower right hand side of GUI)
    def display_image_in_label(self, img_path):
        adjustLblPath = qr.getContinousPath() / 'Images'
        full_path = os.path.join(adjustLblPath, img_path)
        if not os.path.exists(full_path):
            print(f"Image not found at {full_path}")
            return

        pixmap = QPixmap(full_path)
        if pixmap.isNull():
            print(f"Failed to load image at {full_path}")
            return

        scaled_pixmap = pixmap.scaled(self.image_display_label.size(), Qt.KeepAspectRatioByExpanding, Qt.SmoothTransformation)
        self.image_display_label.setPixmap(scaled_pixmap)
        print(f"Image displayed: {img_path}")
    

if __name__ == '__main__':
    app = QApplication(sys.argv)
    mainWindow = MainWindow()
    mainWindow.showMaximized()
    mainWindow.show()
    sys.exit(app.exec_())