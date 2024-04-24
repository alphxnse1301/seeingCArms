import cv2 as cv
import numpy as np
import sys
import os
import ast
import time
from PIL import Image
from pathlib import Path
#import pyrealsense2 as rs

DIFF_CONST = 0.05
retImgSaved = False

current_rvec = None
current_tvec = None
saved_rvec = None
saved_tvec = None

overlay = False

returning = False

savedZoom = None #not implemented currently

""" Get the absolute path to the files for saving images, image data, and ect."""
def getPath(relative_path):
    
    if getattr(sys, 'frozen', False):
        base_path = Path(sys._MEIPASS)
    else:
        base_path = Path(__file__).parent
    return base_path / relative_path
#=========================================================================================================
#Function to get the saved Image coords
def getInitialPoints(imgName):
    savedImgPath = getContinousPath() / 'imgData' / 'Data.txt'

    with open(savedImgPath, 'r') as file:
        lines = file.readlines()
        for line in lines:
            if line.startswith(imgName + ':'):
                data_str = line.split(':', 1)[1].strip()
                try:
                    arrays = ast.literal_eval(data_str)
                    if len(arrays) == 3:  # Ensuring a zoom factor is included for the returning imagae
                        tvec = np.array(arrays[0], dtype=float).reshape((3, 1))
                        rvec = np.array(arrays[1], dtype=float).reshape((3, 1))
                        zoomFactor = arrays[2]
                        print(f"ZOOM: {zoomFactor}")
                        return tvec, rvec, zoomFactor
                except ValueError as e:
                    print("Error parsing data:", e)
                    return None, None, None
    print("Image data not found.")
    return None, None, None
#=========================================================================================================

def rotation_vector_to_matrix(vec):
    if vec is None:
        print("Error: rotation vector is None")
        return None
    if not isinstance(vec, np.ndarray):
        print(f"Error: rotation vector is not a numpy array, it is {type(vec)}")
        return None
    try:
        R, _ = cv.Rodrigues(vec)
        return R
    except Exception as e:
        print(f"Error in Rodrigues conversion: {e}")
        return None
#=========================================================================================================
def angular_difference(vec1, vec2):
    R1 = rotation_vector_to_matrix(vec1)
    R2 = rotation_vector_to_matrix(vec2)
    if R1 is None or R2 is None:
        print("Error: One of the rotation matrices is None")
        return 0
    R_diff = np.dot(R1, R2.T)
    trace = np.trace(R_diff)
    angle = np.arccos((trace - 1) / 2.0)
    return np.degrees(angle)
#=========================================================================================================

def calculate_percentage_difference(saved_tvec, live_tvec, saved_rvec, live_rvec, threshold=DIFF_CONST):
    distance_diff = np.linalg.norm(saved_tvec - live_tvec)
    angle_diff = angular_difference(saved_rvec, live_rvec)
    
    # Calculate the percentage difference based on the threshold
    distance_percentage = min((distance_diff / threshold) * 100, 100)
    angle_percentage = min((angle_diff / threshold) * 100, 100)
    
    # Check if both percentages are 100
    if distance_percentage == 100 and angle_percentage == 100:
        return 100
    else:
        # Return the lower of the two percentages to indicate the degree of matching
        #return min(distance_percentage, angle_percentage)
    
        # Return the higher of the two percentages to indicate the degree of matching
        return max(distance_percentage, angle_percentage)
    
#=========================================================================================================

def returnGuidance(current_rvec, saved_rvec, current_tvec, saved_tvec):
    rvecGuide = "No rotation adjustment needed."
    tvecGuide = "No translation adjustment needed."

    axis_names = ['X-axis (left-right)', 'Y-axis (up-down)', 'Z-axis (forward-backward)']
    rotation_directions = ['clockwise', 'counterclockwise']
    move_directions = ['right', 'left', 'up', 'down', 'backward', 'forward']

   # Handling rotations
    for i, (current, saved) in enumerate(zip( saved_rvec, current_rvec)):
        diff = current - saved
        if abs(diff) > DIFF_CONST:
            direction = rotation_directions[int(diff < 0)]
            rvecGuide = f"Rotate {direction} around the {axis_names[i]} to adjust."
        else:
            rvecGuide = "NONE."


    # Handling translations
    for i, (current, saved) in enumerate(zip( saved_tvec,current_tvec)):
        diff = current - saved
        if abs(diff) > DIFF_CONST:
            index = 2 * i + int(diff < 0)
            tvecGuide = f"Move {move_directions[index]} to adjust along the {axis_names[i]}."
        else:
            tvecGuide = "NONE."

    return rvecGuide, tvecGuide


#=========================================================================================================
def read_camera_parameters():
    filepath = getPath('camera_parameters/intrinsic.dat')
    inf = open(filepath, 'r')

    cmtx = []
    dist = []

    #ignore first line
    line = inf.readline()
    for _ in range(3):
        line = inf.readline().split()
        line = [float(en) for en in line]
        cmtx.append(line)

    #ignore line that says "distortion"
    line = inf.readline()
    line = inf.readline().split()
    line = [float(en) for en in line]
    dist.append(line)

    #cmtx = camera matrix, dist = distortion parameters
    return np.array(cmtx), np.array(dist)
#=========================================================================================================

def saveImageNLoc( rvecSaved,tvecSaved, img, zoomFactor):
    # Retrieve the base path for continuous use
    base_path = getContinousPath()

    # Define the directory path for images
    images_dir = base_path / 'Images'
    images_dir.mkdir(exist_ok=True)  # Ensure the directory exists

    # Count existing files to name the new file uniquely
    count = len([name for name in os.listdir(images_dir) if os.path.isfile(os.path.join(images_dir, name))])

    # Define the save path for the new image
    image_path = images_dir / f'Image_{count}.png'

    # Save the image using OpenCV
    cv.imwrite(str(image_path), img)

    # Define the data file path
    data_file_path = base_path / 'imgData' / 'Data.txt'
    data_file_path.parent.mkdir(exist_ok=True)  # Ensure the directory exists

    # Prepare the data to be saved
    saved_data = [tvecSaved.flatten().tolist(), rvecSaved.flatten().tolist(), zoomFactor]  # Flatten and listify the vectors

    # Append the new data to the Data.txt file
    with open(data_file_path, 'a') as f:
        f.write(f"Image_{count}.png: {saved_data}\n")
        print('Data written to file')
                    
#=========================================================================================================
def get_qr_coords(cmtx, dist, points):

    '''Selected coordinate points for each corner of QR code. USE WHEN QR CODE SIZE UNKNOWN, THIS IS FOR UNVERSIAL SIZES OF QRCODES'''
    qr_edges = np.array([[0,0,0],
                         [0,1,0],
                         [1,1,0],
                         [1,0,0]], dtype = 'float32').reshape((4,1,3))
    
    '''used WHEN THE SIZE OF A QR CODE IS DEFINITIVELY KNOWN CONVERT YOUR QRCODE SIZE TO METERS I THINK
    qr_code_size = 0.05715  # 2.25 inches in meters
    qr_edges = np.array([
        [-qr_code_size/2, -qr_code_size/2, 0],  # Bottom-left corner
        [-qr_code_size/2, qr_code_size/2, 0],   # Top-left corner
        [qr_code_size/2, qr_code_size/2, 0],    # Top-right corner
        [qr_code_size/2, -qr_code_size/2, 0]    # Bottom-right corner
    ], dtype='float32').reshape((4, 1, 3))'''

    #determine the orientation of QR code coordinate system with respect to camera coorindate system.
    ret, rvec, tvec = cv.solvePnP(qr_edges, points, cmtx, dist)

    #Define unit xyz axes. These are then projected to camera view using the rotation matrix and translation vector.
    unitv_points = np.array([[0,0,0], [1,0,0], [0,1,0], [0,0,1]], dtype = 'float32').reshape((4,1,3))
    if ret:
        points, jac = cv.projectPoints(unitv_points, rvec, tvec, cmtx, dist)
        return points, rvec, tvec

    #return empty arrays if rotation and translation values not found
    else: return [], [], []

#=========================================================================================================
    
def process_frame(frame, cmtx, dist, update_callback=None, zoom_factor=1.0, return_tvec = None, return_rvec = None):
    global current_rvec, current_tvec

    # Defined frame rate
    frameRate = 30
    frame_duration = 1.0 / frameRate

    # Start time for frame rate control
    start_time = time.time()

    # Simulate zoom by cropping and resizing
    if zoom_factor > 1.0:
        height, width = frame.shape[:2]
        new_width = int(width / zoom_factor)
        new_height = int(height / zoom_factor)
        x_offset = (width - new_width) // 2
        y_offset = (height - new_height) // 2

        frame = frame[y_offset:y_offset+new_height, x_offset:x_offset+new_width]
        if zoom_factor > 2.0:
            frame = cv.resize(frame, (width, height))#, interpolation=cv.INTER_LINEAR)
        else:
            frame = cv.resize(frame, (width, height))

        #frame = enhance_image(frame)

    qr = cv.QRCodeDetector()
    ret_qr, points = qr.detect(frame)

    if ret_qr:
        axis_points, rvec, tvec = get_qr_coords(cmtx, dist, points)

        if len(axis_points) > 0:
            axis_points = axis_points.reshape((4, 2))
            origin = (int(axis_points[0][0]), int(axis_points[0][1]))
            current_rvec = rvec
            current_tvec = tvec

            if update_callback is not None:
                update_callback(tvec, rvec)
            '''This add arrows of what way it needs to go
            if return_tvec is not None:
                for i, (p, c) in enumerate(zip(axis_points[1:], [(255, 0, 0), (0, 255, 0), (0, 0, 255)])):
                    p = (int(p[0]), int(p[1]))
                    direction = np.sign(return_tvec[i] - tvec[i])
                    scaled_point = (int(origin[0] + (p[0] - origin[0]) * direction),
                                    int(origin[1] + (p[1] - origin[1]) * direction))
                    cv.arrowedLine(frame, origin, scaled_point, c, 5, tipLength=0.3)
            else:
                print("tvec is none")
                for p, c in zip(axis_points[1:], [(255, 0, 0), (0, 255, 0), (0, 0, 255)]):
                    p = (int(p[0]), int(p[1]))
                    cv.line(frame, origin, p, c, 5)'''

            ''' this does scaling of the axis indicators on return'''
            # Scale factor for axis lines based on the difference in tvec
            if return_tvec is not None and returning:
                scale_factors = np.clip(1 - np.abs(return_tvec - tvec) / DIFF_CONST, 0.1, 1)
            else:
                
                scale_factors = [1, 1, 1]

            for i, (p, c) in enumerate(zip(axis_points[1:], [(255, 0, 0), (0, 255, 0), (0, 0, 255)])):
                p = (int(p[0]), int(p[1]))
                scaled_point = (int(origin[0] + (p[0] - origin[0]) * scale_factors[i]),
                                int(origin[1] + (p[1] - origin[1]) * scale_factors[i]))
                cv.line(frame, origin, scaled_point, c, 5)
            
            '''This is to show the roation needed
            if return_rvec is not None:
                angle_diff = angular_difference(current_rvec, return_rvec)
                rotation_scale = np.clip(angle_diff / np.pi, 0, 1)  # Scale based on angular difference
                rotation_direction = np.sign(return_rvec - current_rvec)  # Determine direction of rotation

                # Draw rotation indicators (arrows) to the side of the frame
                frame_height, frame_width = frame.shape[:2]
                indicator_start = (frame_width - 50, frame_height // 2)
                for i in range(3):
                    indicator_end = (
                        indicator_start[0],
                        int(indicator_start[1] + 50 * rotation_scale * rotation_direction[i])
                    )
                    cv.arrowedLine(frame, indicator_start, indicator_end, (255, 255, 255), 2, tipLength=0.3)
                    indicator_start = (indicator_start[0] - 20, indicator_start[1])'''

            '''Standard axis
            for p, c in zip(axis_points[1:], [(255, 0, 0), (0, 255, 0), (0, 0, 255)]):
                    p = (int(p[0]), int(p[1]))

                        #Sometimes qr detector will make a mistake and projected point will overflow integer value. We skip these cases. 
                    #if origin[0] > 5*img.shape[1] or origin[1] > 5*img.shape[1]:break
                    #if p[0] > 5*img.shape[1] or p[1] > 5*img.shape[1]:break

                    cv.line(frame, origin, p, c, 5)'''

    # Frame rate control
    elapsed_time = time.time() - start_time
    time_to_wait = frame_duration - elapsed_time
    if time_to_wait > 0:
        time.sleep(time_to_wait)

    return frame

def enhance_image(frame):
    # Apply sharpening filter
    kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
    sharpened_frame = cv.filter2D(frame, -1, kernel)
    return sharpened_frame
#=========================================================================================================

def returnToLoc(main_frame):
    global current_rvec, current_tvec, saved_rvec, saved_tvec, savedZoom

    if saved_rvec is None or saved_tvec is None:
        print("Saved data is missing")
        return False, 0
    
    if savedZoom is None:
        savedZoom = 1.0
    
    saved_img_path = getContinousPath() / 'Images' / main_frame.saved_img_name
    saved_img = cv.imread(str(saved_img_path))
    if saved_img is not None:
        # Resize saved_img according to its saved zoom factor
        if savedZoom != 1.0:
            height, width = saved_img.shape[:2]
            new_width = int(width / savedZoom)
            new_height = int(height / savedZoom)
            x_offset = (width - new_width) // 2
            y_offset = (height - new_height) // 2
            saved_img = saved_img[y_offset:y_offset+new_height, x_offset:x_offset+new_width]
            saved_img = cv.resize(saved_img, (width, height))

    matched = (abs(saved_tvec[0] - current_tvec[0]) <= DIFF_CONST and abs(saved_tvec[1] - current_tvec[1]) <= DIFF_CONST and abs(saved_tvec[2] - current_tvec[2]) <= DIFF_CONST
        and abs(saved_rvec[0] - current_rvec[0]) <= DIFF_CONST and abs(saved_rvec[1] - current_rvec[1]) <= DIFF_CONST and abs(saved_rvec[2] - current_rvec[2]) <= DIFF_CONST)
    
    if matched:
        percentage = calculate_percentage_difference(saved_tvec, current_tvec, saved_rvec, current_rvec)
        print("--MATCHED--")
        #main_frame.update_status(True)
        return True, percentage
    else:
        percentage = calculate_percentage_difference(saved_tvec, current_tvec, saved_rvec, current_rvec)
        print("--NOT MATCHED--")
        #main_frame.update_status(False)
        return False, percentage
#=========================================================================================================
def clearAllData():
    print("DATA CLEARING...")
    # Relative path to the directory containing images
    dirPath = getContinousPath() / 'Images'

    # Remove all .png files in the directory
    for file in os.listdir(dirPath):
        if file.endswith('.png'):
            fullPath = os.path.join(dirPath, file)  # Get the full path of the file
            os.remove(fullPath)  # Remove the file using the full path

    # Clear the contents of the data file
    dataFilePath = getContinousPath() / 'imgData'/'Data.txt'
    dataFilePath = dataFilePath
    with open(dataFilePath, 'w') as f:
        f.write('')  # Opening in 'w' mode and writing an empty string to truncate the file can also be doen using f.truncate()

#=========================================================================================================

#used for the the clickable app icon, so that the data can be used throguh many sessions
def getContinousPath():
    home = Path.home()
    path = home / 'savedImages'
    images_path = path / 'Images'
    img_data_path = path / 'imgData'
    
    for p in [path, images_path, img_data_path]:
        if not p.exists():
            p.mkdir(parents=True)
    
    return path
#=========================================================================================================
                    
def show_axes(cmtx, dist, in_source):
    cap = cv.VideoCapture(in_source)

    if (not cap.isOpened()): raise IOError("CANNOT OPEN WEBCAM")
    #cap.set(cv.CAP_PROP_FRAME_WIDTH, 640)
    #cap.set(cv.CAP_PROP_FRAME_HEIGHT, 360)

    qr = cv.QRCodeDetector()
    font = cv.FONT_HERSHEY_DUPLEX

# org 
    xOrg = (0, 50)
    xOrg2 = (500, 50)
    yOrg = (0, 100) 
    zOrg = (0, 150) 

    xRotOrg = (0, 200)
    yRotOrg = (0, 250) 
    zRotOrg = (0, 300) 
  
# Font Scale 
    fontScale = 1
   
# Color in BGR 
    color = (255, 255, 255) 
  
# Line thickness of 2 px 
    thickness = 1

    returnToPoint = False
    saveT = [0, 0, 0]
    saveR = [0, 0, 0]

    while True:

        ret, img = cap.read()
        if ret == False: break
        img = cv.resize(img, (0,0), fx=1.5, fy=1.5) 
        ret_qr, points = qr.detect(img)

        if ret_qr:
            axis_points, rvec, tvec = get_qr_coords(cmtx, dist, points)

            #BGR color format
            colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (0,0,0)]

            #check axes points are projected to camera view.
            if len(axis_points) > 0:
                axis_points = axis_points.reshape((4,2))

                origin = (int(axis_points[0][0]),int(axis_points[0][1]) )

                for p, c in zip(axis_points[1:], colors[:3]):
                    p = (int(p[0]), int(p[1]))

                    #Sometimes qr detector will make a mistake and projected point will overflow integer value. We skip these cases. 
                    if origin[0] > 5*img.shape[1] or origin[1] > 5*img.shape[1]:break
                    if p[0] > 5*img.shape[1] or p[1] > 5*img.shape[1]:break

                    #cv.line(img, origin, p, c, 5)

                image = cv.putText(img, "X Translation: " + str(round(float(tvec[0][0]), 2)), xOrg, font,  
                                    fontScale, color, thickness, cv.LINE_AA)

                image = cv.putText(img, "Y Translation: " + str(round(float(tvec[1][0]), 2)), yOrg, font,  
                                    fontScale, color, thickness, cv.LINE_AA)

                image = cv.putText(img, "Z Translation: " + str(round(float(tvec[2][0]), 2)), zOrg, font,  
                                    fontScale, color, thickness, cv.LINE_AA)


                image = cv.putText(img, "X Rotation: " + str(round(abs(float(rvec[0][0])), 2)), xRotOrg, font,  
                                    fontScale, color, thickness, cv.LINE_AA)

                image = cv.putText(img, "Y Rotation: " + str(round(abs(float(rvec[1][0])), 2)), yRotOrg, font,  
                                    fontScale, color, thickness, cv.LINE_AA)

                image = cv.putText(img, "Z Rotation: " + str(round(abs(float(rvec[2][0])), 2)), zRotOrg, font,  
                                    fontScale, color, thickness, cv.LINE_AA)

                if (returnToPoint):

                    '''if((saveT[0] < tvec[0]) and (abs(saveT[0] - tvec[0]) > DIFF_CONST)):
                        image = cv.arrowedLine(image, origin, (int(axis_points[2][0]), int(axis_points[2][1])), color, 5)
                    
                    if((saveT[1] < tvec[1]) and (abs(saveT[1] - tvec[1]) > DIFF_CONST)):
                        image = cv.arrowedLine(image, origin, (int(axis_points[1][0]), int(axis_points[1][1])), color, 5)

                    if((saveT[2] < tvec[2]) and (abs(saveT[2] - tvec[2]) > DIFF_CONST)):
                        image = cv.arrowedLine(image, origin, (int(axis_points[3][0]), int(axis_points[3][1])), color, 5)


                    
                    if((saveT[0] > tvec[0]) and (abs(saveT[0] - tvec[0]) > DIFF_CONST)):
                        image = cv.arrowedLine(image, origin, (int(-axis_points[2][0]), int(axis_points[2][1])), color, 5)
                    
                    if((saveT[1] > tvec[1]) and (abs(saveT[1] - tvec[1]) > DIFF_CONST)):
                        image = cv.arrowedLine(image, origin, (int(-axis_points[1][0]), int(axis_points[1][1])), color, 5)

                    if((saveT[2] > tvec[2]) and (abs(saveT[2] - tvec[2]) > DIFF_CONST)):
                        image = cv.arrowedLine(image, origin, (int(-axis_points[3][0]), int(-axis_points[3][1])), color, 5)'''

                    if (abs(saveT[0] - tvec[0]) <= DIFF_CONST and abs(saveT[1] - tvec[1]) <= DIFF_CONST and abs(saveT[2] - tvec[2]) <= DIFF_CONST
                        and abs(saveR[0] - rvec[0]) <= DIFF_CONST and abs(saveR[1] - rvec[1]) <= DIFF_CONST and abs(saveR[2] - rvec[2]) <= DIFF_CONST):
                        image = cv.putText(img, "RETURNED", xOrg2, font, fontScale, color, thickness, cv.LINE_AA)
                        if(not retImgSaved):
                            cv.imwrite("Returned.jpg", image)
                            retImgSaved = True

        cv.imshow('frame', img)

        k = cv.waitKey(10)
        if k == 27: break #27 is ESC key.
        if k == 83 or k == 115:
            saveT[0] = tvec[0]
            saveT[1] = tvec[1]
            saveT[2] = tvec[2]

            saveR[0] = rvec[0]
            saveR[1] = rvec[1]
            saveR[2] = rvec[2]
            cv.imwrite("SavedImage.jpg", image)
            retImgSaved = False
            returnToPoint = False
        if k == 82 or k == 114: returnToPoint = True

    cap.release()
    cv.destroyAllWindows()

if __name__ == '__main__':

    #read camera intrinsic parameters.
    cmtx, dist = read_camera_parameters()

    input_source = getPath('media/test.mp4')
    if len(sys.argv) > 1:
        input_source = int(sys.argv[1])

    show_axes(cmtx, dist, input_source)