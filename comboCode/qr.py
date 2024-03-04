import cv2 as cv
import numpy as np
import sys
import os
import ast

DIFF_CONST = 0.03
retImgSaved = False

current_rvec = None
current_tvec = None
saved_rvec = None
saved_tvec = None
#=========================================================================================================
#Function to get the homeImage coords
def getInitialPoints():
    fname = input('Enter name of Base Image without extension\n>')
    target_line_prefix = fname + ':'
    
    with open('savedImages/imgData/Data.txt', 'r') as file:
        lines = file.readlines()
        for line in lines:
            if line.startswith(target_line_prefix):
                print('img data found!\n')
                data_str = line.split(':', 1)[1].strip()
                try:
                    arrays = ast.literal_eval(data_str)
                    if len(arrays) == 2:
                        #found and extracted the arrays
                        print("Arrays extracted:", arrays)
                        return arrays[0], arrays[1]
                except ValueError as e:
                    print("Error parsing data:", e)
                    return None, None
    
    print("Image data not found.")
    return None, None
#=========================================================================================================

def rotation_vector_to_matrix(vec):
    R, _ = cv.Rodrigues(vec)
    return R
#=========================================================================================================
def angular_difference(vec1, vec2):
    R1 = rotation_vector_to_matrix(vec1)
    R2 = rotation_vector_to_matrix(vec2)
    R_diff = np.dot(R1, R2.T)
    trace = np.trace(R_diff)
    angle = np.arccos((trace - 1) / 2.0)
    return np.degrees(angle)
#=========================================================================================================
def calculate_percentage_difference(saved_tvec, live_tvec, saved_rvec, live_rvec, threshold= DIFF_CONST):
    distance_diff = np.linalg.norm(saved_tvec - live_tvec)
    angle_diff = angular_difference(saved_rvec, live_rvec)
    
    # Calculate the percentage difference based on the threshold
    distance_percentage = min((distance_diff / threshold) * 100, 100)
    angle_percentage = min((angle_diff / threshold) * 100, 100)
    
    # Use the maximum of distance and angle percentages to ensure 100% accuracy only if both are within the threshold
    total_percentage_diff = max(distance_percentage, angle_percentage)
    return 100 - total_percentage_diff  # Return 100% if within the threshold, otherwise decrease proportionally

#=========================================================================================================
def read_camera_parameters(filepath = 'camera_parameters/intrinsic.dat'):

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
def get_qr_coords(cmtx, dist, points):

    #Selected coordinate points for each corner of QR code.
    qr_edges = np.array([[0,0,0],
                         [0,1,0],
                         [1,1,0],
                         [1,0,0]], dtype = 'float32').reshape((4,1,3))

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
    
def saveImageNLoc( rvecSaved,tvecSaved, img):
    dirPath = r'savedImages/images'
    count = 0

    savedDat = [tvecSaved.flatten().tolist(),rvecSaved.flatten().tolist()]#savedDat = [list(tvecSaved),list(rvecSaved)] 

    for path in os.listdir(dirPath):
        if os.path.isfile(os.path.join(dirPath, path)):
            count +=1
            print('File count: ', count)

    save_path0 = f'savedImages/images/guiSave{count}.png'
    cv.imwrite(save_path0, img)
    #count +=1
            
    with open('savedImages/imgData/Data.txt', 'a') as f:
        f.write("img" + str(count) + ": " + str(savedDat) + "\n")
        print('DAT written to file')
                    
#=========================================================================================================
def process_frame(frame, cmtx, dist):
    global current_rvec, current_tvec

    qr = cv.QRCodeDetector()
    #img = cv.resize(img, (0,0), fx=1, fy=1) 
    ret_qr, points = qr.detect(frame)

    if ret_qr:
        axis_points, rvec, tvec = get_qr_coords(cmtx, dist, points)

        if len(axis_points) > 0:
            axis_points = axis_points.reshape((4, 2))

            origin = (int(axis_points[0][0]), int(axis_points[0][1]))

            current_rvec = rvec
            current_tvec = tvec

            for p, c in zip(axis_points[1:], [(255, 0, 0), (0, 255, 0), (0, 0, 255)]):
                    p = (int(p[0]), int(p[1]))

                        #Sometimes qr detector will make a mistake and projected point will overflow integer value. We skip these cases. 
                    #if origin[0] > 5*img.shape[1] or origin[1] > 5*img.shape[1]:break
                    #if p[0] > 5*img.shape[1] or p[1] > 5*img.shape[1]:break

                    #cv.line(img, origin, p, c, 5)


            # Add more processing as needed, such as drawing text or other markers

    return frame
                    
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

    input_source = 'media/test.mp4'
    if len(sys.argv) > 1:
        input_source = int(sys.argv[1])

    show_axes(cmtx, dist, input_source)