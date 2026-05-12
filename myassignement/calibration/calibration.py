import numpy as np
import cv2
import glob

# ==========================================
# 1. CHECKERBOARD CONFIGURATION
# ==========================================
# Define the number of inner corners of the checkerboard (columns, rows)
# Note: this is not the number of squares, but the number of intersections!
CHECKERBOARD = (9, 6)

# Real size of a printed square's side (in millimeters)
SQUARE_SIZE = 22.0 

# Stopping criteria to refine corner detection at the sub-pixel level
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# Preparation of theoretical 3D points in space (Z=0, flat surface)
# Format: (0,0,0), (15,0,0), (30,0,0) ... 
objp = np.zeros((CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)
objp = objp * SQUARE_SIZE

# Arrays to store points from all images
objpoints = [] # 3D points in real world space
imgpoints = [] # 2D points (pixels) in image plane

# ==========================================
# 2. CORNER EXTRACTION FROM IMAGES
# ==========================================
# Load all images from the folder (modify the extension if needed: .jpg, .png)
images = glob.glob('images_calibration/*.png')
print(f"Found {len(images)} images for calibration.")

for fname in images:
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Find the checkerboard corners in the image
    ret, corners = cv2.findChessboardCorners(gray, CHECKERBOARD, None)

    # If corners are found, refine and store them
    if ret == True:
        objpoints.append(objp)
        
        # Sub-pixel refinement for higher precision
        corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        imgpoints.append(corners2)

        # (Optional) Uncomment these lines to see the detection live
        # cv2.drawChessboardCorners(img, CHECKERBOARD, corners2, ret)
        # cv2.imshow('Detection', img)
        # cv2.waitKey(200)

# cv2.destroyAllWindows()

# ==========================================
# 3. CALIBRATION COMPUTATION
# ==========================================
print("\nCalculating parameters...")
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

print("\n================ RESULTS ================")
print("\n1. Camera Matrix (mtx):")
print("Contains the focal length in pixels (fx, fy) and the optical center (cx, cy).")
print(mtx)

print("\n2. Distortion Coefficients (dist):")
print("Order: [k1, k2, p1, p2, k3]")
print(dist)
print("===========================================\n")

# ==========================================
# 4. UNDISTORTION TEST
# ==========================================
# Let's take the first image to "undistort" it with our new parameters
if len(images) > 0:
    img_test = cv2.imread(images[0])
    
    # Apply mathematical correction
    img_undistorted = cv2.undistort(img_test, mtx, dist, None, mtx)
    
    # Save the result
    cv2.imwrite('undistorted_result.png', img_undistorted)
    print("A corrected test image has been saved as 'undistorted_result.png'.")