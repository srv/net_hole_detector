#!/usr/bin/env python3
import cv2
import numpy as np

class ScaleEstimator:
    def __init__(self):
        pass

    def get_scale_from_blobs(self, cv_image, real_area_m2=0.000225, debug=False) -> float:
        """
        Calculates Meters/Pixel scale based in SQUARE ROOT of AREA of blobs
        This is more robust than using the width. With a black net and blue background

        Parameters
        ----------
            real_area_m2: float
                Real area of the Net in square meters
                If square size = 1.5cm -> 0.015 * 0.015 = 0.000225 m2
            debug: bool
                If it is True, stores process images

        Returns
        --------
            float
                meters/pixel ratio
        """
        # 1. Split channels
        # If the background is blue, the Blue channel will have the maximum contrast
        # between the net (dark) and the background (light).
        if len(cv_image.shape) == 3:
            blue = cv_image[:, :, 0]
        else:
            blue = cv_image

        # 2. Thresholding
        # We want the image to be either black or white, to do so 
        # everything 'blackish' will be black and everything 'whiteish' will be white.
        #_, thresh = cv2.threshold(blue, 50, 255, cv2.THRESH_BINARY)
        thresh = cv2.adaptiveThreshold(blue, 255, 
                                     cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                     cv2.THRESH_BINARY, 41, 5)

        # Optional: Noise removal
        kernel = np.ones((5,5), np.uint8)
        opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)

        # DEBUG: Store images
        if debug:
            cv2.imwrite("debug_1_blue.jpg", blue)
            cv2.imwrite("debug_2_thresh.jpg", thresh)
            cv2.imwrite("debug_3_opening.jpg", opening)
            
        # 3. Find contours
        contours, _ = cv2.findContours(opening, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        sqrt_areas = []
        valid_contours = [] # Just to print debug

        # Find image dimensions, to reject huge object
        img_h, img_w = blue.shape

        for cnt in contours:
            # First we calculate the area
            area_px = cv2.contourArea(cnt)
            
            # --- FILTERS ---
            
            # 3.1. Size Filter
            # Reject small noise (< 10px width)
            # Reject huge breakages (> 15% image width)
            if area_px < 50 or area_px > (img_w * img_h * 0.05):
                continue

            # 3.2. Aspect Ratio Filter (Form):
            x, y, w, h = cv2.boundingRect(cnt)
            aspect_ratio = float(w) / h
            if aspect_ratio < 0.6 or aspect_ratio > 1.6:
                continue
            
            # IF IS VALID:
            # We use the square root of the area as our lineal metric "L"
            # L_px = sqrt(Area_px)
            l_px = np.sqrt(area_px)
            sqrt_areas.append(l_px)

            if debug:
                valid_contours.append(cnt)

        if not sqrt_areas:
            return None
        
        # 4. Using MEDIAN to avoid Damaged Holes (that passed the filter) to affect the calculations
        # we call this the "Characteristic Longitude"
        median_l_px = np.median(sqrt_areas)

        # Calculate the real "Characteristic Longitude" (size of the equivalente square)
        real_l_meters = np.sqrt(real_area_m2)

        if debug:
            # Draw over original image detected contoursin RED
            debug_img = cv_image.copy()
            cv2.drawContours(debug_img, valid_contours, -1, (0, 0, 255), 3)
            cv2.imwrite("debug_4_result.jpg", debug_img)

            # B) New Image, white background, black contours
            clean_img = np.ones_like(cv_image) * 255
            
            # Valid hexagons in black
            # Grosor -1 para rellenarlos y ver las "manchas", o 2 para ver solo el borde
            # Probamos con -1 (relleno) para que parezca un mapa de dálmata, se ve muy claro.
            cv2.drawContours(clean_img, valid_contours, -1, (0, 0, 0), -1)
            
            cv2.imwrite("debug_5_clean.jpg", clean_img)
            print(f"[DEBUG] Valid hexagons: {len(sqrt_areas)}. Median: {median_l_px:.2f} px")
        
        # Unified return (Meters / Pixel)
        # scale = L_real / L_pixel
        return real_l_meters / median_l_px


    def get_scale_from_laser_lines(self, cv_image, real_dist_meters) -> float:
        """
        Detects two parallel laser lines and calculates the scale.
        Assumes the lasers are much brighter than the rest.

        Parameters
        ----------
            cv_image
                Image to get the scale from
            real_dist_meters
                Real distance between the lines

        Returns
        ---------
            float
                meters/pixel ratio between the lasers
        """
        # 1. Color Mask (Assuming Red Laser, adjust if it is Green)
        # --------------------------------------------------------------
        # ------------------ ADJUST TO LASER COLOR ---------------------
        # --------------------------------------------------------------
        hsv = cv2.cvtColor(cv_image, cv2.COLOR_BGR2HSV)

        # Red Ranges
        lower_red1 = np.array([0, 100, 100])
        upper_red1 = np.array([10, 255, 255])
        lower_red2 = np.array([170, 100, 100])
        upper_red2 = np.array([180, 255, 255])

        mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
        mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
        mask = mask1 + mask2

        # Clean the mask
        kernel = np.ones((3,3), np.uint8)
        mask = cv2.dilate(mask, kernel, iterations=1)
        # --------------------------------------------------------------
        # --------------------------------------------------------------

        # 2. Detect Lines with Hough
        # rho=1 pixel, theta=1 degree, threshold=minimum intersections
        lines = cv2.HoughLines(mask, 1, np.pi / 180, 100)
        
        if lines is None:
            return None
        
        # lines comes with the form [[rho, theta], [rho, theta], ...]
        # we want to group them by angle. to find the parallel ones

        valid_rhos = []
        target_theta = None

        # Step A: Find dominant angle
        # (We Assume lasers are the strongest/longest lines)
        thetas = lines[:, 0, 1]

        # Angle histogram to find the dominant one, convert to degrees
        degrees = np.rad2deg(thetas)
        hist, bin_edges = np.histogram(degrees, bins=range(0, 180, 5))
        peak_bin = np.argmax(hist)
        dominant_angle_min = bin_edges[peak_bin]
        dominant_angle_max = bin_edges[peak_bin+1]

        # Step B: Stay only with the lines with this angle
        for line in lines:
            rho, theta = line[0]
            angle_deg = np.degrees(theta)
            
            if dominant_angle_min <= angle_deg <= dominant_angle_max:
                valid_rhos.append(rho)
        
        if len(valid_rhos) < 2:
            return None
        
        # STEP C: Search for the two distance groups (Rho)
        # We use simple K-Means or order and find the gap
        valid_rhos.sort()
        
        # Search for the biggest gap between the detected lines
        # This split left and right lasers
        max_gap = 0
        split_idx = 0

        for i in range(len(valid_rhos) - 1):
            gap = valid_rhos[i+1] - valid_rhos[i]
            if gap > max_gap:
                max_gap = gap
                split_idx = i

        # If the gap is too small (10px), we only detected ONE laser (with double line)
        if max_gap < 20:
            return None
        
        # Average rho from group 1 and 2
        rho_group1 = np.mean(valid_rhos[:split_idx+1])
        rho_group2 = np.mean(valid_rhos[split_idx+1:])
        
        dist_pixels = abs(rho_group1 - rho_group2)
        
        return real_dist_meters / dist_pixels

