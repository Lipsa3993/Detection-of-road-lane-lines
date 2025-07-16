import cv2
import numpy as np

def canny_edge_detector(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    return cv2.Canny(blur, 50, 150)

def region_of_interest(image):
    height = image.shape[0]
    polygon = np.array([[
        (100, height), 
        (image.shape[1] - 100, height), 
        (image.shape[1] // 2, int(height * 0.6))
    ]])
    mask = np.zeros_like(image)
    cv2.fillPoly(mask, polygon, 255)
    return cv2.bitwise_and(image, mask)

def display_lines(image, lines):
    line_image = np.zeros_like(image)
    if lines is not None:
        for x1, y1, x2, y2 in lines:
            cv2.line(line_image, (x1, y1), (x2, y2), (0, 255, 0), 10)
    return line_image

def average_slope_intercept(image, lines):
    left, right = [], []
    if lines is None:
        return None

    for line in lines:
        x1, y1, x2, y2 = line.reshape(4)
        if x2 - x1 == 0:
            continue
        slope = (y2 - y1) / (x2 - x1)
        intercept = y1 - slope * x1
        (left if slope < 0 else right).append((slope, intercept))
    
    result = []
    for side in [left, right]:
        if side:
            avg = np.average(side, axis=0)
            result.append(make_coordinates(image, avg))
    return np.array(result)

def make_coordinates(image, params):
    slope, intercept = params
    y1 = image.shape[0]
    y2 = int(y1 * 0.6)
    x1 = int((y1 - intercept) / slope)
    x2 = int((y2 - intercept) / slope)
    return [x1, y1, x2, y2]

def process_frame(frame):
    edges = canny_edge_detector(frame)
    roi = region_of_interest(edges)
    lines = cv2.HoughLinesP(roi, 2, np.pi/180, threshold=100, minLineLength=40, maxLineGap=5)
    averaged = average_slope_intercept(frame, lines)
    line_image = display_lines(frame, averaged)
    return cv2.addWeighted(frame, 0.8, line_image, 1, 1)
