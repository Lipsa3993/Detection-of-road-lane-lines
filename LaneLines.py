import cv2
import numpy as np
import matplotlib.image as mpimg


def hist(img):
    bottom_half = img[img.shape[0]//2:, :]
    return np.sum(bottom_half, axis=0)


class LaneLines:
    def __init__(self):
        self.left_fit = None
        self.right_fit = None
        self.binary = None
        self.nonzero = None
        self.nonzerox = None
        self.nonzeroy = None
        self.clear_visibility = True
        self.dir = []

        self.left_curve_img = mpimg.imread('left_turn.png')
        self.right_curve_img = mpimg.imread('right_turn.png')
        self.keep_straight_img = mpimg.imread('straight.png')

        self.left_curve_img = cv2.normalize(self.left_curve_img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        self.right_curve_img = cv2.normalize(self.right_curve_img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        self.keep_straight_img = cv2.normalize(self.keep_straight_img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

        self.nwindows = 9
        self.margin = 100
        self.minpix = 50

    def forward(self, img):
        self.extract_features(img)
        return self.fit_poly(img)

    def pixels_in_window(self, center, margin, height):
        topleft = (center[0] - margin, center[1] - height // 2)
        bottomright = (center[0] + margin, center[1] + height // 2)

        condx = (topleft[0] <= self.nonzerox) & (self.nonzerox <= bottomright[0])
        condy = (topleft[1] <= self.nonzeroy) & (self.nonzeroy <= bottomright[1])
        return self.nonzerox[condx & condy], self.nonzeroy[condx & condy]

    def extract_features(self, img):
        self.img = img
        self.window_height = int(img.shape[0] // self.nwindows)
        self.nonzero = img.nonzero()
        self.nonzerox = np.array(self.nonzero[1])
        self.nonzeroy = np.array(self.nonzero[0])

    def find_lane_pixels(self, img):
        out_img = np.dstack((img, img, img)) * 255
        histogram = hist(img)
        midpoint = histogram.shape[0] // 2
        leftx_base = np.argmax(histogram[:midpoint])
        rightx_base = np.argmax(histogram[midpoint:]) + midpoint

        leftx_current = leftx_base
        rightx_current = rightx_base
        y_current = img.shape[0] + self.window_height // 2

        leftx, lefty, rightx, righty = [], [], [], []

        for _ in range(self.nwindows):
            y_current -= self.window_height
            center_left = (leftx_current, y_current)
            center_right = (rightx_current, y_current)

            good_left_x, good_left_y = self.pixels_in_window(center_left, self.margin, self.window_height)
            good_right_x, good_right_y = self.pixels_in_window(center_right, self.margin, self.window_height)

            leftx.extend(good_left_x)
            lefty.extend(good_left_y)
            rightx.extend(good_right_x)
            righty.extend(good_right_y)

            if len(good_left_x) > self.minpix:
                leftx_current = int(np.mean(good_left_x))
            if len(good_right_x) > self.minpix:
                rightx_current = int(np.mean(good_right_x))

        return leftx, lefty, rightx, righty, out_img

    def fit_poly(self, img):
        leftx, lefty, rightx, righty, out_img = self.find_lane_pixels(img)

        if len(lefty) > 1500:
            self.left_fit = np.polyfit(lefty, leftx, 2)
        if len(righty) > 1500:
            self.right_fit = np.polyfit(righty, rightx, 2)

        if self.left_fit is None or self.right_fit is None:
            return out_img

        maxy = img.shape[0] - 1
        miny = img.shape[0] // 3
        if len(lefty):
            maxy = max(maxy, np.max(lefty))
            miny = min(miny, np.min(lefty))
        if len(righty):
            maxy = max(maxy, np.max(righty))
            miny = min(miny, np.min(righty))

        ploty = np.linspace(miny, maxy, img.shape[0])
        left_fitx = self.left_fit[0] * ploty ** 2 + self.left_fit[1] * ploty + self.left_fit[2]
        right_fitx = self.right_fit[0] * ploty ** 2 + self.right_fit[1] * ploty + self.right_fit[2]

        for i, y in enumerate(ploty):
            l = int(left_fitx[i])
            r = int(right_fitx[i])
            y = int(y)
            cv2.line(out_img, (l, y), (r, y), (0, 255, 0))

        return self.plot(out_img)

    def plot(self, out_img):
        if self.left_fit is None or self.right_fit is None:
            return out_img

        lR, rR, pos = self.measure_curvature()
        value = self.left_fit[0] if abs(self.left_fit[0]) > abs(self.right_fit[0]) else self.right_fit[0]

        if abs(value) <= 0.00015:
            self.dir.append('F')
        elif value < 0:
            self.dir.append('L')
        else:
            self.dir.append('R')

        if len(self.dir) > 10:
            self.dir.pop(0)

        W, H = 400, 500
        widget = np.copy(out_img[:H, :W])
        widget //= 2
        widget[0, :] = [0, 0, 255]
        widget[-1, :] = [0, 0, 255]
        widget[:, 0] = [0, 0, 255]
        widget[:, -1] = [0, 0, 255]
        out_img[:H, :W] = widget

        direction = max(set(self.dir), key=self.dir.count)
        msg = "Keep Straight Ahead"
        curvature_msg = f"Curvature = {min(lR, rR):.0f} m"

        img_map = {
            'L': (self.left_curve_img, "Left Curve Ahead"),
            'R': (self.right_curve_img, "Right Curve Ahead"),
            'F': (self.keep_straight_img, "Keep Straight Ahead")
        }

        img_icon, msg = img_map[direction]
        if img_icon.shape[2] == 4:
            y, x = img_icon[:, :, 3].nonzero()
        else:
            y, x = img_icon[:, :, 0].nonzero()
        out_img[y, x - 100 + W // 2] = img_icon[y, x, :3]

        cv2.putText(out_img, msg, (10, 240), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        if direction in 'LR':
            cv2.putText(out_img, curvature_msg, (10, 280), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        cv2.putText(out_img, "Good Lane Keeping", (10, 400), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 2)
        cv2.putText(out_img, f"Vehicle is {pos:.2f} m away from center", (10, 450), cv2.FONT_HERSHEY_SIMPLEX, 0.66, (255, 255, 255), 2)

        return out_img

    def measure_curvature(self):
        ym, xm = 30 / 720, 3.7 / 700
        left_fit = self.left_fit.copy()
        right_fit = self.right_fit.copy()
        y_eval = 700 * ym

        left_curveR = ((1 + (2 * left_fit[0] * y_eval + left_fit[1]) ** 2) ** 1.5) / np.abs(2 * left_fit[0])
        right_curveR = ((1 + (2 * right_fit[0] * y_eval + right_fit[1]) ** 2) ** 1.5) / np.abs(2 * right_fit[0])

        xl = np.polyval(left_fit, 700)
        xr = np.polyval(right_fit, 700)
        pos = (1280 // 2 - (xl + xr) / 2) * xm
        return left_curveR, right_curveR, pos
