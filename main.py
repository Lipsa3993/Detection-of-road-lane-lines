import os
import numpy as np
import cv2
from moviepy.editor import VideoFileClip

# Import custom modules
from CameraCalibration import CameraCalibration
from PerspectiveTransformation import PerspectiveTransformation
from Thresholding import Thresholding
from LaneLines import LaneLines

class FindLaneLines:
    def __init__(self):  #Fix: correct constructor
        # Provide required arguments to CameraCalibration
        image_dir = 'camera_cal'  # folder with calibration images
        nx = 9                    # chessboard corners along x
        ny = 6                    # chessboard corners along y
        self.calibration = CameraCalibration(image_dir, nx, ny)
        self.transform = PerspectiveTransformation()
        self.threshold = Thresholding()
        self.lane_lines = LaneLines()

    def forward(self, img):  #This was missing
        original = np.copy(img)

        # 1. Undistort the image
        undistorted = self.calibration.undistort(img)

        # 2. Perspective transform (warp)
        warped = self.transform.forward(undistorted)

        # 3. Thresholding
        binary = self.threshold.forward(warped)

        # 4. Lane detection
        lane_overlay = self.lane_lines.forward(binary)

        # 5. Warp lane overlay back to original perspective
        lane_unwarped = self.transform.backward(lane_overlay)

        # 6. Overlay lane on original image
        result = cv2.addWeighted(original, 1, lane_unwarped, 0.6, 0)

        # 7. Plot additional info like curvature, direction
        result = self.lane_lines.plot(result)

        return result

    def process_video(self, input_path, output_path):
        if not os.path.exists(input_path):
            print(f"Error: Input file does not exist -> {input_path}")
            return

        output_dir = os.path.dirname(output_path)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)

        try:
            clip = VideoFileClip(input_path)
            if clip.duration == 0 or clip.reader.nframes == 0:
                print(f"Error: The video is empty or corrupted -> {input_path}")
                return

            print("Processing video...")
            processed_clip = clip.fl_image(self.forward)
            processed_clip.write_videofile(output_path, audio=False)
            print(f"Processing complete! Saved to {output_path}")

        except Exception as e:
            print(f"Error during video processing: {e}")

if __name__ == "__main__":
    lane_finder = FindLaneLines()

    input_path = "input_video/challenge.mp4"
    output_path = "output_video/challenge_output.mp4"

    lane_finder.process_video(input_path, output_path)
