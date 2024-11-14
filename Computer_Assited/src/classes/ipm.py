import cv2
import numpy as np

class IPM:
    def __init__(self):
        self.transformation_matrix = None

    def get_birds_eye_view(self, image):
        height, width = image.shape[:2]

        # Define perspective transform points
        src_points = np.float32([
            [width * 0.4, height * 0.65],
            [width * 0.7, height * 0.65],
            [width, height],
            [0, height]
        ])

        dst_points = np.float32([
            [width * 0.2, height * 0.5],
            [width * 0.8, height * 0.5],
            [width * 0.8, height],
            [width * 0.2, height]
        ])

        self.transformation_matrix = cv2.getPerspectiveTransform(src_points, dst_points)
        birds_eye_view = cv2.warpPerspective(image, self.transformation_matrix, 
                                           (width, height), flags=cv2.INTER_LINEAR)
        
        return birds_eye_view, self.transformation_matrix

    def transform_point(self, point):
        """
        Transform a single point using the IPM transformation matrix
        point: tuple of (x, y)
        """
        if self.transformation_matrix is None:
            raise ValueError("Transformation matrix not computed. Run get_birds_eye_view first.")
            
        point_array = np.array([[[point[0], point[1]]]], dtype=np.float32)
        transformed_point = cv2.perspectiveTransform(point_array, self.transformation_matrix)
        return (int(transformed_point[0][0][0]), int(transformed_point[0][0][1]))