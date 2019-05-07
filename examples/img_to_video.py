import numpy as np
import cv2
import os

image_folder = 'single_png'
video_name = 'single_train.avi'

images = [img for img in os.listdir(image_folder) if img.endswith(".png")]
sort_imgs = np.sort(images)

frame = cv2.imread(os.path.join(image_folder, images[0]))
height, width, layers = frame.shape

video = cv2.VideoWriter(video_name, 0, 5, (width,height))

for image in sort_imgs:
    video.write(cv2.imread(os.path.join(image_folder, image)))

cv2.destroyAllWindows()
video.release()
