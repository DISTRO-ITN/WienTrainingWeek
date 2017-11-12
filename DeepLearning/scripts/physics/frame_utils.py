import sys
import json
import os
import subprocess
import numpy as np
from PIL import Image

leading_zeros = 4
extension = '.jpg'
dim_x = 256
dim_y = 256
input_frames = 10
output_frames = 1
total_frames = 251

def generate_video(frames_path, frames_output_path):
	command = ['ffmpeg', '-y',
			   '-f', 'image2',
			   '-framerate', '20',
			   '-i', frames_path + '%04d.jpg',
			   frames_output_path]
	ffmpeg = subprocess.Popen(command, stderr=subprocess.PIPE, stdout=subprocess.PIPE)
	video, err = ffmpeg.communicate()
	if (err):
		print(err)
	return video

def get_output_frame_name(index):
	return str(index).zfill(leading_zeros) + extension

def save_image(npdata, filename, display):
	if display:
		print("Save path: " + filename)
	image = Image.fromarray(np.uint8(npdata * 255))
	result = image.save(filename)

def save_frames(file_path, y_data, display=False):
	for frame_index in range(total_frames):
		if not os.path.exists(file_path):
			os.makedirs(file_path)
		save_image(y_data[:, :, frame_index], file_path + '/' + get_output_frame_name(frame_index), display)

def get_frame_name(index):
	return "/render/" + str(index).zfill(leading_zeros) + extension

def load_image(filename, display):
	if display:
		print("Load path: " + filename)
	image = Image.open(filename).convert('L')
	np_image = np.asarray(image, dtype=np.float32)
	return np_image / 255.0

def load_frames(video_path, starting_frame_index, display=False):
	x_data = np.full([dim_x, dim_y, input_frames], np.nan)
	y_data = np.full([dim_x, dim_y, output_frames], np.nan)
	# Fill the placeholders with the requested video frames.
	for frame_index in range(input_frames):
		filename = video_path + get_frame_name(frame_index + starting_frame_index)
		if not os.path.isfile(filename):
			assert frame_index != 0, "Training data elements don't exist: {}.".format(filename)
			# Freeze the video on the last frame to match the requested video length.
			filename = video_path + get_frame_name(frame_index - 1 + starting_frame_index)
		x_data[:, :, frame_index] = load_image(filename, display)
	assert not np.any(np.isnan(x_data)), "Array x_data was incorrectly filled."
	y_data = load_image(video_path + get_frame_name(input_frames + starting_frame_index), display)
	return (x_data, y_data)
