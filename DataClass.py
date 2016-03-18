import math
import numpy as np

class DataClass(object):
	"""Data Class of the TrackClass of the ObjectTracker program.
	This class stores either the data from the different objects that are tracked
	(positions, histograms, etc) or from the target areas (coordinates, area
	entries, etc).
	2015 Stephan Meyer fuschro@gmail.com
   	"""		
	def __init__(self):
		self.distances = []
		self.distances_x = []
		self.histogram_line = None
		self.past_positions = []
		self.pts = []
		self.roiPts = []
		self.roiHist = []
		self.target_areas =[]
		self.target_entries_entered = []
		self.target_entries_x = []
		self.target_outline = []
	
	def get_track_only(self):
		track = np.array(self.past_positions, np.int32)
		return track
		
	def get_target_only(self):	
		target =  np.array(self.target_outline, np.int32)
		return target
			
	def get_normalized_speed_only(self):
		max_speed = max(self.distances)
		min_speed = min(self.distances)
		mean_speed = reduce(lambda x, y: x + y, self.distances) / float(len(self.distances))
		self.distances[:] = [x / max_speed for x in self.distances]
		self.distances[:] = [x * 255 for x in self.distances]
		return self.distances
		
	def applyLUT(self,img_in):
		red = lambda x: x*2 if x < 128 else 255 
		green = lambda x: x
		blue = lambda x: x*2-128 if x > 128 else 0
		dim = img_in.shape
		for x_pixel in range (0, dim[0]):
			 for y_pixel in range (0, dim[1]):
				 current_red =  img_in[x_pixel, y_pixel][0]
				 current_green =  img_in[x_pixel, y_pixel][1]
				 current_blue =  img_in[x_pixel, y_pixel][2]
				 img_in[x_pixel, y_pixel] = (red(current_red) ,green(current_green) , blue(current_blue) )
		return img_in
	
	def normalize_image(self,img_in, max_range):
		max_val = max(1,img_in.max())
		img_in = np.array(img_in, dtype=float)/max_val
		img_in = np.array(img_in, dtype=float)*max_range
		img_in = np.array(img_in, dtype=int)*1
		return img_in
		
	def pixel_distance(self,start, end):
		"""make sure all are above 0"""
		end_x = end[0] if end[0] > -1 else -end[0]
		start_x = start[0] if start[0] > -1 else -start[0]
		end_y = end[1] if end[1] > -1 else -end[1]
		start_y = start[1] if start[1] > -1 else -start[1]
		x_step = end_x - start_x
		y_step = end_y - start_y
		distance_sqr = math.pow(x_step, 2) + math.pow(y_step, 2)
		distance = math.sqrt(distance_sqr)
		return distance		