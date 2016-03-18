import cv2
import copy
from datetime import datetime
import logging
from matplotlib.path import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
import time
from Tkinter import Tk
from tkFileDialog import askopenfilename
import colorsys
#from pykalman import KalmanFilter

import DataClass
from DataClass import DataClass

class TrackClass(object):
	"""TrackClass of the ObjectTracker program.
	This class either opens a camera stream or loads a movie file. The user 
	can then outline objects within the movies which get tracked using CV2
	histogram comparison. Target areas can also be outlined and it will be
	recorded when the center of a tracked object enters the target area.
	Multiple object and target areas can be tracked simultanously.
	Once stopped a report of the tracks, target entries, speeds and locations
	will be generated and saves in image and txt coordinate form.
	2015 Stephan Meyer fuschro@gmail.com
   	"""
    
	
	def __init__(self):
		self.camera = None
		self.cancel_flag = False
		self.track_objects = []
		self.target_objects = []
		self.filename = None
		self.fps = None
		self.frame = None
		self.height = None
		self.inputMode = False
		self.save_name = 'Stream'
		self.temp_object = []
		self.temp_area = []
		self.width = None 
			
	def select_target(self, event, x, y, flags, param):
		if event is  1: 
			self.temp_area.append((x, y))
			cv2.circle(self.frame, (x, y), 4,
				(10*2, 50*2, 255), 2)
			cv2.imshow("frame", self.frame)
		else: 
			self.cancel_flag = True
			
	def select_object(self, event, x, y, flags, param):
		if event is  1:
			self.temp_object.append((x, y))
			cv2.circle(self.frame, (x, y), 4,
				(0*2, 10*2, 0), 2)
			cv2.imshow("frame", self.frame)
		else: 
			self.cancel_flag = True
			
	def initialize_stream(self, logger):
		self.camera = cv2.VideoCapture(0)
		if not (self.camera.read()[0]):
			logger.info('Loading Movie')
			print('no camera')
			Tk().withdraw() 
			self.filename = askopenfilename() 
			self.save_name = os.path.basename(self.filename)
			if not self.save_name:
				print 'user load path invalid'
				logger.info('user load path invalid')
				sys.exit()
			print(self.save_name)
			self.camera = cv2.VideoCapture(self.filename)
			logger.info('Loading file:'+str(self.filename))
		else:
			logger.info('Reading from Camera')
		self.fps = self.camera.get(cv2.cv.CV_CAP_PROP_FPS)
		if self.fps == 0.0:
			 self.fps = 24
		cv2.namedWindow("frame")
			
	def show_instructions(self, logger):
		logger.info('Show Instructions')
		single_frame = self.camera.read()[1]
		self.width = np.size(single_frame, 1) 
		self.height = np.size(single_frame, 0)
		img_instruction_start = np.zeros((self.height,self.width,3), np.uint8)
		cv2.putText(img_instruction_start,"INSTRUCTIONS", (10,17),
			cv2.FONT_HERSHEY_SIMPLEX, 0.5,(255,255,255),1)
		cv2.putText(img_instruction_start,"-Press space key to pause movie and see instructions", (10,40),
			cv2.FONT_HERSHEY_SIMPLEX, 0.45,(255,255,255),1)
		cv2.putText(img_instruction_start,"-Press 'o' to OUTLINE OBJECT to track", (10,60),
			cv2.FONT_HERSHEY_SIMPLEX, 0.45,(255,255,255),1)
		cv2.putText(img_instruction_start,"-Press 't' to OUTLINE TARGET AREAS to monitor", (10,80),
			cv2.FONT_HERSHEY_SIMPLEX, 0.45,(255,255,255),1)
		cv2.putText(img_instruction_start,"-Press 'q' to quit at any time", (10,100),
			cv2.FONT_HERSHEY_SIMPLEX, 0.45,(255,255,255),1)
		cv2.putText(img_instruction_start,"-Data will be saved in this directory", (10,120),
			cv2.FONT_HERSHEY_SIMPLEX, 0.45,(255,255,255),1)
		cv2.putText(img_instruction_start,"-The more targets/objects you add, the slower it runs", (10,140),
			cv2.FONT_HERSHEY_SIMPLEX, 0.45,(255,255,255),1)
		cv2.putText(img_instruction_start,"OPTIMIZE PERFORMANCE", (10,170),
			cv2.FONT_HERSHEY_SIMPLEX, 0.41,(255,255,0),1)
		cv2.putText(img_instruction_start,"If performance is slow with many objects or target areas being updated every frame,", (10,190),
			cv2.FONT_HERSHEY_SIMPLEX, 0.4,(255,255,0),1)
		cv2.putText(img_instruction_start,"decrease the updates per frame. Changing update rates to 5 objects per frame with a ", (10,210),
			cv2.FONT_HERSHEY_SIMPLEX, 0.4,(255,255,0),1)
		cv2.putText(img_instruction_start,"20fps video means 100 objects can be updated once per second with good speed. If your", (10,230),
			cv2.FONT_HERSHEY_SIMPLEX, 0.4,(255,255,0),1)
		cv2.putText(img_instruction_start,"object moves fast, tracking then won't be precise as each objects tracks once per second.", (10,250),
			cv2.FONT_HERSHEY_SIMPLEX, 0.4,(255,255,100),1)
		cv2.putText(img_instruction_start,"For MANY OBJECTS, limit object updates/frame by now typing a NUMBER FROM 1-9 (1 runs fastest).", (10,270),
			cv2.FONT_HERSHEY_SIMPLEX, 0.4,(0,50,255),1)
		cv2.putText(img_instruction_start,"For MANY AREAS, limit area updates/frame by now typing a LETTER FROM a-i (a runs fastest).", (10,290),
			cv2.FONT_HERSHEY_SIMPLEX, 0.4,(0,50,255),1)
		cv2.putText(img_instruction_start,"Tracking of all objects and all areas with each frame is the default.", (10,310),
			cv2.FONT_HERSHEY_SIMPLEX, 0.4,(100,255,170),1)
		cv2.imshow("frame",img_instruction_start)
	

	def detect(path):
		
		cascade = cv2.CascadeClassifier("/haar_cascade_Mice_randombg.xml")
		
		
		rects = cascade.detectMultiScale(img, 1.3, 4, cv2.cv.CV_HAAR_SCALE_IMAGE, (20,20))

		if len(rects) == 0:
			return [], img
		rects[:, 2:] += rects[:, :2]
		return rects, img



	def check_object_position(self, termination, frame_count, object_updates_per_frame, area_updates_per_frame, current_object_update_offset, logger):
		logger.debug("Offset in:"+str(current_object_update_offset))
		objects_in_areas = []
		current_object_tracked = 0
		current_target_checked = 0
		offset_for_object = 0
		offset_for_target = 0
		use_haar = True

		cascade = cv2.CascadeClassifier("haar_cascade_Mice_randombg.xml")
		"""if all objects are to be updates, set loop to all objects"""
		if object_updates_per_frame is None and area_updates_per_frame is not None:
			object_updates_per_frame = len(self.track_objects)
			offset_for_target = current_object_update_offset
		"""if all targets are to be check, set loop to all targets"""
		if area_updates_per_frame is None and object_updates_per_frame is not None:
			area_updates_per_frame = len(self.target_objects)
			offset_for_object = current_object_update_offset
		if area_updates_per_frame is None and object_updates_per_frame is None:
			object_updates_per_frame = len(self.track_objects)
			area_updates_per_frame = len(self.target_objects)
		"""loop as many times as object/target areas to be updates per frame, but not more than possible"""
		if object_updates_per_frame > len(self.track_objects)-1:
			object_updates_per_frame = len(self.track_objects)
		if area_updates_per_frame > len(self.target_objects)-1:
			area_updates_per_frame = len(self.target_objects)
		logger.debug("TRACK LOOP START, ob upates:"+str(object_updates_per_frame))
		for current_object_index in range (0, object_updates_per_frame):
			"""add index offset"""
			current_object_tracked = current_object_index + offset_for_object
			"""if past end, start at beginning"""
			if current_object_tracked > len(self.track_objects)-1:
				current_object_tracked -= len(self.track_objects)
			logger.debug("current_object_tracked:"+str(current_object_tracked))
			#"""first convert color space of current frame to HSV"""
			#hsv = cv2.cvtColor(self.frame, cv2.COLOR_BGR2HSV)



			if use_haar == True:
				#rects = cascade.detectMultiScale(self.frame, 1.3, 4, cv2.cv.CV_HAAR_SCALE_IMAGE, (20,20))
				#rects = cascade.detectMultiScale(self.frame)
				rects = cascade.detectMultiScale(self.frame, scaleFactor=1.1, minNeighbors=4,
                                             minSize=(90, 90), flags=cv2.cv.CV_HAAR_SCALE_IMAGE)
				#print 'objects found: %s' % len(rects)
				#cv2.rectangle(image_mouse,(x,y),(x+w,y+h),(0,255,0),2)
				if len(rects) > 0:
					x1_coord = [rects[0][0], rects[0][1]]
					y1_coord = [rects[0][0]+rects[0][2], rects[0][1]+rects[0][3]]
					x2_coord = [rects[0][0]+rects[0][2], rects[0][1]]
					y2_coord = [rects[0][0], rects[0][1]+rects[0][3]]
					location =  np.asarray([x1_coord, x2_coord,y1_coord, y2_coord])
					#print location
					self.track_objects[current_object_tracked].pts = location
					temp_pts = self.track_objects[current_object_tracked].pts
				else:
					try:
						"""if error, use the coordinates from before"""
						cv2.putText(self.frame,"LOST OBJECT TRACK..", (10,50),
							cv2.FONT_HERSHEY_SIMPLEX, 0.55,(40,70,50),2)
						self.track_objects[current_object_tracked].pts = temp_pts
					except:
						pass
						#cv2.putText(self.frame,"SEARCHING FOR OBJECT..", (10,50),
						#	cv2.FONT_HERSHEY_SIMPLEX, 0.55,(40,70,50),2)

			else:
				"""now compare the current frame histogram to the one stored from the object"""
				backProj = cv2.calcBackProject([self.frame.astype('float32')], [0], self.track_objects[current_object_tracked].roiHist, [0, 180], 1)
				"""now shift the histogram_line to where the histogram has the best match"""
				(r, self.track_objects[current_object_tracked].histogram_line) = cv2.CamShift(backProj, self.track_objects[current_object_tracked].histogram_line, termination)
			
				"""use error handling to indicate lost of track"""
				temp_pts = self.track_objects[current_object_tracked].pts
				try:
					"""get new track"""
					self.track_objects[current_object_tracked].pts = np.int0(cv2.cv.BoxPoints(r))
				except: 
					"""if error, use the coordinates from before"""
					cv2.putText(self.frame,"LOST OBJECT TRACK..", (10,50),
						cv2.FONT_HERSHEY_SIMPLEX, 0.55,(40,70,50),2)
					self.track_objects[current_object_tracked].pts = temp_pts
			if self.track_objects[current_object_tracked].pts is not None:
				normalized_color = int(float(current_object_tracked)/len(self.track_objects)*128)
				"""draw outline"""
				try:
					cv2.polylines(self.frame, [self.track_objects[current_object_tracked].pts], 
					True, (0, 70, 40), 1)
				except:
					print 'failed object outline display'
					logger.debug("ERROR: failed object outline display: "+str(self.track_objects[current_object_tracked].pts))
				current_poly = self.track_objects[current_object_tracked].pts.reshape((-1,1,2))
				text_coords = tuple(map(tuple,current_poly[0]))
				cv2.putText(self.frame,"ob "+str(current_object_index+1), text_coords[0], 
					cv2.FONT_HERSHEY_SIMPLEX, 0.5,(0,70,100),1)
				object_center = np.average(self.track_objects[current_object_tracked].pts, axis=0)
				"""draw center"""
				cv2.circle(self.frame, (int(object_center[0]), int(object_center[1])),
					4, (0,255,0), 2)
			      	self.track_objects[current_object_tracked].past_positions.append((object_center[0], object_center[1]));
				self.track_objects[current_object_tracked].pts = np.array(self.track_objects[current_object_tracked].past_positions, np.int32)
				cv2.polylines(self.frame,[self.track_objects[current_object_tracked].pts],
					False,TrackClass.pick_color(50), 2)
				

				"""get the length of displacement from previous point"""
				if len(self.track_objects[current_object_tracked].past_positions) > 1:
					dinstance_temp = self.track_objects[current_object_tracked].pixel_distance(
						self.track_objects[current_object_tracked].past_positions[len(self.track_objects[current_object_tracked].past_positions)-2], 
						self.track_objects[current_object_tracked].past_positions[len(self.track_objects[current_object_tracked].past_positions)-1] 
						)
					self.track_objects[current_object_tracked].distances.append(dinstance_temp)
					self.track_objects[current_object_tracked].distances_x.append(frame_count)
					"""check if another target had been added then extent array storing entries"""
					if len(self.track_objects[current_object_tracked].target_entries_entered)<len(self.target_objects):
						self.track_objects[current_object_tracked].target_entries_entered.append([])
						self.track_objects[current_object_tracked].target_entries_x.append([])
					"""now with the current object position, check all requested targets areas if entered"""
					for current_target_index in range (0, area_updates_per_frame):
						"""add index offset"""
						current_target_checked = current_target_index + offset_for_target
						"""if past end, start at beginning"""
						if current_target_checked > len(self.target_objects)-1:
							current_target_checked -= len(self.target_objects)
						logger.debug("current_target_checked:"+str(current_target_checked))
						polygon_path = Path(self.target_objects[current_target_checked].target_outline[0])
						is_within = polygon_path.contains_point(
							[object_center[0], object_center[1]]
							)
						if is_within:
							objects_in_areas = 'Object in area'
							"""check if target_entries exists, if added after target added, append"""
							if current_target_checked > len(self.track_objects[current_object_tracked].target_entries_entered)-1:
							#if not self.track_objects[current_object_tracked].target_entries_entered[current_target_checked]:
								self.track_objects[current_object_tracked].target_entries_entered.append([])
								self.track_objects[current_object_tracked].target_entries_x.append([])
							"""for each track object, store each target entry info separately in list of list"""
							self.track_objects[current_object_tracked].target_entries_entered[current_target_checked].append(1)
							self.track_objects[current_object_tracked].target_entries_x[current_target_checked].append(frame_count)                                        
						else:
							if current_target_checked > len(self.track_objects[current_object_tracked].target_entries_entered)-1:
							#if not self.track_objects[current_object_tracked].target_entries_entered[current_target_checked]:
								self.track_objects[current_object_tracked].target_entries_entered.append([])
								self.track_objects[current_object_tracked].target_entries_x.append([])
							self.track_objects[current_object_tracked].target_entries_entered[current_target_checked].append(0)
							self.track_objects[current_object_tracked].target_entries_x[current_target_checked].append(frame_count) 
					if objects_in_areas:
						cv2.putText(self.frame, objects_in_areas, (10,20),
									cv2.FONT_HERSHEY_SIMPLEX, 0.55,(40, 10,50),2)

		#cv2.imwrite('Screenshot%d.jpg' %frame_count,self.frame)
		"""update the offset"""
		if object_updates_per_frame is len(self.track_objects):
			logger.debug("ret:"+str(current_target_checked))
			return current_target_checked+1
		else:	
			return current_object_tracked+1
								
	def draw_all_targets(self, logger):
		for target_index, current_target in enumerate(self.target_objects):
			logger.debug("Drawing target:"+str(target_index))
			current_poly = np.array(current_target.target_outline, np.int32)
			current_poly = current_poly.reshape((-1,1,2))
			cv2.polylines(self.frame,[current_poly],
				True,(10,40,int(200-((target_index+1)*20))),4)
			text_coords = tuple(map(tuple,current_poly[0]))
			logger.debug("at position:"+str(text_coords))
			cv2.putText(self.frame,"target "+str(target_index+1), text_coords[0], 
				cv2.FONT_HERSHEY_SIMPLEX, 0.5,(40,70,int(200-((target_index+1)*20))),1)
				
	def get_object_outline(self, key):
		cv2.rectangle(self.frame, (0, 0),(500,50),(100,0,100),-1)
		cv2.putText(self.frame,"Select points around the OBJECT to track", (5,14),
			cv2.FONT_HERSHEY_SIMPLEX, 0.5,(255,255,255),1)
		cv2.putText(self.frame,"When done, press o to convert to object or any to continue", (5,30),
			cv2.FONT_HERSHEY_SIMPLEX, 0.5,(255,255,255),1)
		cv2.putText(self.frame,"When less than 3 pts, any key to cancel (q to quit)", (5,44),
			cv2.FONT_HERSHEY_SIMPLEX, 0.5,(255,255,255),1)
		cv2.imshow("frame", self.frame)
		"""change the function bound to mouse to Target function"""
		cv2.setMouseCallback("frame", self.select_object)
		user_input = cv2.waitKey(0);
		inputMode = True
		orig = self.frame.copy()
		while len(self.temp_object) < 3:
			cv2.imshow("frame", self.frame)
			if self.cancel_flag is  True:
				self.temp_object = [];
				break
			key = cv2.waitKey(0)
		if len(self.temp_object) > 2:
			"""add new object object"""
			object_data_temp = DataClass()
			outline = self.temp_object
			self.temp_object = np.array(self.temp_object)
			s = self.temp_object.sum(axis = 1)
			tl = self.temp_object[np.argmin(s)]
			br = self.temp_object[np.argmax(s)]
			roi = orig[tl[1]:br[1], tl[0]:br[0]]
			roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
			object_data_temp.roiHist = cv2.calcHist([roi.astype('float32')], [0], None, [180], [0, 180])
			object_data_temp.roiHist = cv2.normalize(object_data_temp.roiHist, object_data_temp.roiHist, 0, 255, cv2.NORM_MINMAX)
			object_data_temp.histogram_line = (tl[0], tl[1], br[0], br[1])
			self.track_objects.append(object_data_temp)
			self.temp_object = []
			cv2.putText(self.frame,"ob "+str(len(self.track_objects)), (tl[0],tl[1]), 
				cv2.FONT_HERSHEY_SIMPLEX, 0.5,(0,70,100),1)
			current_poly = np.array(outline, np.int32)
			current_poly = current_poly.reshape((-1,1,2))
			cv2.polylines(self.frame,[current_poly],
				True,(0,70,100))
		self.cancel_flag is  False
		return user_input
				
	def get_target_outline(self, key):
		cv2.rectangle(self.frame, (0, 0),(500,50),(0,0,0),-1)
		cv2.putText(self.frame,"Select points around the TARGET area.", (5,14),
			cv2.FONT_HERSHEY_SIMPLEX, 0.5,(255,255,255),1)
		cv2.putText(self.frame,"When done, press t to convert to target or any to continue.", (5,30),
			cv2.FONT_HERSHEY_SIMPLEX, 0.5,(255,255,255),1)
		cv2.putText(self.frame,"When less than 3 pts, any key to cancel (q to quit).", (5,44),
			cv2.FONT_HERSHEY_SIMPLEX, 0.5,(255,255,255),1)
		cv2.imshow("frame", self.frame)
		"""change the function bound to mouse to Target function"""
		cv2.setMouseCallback("frame", self.select_target)
		user_input = cv2.waitKey(0);
		inputMode = True
		orig = self.frame.copy()
		while len(self.temp_area) < 3:
			cv2.imshow("frame", self.frame)
			if self.cancel_flag is  True:
				self.temp_area = [];
				break
			key = cv2.waitKey(0)
		if len(self.temp_area) > 2:
			"""add new target object"""
			target_data_temp = DataClass()
			target_data_temp.target_outline.append(self.temp_area)
			target_data_temp.target_entries_entered.append([])
			target_data_temp.target_entries_x.append([])			
			target_data_temp.temp_area = [];
			self.target_objects.append(target_data_temp)
			self.temp_area = []
		self.cancel_flag is  False
		return user_input
				
	def write_data_handler(self, frame_count, copied_image, logger):
		self.show_image_handler(copied_image, logger)
		self.plot_graphs_handler(frame_count, logger)
	
	def show_image_handler(self, copied_image, logger):
		#cv2.imwrite('tr_'+self.save_name+'_Screenshot.png',copied_image)
		logger.debug("writing track image")
		legend_color_dict = self.show_tracks_image()
		logger.debug("writing speed image")
		self.show_speed_track_image()
		logger.debug("writing heat image")
		self.show_heat_image()	
		logger.debug("writing legend image")
		self.show_legend_image(legend_color_dict)
	
	def plot_graphs_handler(self, frame_count, logger):
		logger.debug("writing area entry graphs")
		self.plot_area_entries_handler(frame_count)
		logger.debug("writing area speed graphs")
		self.plot_speed()
		logger.debug("writing area distance graphs")
		self.plot_distance_only()
			
	def plot_distance_only(self):		
		plt.ion()
		fig = plt.figure(2)
		fig.set_size_inches(10, 5)
		ax = fig.gca()
		ax.set_autoscale_on(False)
		"""plot all distances seperately"""
		for object_counter, track_object in enumerate(self.track_objects):
			total_distance = sum(track_object.distances)
			current_color_mod = int(float(object_counter)/len(self.track_objects)*128)
			normalized_color_256 = TrackClass.pick_color(current_color_mod)
			normalized_color_1 = [x/256 for x in normalized_color_256]
			normalized_color_1.reverse()
			plt.plot(track_object.distances_x, track_object.distances, linewidth=1.0, color=normalized_color_1, alpha = 0.7)
			np.savetxt('tra_'+self.save_name+'_DistanceObject'+str(object_counter+1)+'perFrame_plot.txt', np.column_stack([track_object.distances_x, track_object.distances]),fmt='%10.2f')
			title_current = 'Object '+str(object_counter+1)+' distance covered each frame (pixels). Total: '+str(total_distance)
			plt.title(title_current,  fontsize=10)
			plt.axis([0,  max(track_object.distances_x)+1, -0.1,max(track_object.distances)], fontsize = 2)
			plt.yticks(np.arange(0, max(track_object.distances)+1, max(track_object.distances)/2))
			plt.tight_layout()
			plt.tick_params(axis='both', which='minor', labelsize=5)
			plt.tick_params(axis='both', which='minor', labelsize=4)
			plt.xlabel('Frame')
			plt.ylabel('Pixels/frame')
			plt.savefig('tr_'+self.save_name+'_Object'+str(object_counter+1)+'_DistancePlot.png',dpi=300)	
			plt.clf()
		
	def plot_area_entries_handler(self, frame_count):	
		"""make one graph for each target including all objects"""
		for target_counter, target_object in enumerate(self.target_objects):
			plt.ion()
			fig = plt.figure(2)
			fig.set_size_inches(10, 5)
			ax = fig.gca()
			ax.set_autoscale_on(False)
			"""plot one with thick and one with thin lines"""
			self.plot_area_entries(target_counter, target_object, 211, frame_count,use_thick_lines=True)
			self.plot_area_entries(target_counter, target_object, 212, frame_count,use_thick_lines=False)
			plt.savefig('tr_'+self.save_name+'_Target'+str(target_counter+1)+'_Entries.png',dpi=300)	
			plt.clf()
			
	def plot_area_entries(self, target_counter, target_object, sub_plot_position, frames, use_thick_lines):	
		summary_string = []
		"""now for each target plot all object entries with current target as index"""
		for object_counter, track_object in enumerate(self.track_objects):
			counts_in_target = track_object.target_entries_entered[target_counter].count(1)
			area_percent = float(counts_in_target)/max(len(track_object.target_entries_entered[target_counter]),1)*100
			entries = [x for i, x in enumerate(track_object.target_entries_entered[target_counter][:len(track_object.target_entries_entered[target_counter])-1]) if x <  track_object.target_entries_entered[target_counter][i+1]]
			plt.subplot(sub_plot_position)
			current_color_mod = int(float(object_counter)/len(self.track_objects)*128)
			normalized_color_256 = TrackClass.pick_color(current_color_mod)
			normalized_color_1 = [x/256 for x in normalized_color_256]
			normalized_color_1.reverse()
			if use_thick_lines:
				current_line_thick_mod = (len(self.track_objects)*1.4)-1.4*object_counter+0.4
				plt.plot(track_object.target_entries_x[target_counter], track_object.target_entries_entered[target_counter], linewidth=current_line_thick_mod, color=normalized_color_1)
				summary_string +='-ob'+str(object_counter+1)+':'+str(float("{0:.2f}".format(area_percent)))+'%/'+str(entries.count(0))
				np.savetxt('tra_'+self.save_name+'_Ob'+str(object_counter+1)+'_in_Area'+str(target_counter+1)+'perFrame_plot.txt', np.column_stack([track_object.target_entries_x[target_counter], track_object.target_entries_entered[target_counter]]),fmt='%10.2f')
			else:
				current_line_thick_mod = 1
				"""multiply by the index to get spaced out overlays"""
				offset_list = [x*(object_counter+1) for x in track_object.target_entries_entered[target_counter]]
				plt.plot(track_object.target_entries_x[target_counter], offset_list, linewidth=current_line_thick_mod, color=normalized_color_1, alpha=0.7)
		if use_thick_lines:		
			title_current = 'Target Area'+str(target_counter+1)+'(%inside/entries)'+"".join(summary_string)
			plt.title(title_current,  fontsize=10)
			plt.axis([0, frames, -0.1, 1.1])
		else:
			plt.axis([0, frames, -0.1, len(self.track_objects)+1])
		plt.yticks(np.arange(0, 1.1, 1))
		plt.ylabel('Entry')
		if not use_thick_lines:
			plt.xlabel('Frames')
			
	def plot_speed(self):
		"""plot each object speed in new graph"""
		for object_counter, track_object in enumerate(self.track_objects):
			plt.ion()
			fig = plt.figure(2)
			fig.set_size_inches(10, 5)
			ax = fig.gca()
			ax.set_autoscale_on(False)
			number_of_bins = len(track_object.distances_x)/round(self.fps)
			number_of_bins = max(1,number_of_bins)
			n, _ = np.histogram(track_object.distances_x, bins=number_of_bins)
			sy, _ = np.histogram(track_object.distances_x, bins=number_of_bins, weights=track_object.distances)
			mean_speed = sy / n
			x_speed = []
			average_speed = np.mean(mean_speed)
			start_of_track_time = track_object.distances_x[0]/self.fps
			for i in range(0, len(mean_speed)):
				x_speed.append(i+start_of_track_time)
			current_color_mod = int(float(object_counter)/len(self.track_objects)*128)
			normalized_color_256 = TrackClass.pick_color(current_color_mod)
			normalized_color_1 = [x/256 for x in normalized_color_256]
			normalized_color_1.reverse()
			plt.plot(x_speed, mean_speed,linestyle='-', marker='o', linewidth=2, color=normalized_color_1)
			np.savetxt('tra_'+self.save_name+'_SpeedObject'+str(object_counter+1)+'perSecond_plot.txt', np.column_stack([x_speed, mean_speed]),fmt='%10.2f')
			current_plot_max_y = int(max(mean_speed))+1
			current_plot_max_x = int(max(x_speed))+1
			title_current = 'Speed of Object '+str(object_counter+1)+' in pixels/s. Average speed: '+str(average_speed)
			plt.title(title_current,  fontsize=10)
			plt.axis([0,  current_plot_max_x, -0.1,current_plot_max_y], fontsize = 2)
			#plt.yticks(np.arange(0, current_plot_max_y+1, current_plot_max_y/2))
			plt.tight_layout()
			plt.tick_params(axis='both', which='minor', labelsize=5)
			plt.tick_params(axis='both', which='minor', labelsize=4)
			plt.xlabel('Time(s)')
			plt.ylabel('Pixels/s')
			plt.savefig('tr_'+self.save_name+'_Object'+str(object_counter+1)+'_SpeedPlot.png',dpi=300)	
			#plt.show(block=False)	
			plt.clf()
			
	def show_heat_image(self):
		for object_counter, track_object in enumerate(self.track_objects): 
			heat_image = np.zeros((self.height,self.width,3), np.uint8)
			#heat_image_LUT = np.zeros((height,width,3), np.uint8)
			past_points = np.array(track_object.past_positions, np.int32)
			for i in range(0, len(past_points)):
				heat_image[(past_points[i][1],past_points[i][0])] += 1
			heat_image = cv2.blur(heat_image,(4,4))
			heat_image = track_object.normalize_image(heat_image, 255)
			for target_object in self.target_objects:
				target = target_object.get_target_only()
				target = target.reshape((-1,1,2))
				cv2.polylines(heat_image,[target],True,(140,219,219))
			heat_image = track_object.applyLUT(heat_image)
			cv2.imwrite('tr_'+self.save_name+'_Object'+str(object_counter+1)+'_Heatmap.png',heat_image)
			
	def show_legend_image(self,legend_color_dict):	
		legend_image = np.zeros((len(legend_color_dict)*20,50,3), np.uint8)
		for index, entry in enumerate(legend_color_dict):
			cv2.line(legend_image, (5, index*20+9), (10, index*20+9), entry[1], 2)
			cv2.putText(legend_image, str(index+1), (27, index*20+14),
				cv2.FONT_HERSHEY_PLAIN, 1,(255,255,255),1)
		cv2.imwrite('tr_'+self.save_name+'_ColorLegend.png',legend_image)
		
	def show_speed_track_image(self):
		"""color pick functions"""
		"""red increase up to half linear, then stay max"""
		red = lambda x: x*2 if x < 128 else 255 
		"""keep green linear"""
		green = lambda x: x
		"""blue stay 0 to half, then up linear"""
		blue = lambda x: x*2-128 if x > 128 else 0
		track_image = np.zeros((self.height,self.width,3), np.uint8)
		"""plot all areas in one"""
		for object_counter, target_object in enumerate(self.target_objects):
			target = target_object.get_target_only()
			target = target.reshape((-1,1,2))
			cv2.polylines(track_image,[target],True,(100,100,100))
		for track_object in self.track_objects:
			speed_track = track_object.get_normalized_speed_only()
			for i in range (0, len(track_object.past_positions)-1):
				start_line_tuple = ( int(track_object.past_positions[i][0]), int(track_object.past_positions[i][1]) )
				end_line_tuple = ( int(track_object.past_positions[i+1][0]), int(track_object.past_positions[i+1][1]) )
				cv2.line(track_image,start_line_tuple, end_line_tuple ,
					(red(speed_track[i]), green(speed_track[i]), blue(speed_track[i])), 1)
		cv2.imwrite('tr_'+self.save_name+'_AllObjectsSpeeds.png',track_image)
		"""plot each by itself"""
		for object_counter, track_object in enumerate(self.track_objects): 
			track_image = np.zeros((self.height,self.width,3), np.uint8)
			for target_object in self.target_objects:
				target = target_object.get_target_only()
				target = target.reshape((-1,1,2))
				cv2.polylines(track_image,[target],True,(100,100,100))
			speed_track = track_object.get_normalized_speed_only()
			for i in range (0, len(track_object.past_positions)-1):
				start_line_tuple = ( int(track_object.past_positions[i][0]), int(track_object.past_positions[i][1]) )
				end_line_tuple = ( int(track_object.past_positions[i+1][0]), int(track_object.past_positions[i+1][1]) )
				cv2.line(track_image,start_line_tuple, end_line_tuple ,
					(red(speed_track[i]), green(speed_track[i]), blue(speed_track[i])), 1)
			cv2.imwrite('tr_'+self.save_name+'_Object'+str(object_counter+1)+'_Speed.png',track_image)
				
	def show_tracks_image(self):
		"""also create a legend for colors"""
		legend_color_dict = []
		track_image = np.zeros((self.height,self.width,3), np.uint8)
		track_image[:] = (255, 255, 255)
		"""plot all tracks in one"""
		for object_counter, track_object in enumerate(self.track_objects):
			track = track_object.get_track_only()
			normalized_color = int(float(object_counter)/len(self.track_objects)*128)
			cv2.polylines(track_image,[track],False,(TrackClass.pick_color(normalized_color)), 1)
			np.savetxt('tra_'+self.save_name+'_Object'+str(object_counter+1)+'_XYvector.txt', track, fmt='%10.2f', delimiter='\t')  
			legend_color_dict.append((object_counter,TrackClass.pick_color(normalized_color)))
		"""plot areas"""
		for object_counter, target_object in  enumerate(self.target_objects):
			target = target_object.get_target_only()
			target = target.reshape((-1,1,2))
			cv2.polylines(track_image,[target],True,(100,100,100))
			text_coords =tuple(target[0][0])
			cv2.putText(track_image,"target "+str(object_counter+1), text_coords,
				cv2.FONT_HERSHEY_SIMPLEX, 0.5,(0,0,00),1)
			np.savetxt('tra_'+self.save_name+'_Target'+str(object_counter+1)+'_XYvector.txt', track, fmt='%10.2f', delimiter='\t')  
		cv2.imwrite('tr_'+self.save_name+'_AllObjectsTracks.png',track_image)	
		"""plot each by itself"""
		for object_counter, track_object in enumerate(self.track_objects): 
			track_image = np.zeros((self.height,self.width,3), np.uint8)
			track_image[:] = (255, 255, 255)
			track = track_object.get_track_only()
			normalized_color = int(float(object_counter)/len(self.track_objects)*128)
			cv2.polylines(track_image,[track],False,(TrackClass.pick_color(normalized_color)), 1)
			"""plot areas"""
			for target_counter, target_object in enumerate(self.target_objects):
				target = target_object.get_target_only()
				target = target.reshape((-1,1,2))
				cv2.polylines(track_image,[target],True,(100,100,100))
				text_coords =tuple(target[0][0])
				cv2.putText(track_image,"target "+str(target_counter+1), text_coords,
					cv2.FONT_HERSHEY_SIMPLEX, 0.5,(0,0,00),1)
			cv2.imwrite('tr_'+self.save_name+'_Object'+str(object_counter+1)+'_Track.png',track_image)	
		return legend_color_dict

	def clean_up(self):
		self.camera.release()
		#self.videoFile.release()	
		
	def show_menu(self, key):
		cv2.rectangle(self.frame, (0, 0),(450,50),(0,100,100),-1)
		cv2.putText(self.frame,"HELP MENU", (5,14),
			cv2.FONT_HERSHEY_SIMPLEX, 0.5,(255,255,255),1)
		cv2.putText(self.frame,"Press 'o' to outline object or 't' to outline target areas", (5,30),
			cv2.FONT_HERSHEY_SIMPLEX, 0.5,(255,255,255),1)
		cv2.putText(self.frame,"Press space bar to continue", (5,45),
			cv2.FONT_HERSHEY_SIMPLEX, 0.5,(255,255,255),1)
		cv2.imshow("frame", self.frame)
		user_input = cv2.waitKey(0);
		return user_input
	
	def do_track(self):
		"""set up log but don't write to file"""
		logging.basicConfig(level=logging.DEBUG)
		logger = logging.getLogger(__name__)
		logging.disable(logging.CRITICAL)
		try:
			frame_count = 0
			min_sec = datetime.now().strftime('%M%S')
			self.initialize_stream(logger)
			self.show_instructions(logger)
			#self.videoFile  = cv2.VideoWriter()
			#self.videoFile.open('video2.avi', cv2.cv.CV_FOURCC('I','4','2','0'), 20, (self.height, self.width), False)
			"""define track termination condition"""
			termination = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 80, 1)
			user_input = cv2.waitKey(0);
			"""enable logging to file if 'l' is pressed"""
			if user_input is 108:
				logger = TrackClass.set_logger(logger)
				user_input = cv2.waitKey(0);
			logger.info('Menu key pressed:'+str(user_input))
			current_object_update_offset = 0
			if user_input in range (49,57):
				object_updates_per_frame = user_input-48
				logger.info('Objects updates frame:', str(object_updates_per_frame))
			else:
				object_updates_per_frame = None
				logger.info('Objects updates frame:all')
			if user_input in range (97,105):
				area_updates_per_frame = user_input-96
				logger.info('Area updates frame:', str(area_updates_per_frame))
			else:
				area_updates_per_frame = None
				logger.info('Area updates frame:all')	
			"""keep grabbing until camera off, or movie file over"""
			logger.info('Starting Tracking')
			t0 = time.clock()
			fps_frames = 0
			current_fps = 0
			while True:
				(grabbed, self.frame) = self.camera.read()
				if not grabbed:
					#self.frame = copied_image
					break
				frame_count += 1
				fps_frames += 1
				if time.clock() - t0 > 1:
					t0 = time.clock()
					current_fps = fps_frames
					fps_frames = 0
				logger.debug("Frame:"+str(frame_count))
				key = cv2.waitKey(1) & 0xFF
				logger.debug("Key pressed Loop:"+str(key))
				current_object_update_offset = self.check_object_position(termination, frame_count, object_updates_per_frame, area_updates_per_frame, current_object_update_offset, logger)	
				self.draw_all_targets(logger)
				#cv2.imwrite('Screenshot%d.jpg' %frame_count,self.frame)
				"""if space is pressed"""
				if key is 32:
					key = self.show_menu(key)
					logger.debug("show menu help")
					if key is 111:
						key = 0
				"""if o is pressed"""	
				while key is  111:
					self.temp_area = []
					logger.debug("getting object outline")
					key = self.get_object_outline(key)
					cv2.imshow("frame", self.frame)
					if key is not 111:
						break	
				"""if t is pressed"""
				while key is  116: 
					self.temp_area = []
					self.draw_all_targets(logger)
					logger.debug("getting object outline")
					key = self.get_target_outline(key)
					"""if t pressed again, draw next"""
					cv2.imshow("frame", self.frame)
					if key is not 116:
						break
				cv2.putText(self.frame,str(current_fps)+"FPS", (5,self.height-10),
						cv2.FONT_HERSHEY_SIMPLEX, 0.4,(0,140,100),1)
				cv2.imshow("frame", self.frame)
				if key is ord("q"):
					break
				if not grabbed:
					break
				"""copy last image in case the next grab is unsucessful"""
				copied_image = copy.copy(self.frame)
			logger.info('Done Tracking')
			cv2.rectangle(copied_image, (0, 0),(420,20),(0,0,200),-1)
			cv2.putText(copied_image,"WRITING DATA, please wait for this image to close...", (5,14),
						cv2.FONT_HERSHEY_SIMPLEX, 0.5,(0,0,0),1)
			cv2.imshow("frame", copied_image)
			cv2.waitKey(1) & 0xFF
			self.write_data_handler(frame_count, copied_image, logger)
			self.clean_up()
		except:
			logger.exception('Exception in main handler')
			raise
			
	@staticmethod	
	def set_logger(logger):
		LOG_FILENAME = 'logging_file.log'
		handler = logging.FileHandler(LOG_FILENAME)
		handler.setLevel(logging.DEBUG)
		formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
		handler.setFormatter(formatter)
		logger.addHandler(handler)
		logging.disable(logging.NOTSET)
		return logger	
		
	@staticmethod		
	def pick_color(LUT_value):
		RGB_percent = colorsys.hsv_to_rgb(float(LUT_value)/128,1, 1)
		RBG_vals_normalized = [x*256 for x in RGB_percent]
		return RBG_vals_normalized
