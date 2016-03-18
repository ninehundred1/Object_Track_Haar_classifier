from TrackClass_haar import TrackClass

def main():
	"""run this file to start the tracking of object using a webcam or 
	loading a movie file"""
	tracker = TrackClass()
	tracker.do_track()

if __name__ == "__main__":
	main()
	