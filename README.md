# ObjectTracker - Haar-like classifier

##### Track mouse in open environment with good accuracy
#

##### THIS USES THE MORE ACCURATE BUT OBJECT-SPECIFIC Haar-like feature ALGORITHM, COMPARE WITH ObjectTracker_CAMshift FOR COMPARISON
#



If you look at the other repository
[ ObjectTracker_CAMshift](https://github.com/ninehundred1/Object_Track_CAMShift
 "CAMshift") you can see the limitation of accurate tracking and a shaky track. One reason being that the background is not uniformly illuminated and a threshold not being restricted to only the mouse. 
 
 Below are examples of a backprojected Histogram for a few frames of the movie. You can see that there are large regions that also share the same pixel values as our mouse, therefore when the meanshift algorithm tries to find the area with maximal pixel density will move around any of those white regions. 

It might even be locked in in the bottom right corner, as the density of the background is high (white), but there is an area of low (black) around it, so it will not exit the bottom corner, even though the mouse moved away. 


![OBJECTTRACKER_allobjects](http://i.imgur.com/VgZKFd6.jpg)

You can also see that finding the center of the mouse is impossible, as there is so much noise not limited to the actual shape of the mouse sourounding the mouse.

##### Improvement of tracking by training classifiers
#
To enable more smooth and accurate tracking, the use of a trained classifier set might lead to a better performance. 
In this case this can be done with **CV2s Haar-like classifiers**

The steps as also explained 
 [ here on the CV2 site:](http://docs.opencv.org/3.1.0/d7/d8b/tutorial_py_face_detection.html#gsc.tab=0
 "CV2")



![OBJECTTRACKER_allobjects](http://i.imgur.com/pK1lJ82.jpg)

In short:

During training classifiers are generated that match any of the 5 shapes/contours on the left. In the example of the face, you can stretch those boxes to form a good representation of the eyes in the face (in the center image of the face you can see the dark link are the eyes, the white line below is below the eyes, in the right face you can see the dark two sides are the eyes, the white central bar the space between the eyes).

What the training algorithm does is go through 1000s of images and extract those features specific to your input (in our case a black mouse) and then looks for consistency. Once enough are found, those can be used for any other image in a certain range to extract those features.

First we need to make a training set consisting of many mice on top of random backgrounds.

*This can be automated like this sequence for 1000s of frames from the movie and 1000s of random photos:*


![OBJECTTRACKER_allobjects](http://i.imgur.com/gQkVFJA.jpg)


Below are 9 images from the training set (total of 2000 images)

![OBJECTTRACKER_allobjects](http://i.imgur.com/xb9maPa.jpg)

The second set needed for training is just random images of the same size without a mouse.

Below is the comparison of the tracking using the trained classifier (left) with the unspecific CAMshift tracking (right). Note the much smoother track and the truthful representation of the center point of the mouse.

![Link here](http://i.imgur.com/nhJyIcJ.gif)



#### What this file does (same as Object_track_CAMshift):
Creates data output in the form of images and also saves the raw data in .txt files.

Below is an example of the output of two tracked objects. The upper left image shows the speed of the objects, blue like colors depict
a slow speed, brighter colors a faster speed. At the top right the track of two objects (yellow and blue) is shown and three 
target areas that have been defined.

The bottom panel shows the times the different objects (yellow and blue) were inside target area 2.



*Output Example of all objects*
![OBJECTTRACKER_allobjects](http://i.imgur.com/Odb3XIc.png)



The next graphic shows the track, speed and position in the form of a heatmap of object 1.
The bottom panels show the distance object 1 covered during the tracking and the speed while tracked.

*Output Example of the details of the track object 1*
![OBJECTTRACKER_allobjects2](http://i.imgur.com/NgPeQgm.png)




##### Instructions are shown in the actual movie file or stream before it starts playing (eg see below).
![OBJECTTRACKER_allobjects2](http://i.imgur.com/eBl9VPK.png)





### To run you need python 2.7 and the following packages:

- cv2
- matplotlib
- pyplot
- numpy
- Tkinter

Download all files (click Download ZIP on the right), unzip and copy them into a folder YourFolder then you can run
the wrapper file do_track.py from the windows command prompt
```
c:YourFolder python do_track.py
```

I will work on some more fixes and then also generate a standalone executable file that does not need python.

emails to:
- <fuschro@gmail.com>