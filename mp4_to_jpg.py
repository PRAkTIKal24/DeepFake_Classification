'''
Using OpenCV takes a mp4 video and produces a number of images.
Requirements
----
You require OpenCV 3.2 to be installed.
Run
----
Place file in same directory as filenames.
Open the mp4_to_jpg.py and edit student name and filenames. Then run:
$ cd <file_location>
$ python mp4_to_jpg.py
Which will produce a folder based on student name containing all images for all videos.
'''
import cv2
import numpy as np
import os
from os import listdir

fps = 30
path_mani = '../manipulated_sequences/Deepfakes/c23/videos'
path_orig = '../original_sequences/youtube/c23/videos'
typ_orig = 'orignal'
typ_mani = 'manipulated'

def mp4_to_jpegs(seq_type, filename,fps, path):
    # Playing video from file:
    print(filename)
    cap = cv2.VideoCapture(path+'/'+filename)
    cap.set(cv2.CAP_PROP_FPS, fps)
    title=filename.split('.')[0]
    try:
        if not os.path.exists(seq_type):
            os.makedirs(seq_type)
    except OSError:
        print ('Error: Creating directory')

    currentFrame = 0
    while(True):
        # Capture frame-by-frame
        ret, frame = cap.read()
        if not ret:
            break

        # Saves image of the current frame in jpg file
        name = './'+seq_type+'/'+str(title)+'-' + str(currentFrame) + '.jpg'
        print ('Creating...' + name)
        cv2.imwrite(name, frame)

        # To stop duplicate images
        currentFrame += 1

    # When everything done, release the capture
    cap.release()
    #cv2.destroyAllWindows()

def mp4s_to_jpegs(typ,filenames,fps,path):
    for filename in filenames:
        mp4_to_jpegs(typ,filename,fps,path)

if __name__ == '__main__':
    mp4s_to_jpegs(typ_mani,listdir(path_mani),fps,path_mani)
    mp4s_to_jpegs(typ_orig,listdir(path_orig),fps,path_orig)
