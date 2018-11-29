import scipy.io as sio
import numpy as np
import h5py
import os
import argparse
from operator import itemgetter
parser = argparse.ArgumentParser(description='Get Pose Sequence 2D')
parser.add_argument('--down_sample', type=float, default=0.0)
parser.add_argument("--only_train", action="store_true", default=False)
args = parser.parse_args()

inds = [3, 2, 1, 4, 5, 6, 0, 7, 8, 10, 16, 15, 14, 11, 12, 13]
subject_list = [1, 5, 6, 7, 8, 9, 11]
splitNames = ["all"]

action_list = np.arange(2, 17)
subaction_list = np.arange(1, 3)
camera_list = np.arange(1, 5)
IMG_PATH = '/Users/danielzeiberg/Documents/Human3.6/images/'
SAVE_PATH = '/Users/danielzeiberg/Documents/Human3.6/Processed/'
annot_name = 'matlab_meta.mat'

if not os.path.exists(SAVE_PATH):
  os.mkdir(SAVE_PATH)

maxLen = 0
for subject in subject_list:
  for action in action_list:
    for subaction in subaction_list:
      for camera in camera_list:
        folder_name = 's_{:02d}_act_{:02d}_subact_{:02d}_ca_{:02d}'.format(subject, action, subaction, camera)
        annot_file = IMG_PATH + folder_name + '/' + annot_name
        try:
          data = sio.loadmat(annot_file)
        except:
          print 'pass', "{}, {}".format(folder_name, annot_file)
          continue
        meta_Y2d = data['Y2d'].reshape(17, 2, -1)
        d = meta_Y2d[inds,:,0:-1:5]
        n = d.shape[-1]
        maxLen = max(maxLen, n)
print(maxLen)
instances = []
length = []
subjects = []
actions = []
subactions = []
cameras = []
for subject in subject_list:
  for action in action_list:
    for subaction in subaction_list:
      for camera in camera_list:
        print("subject: {} action: {} subaction:{}, camera: {}".format(subject, action, subaction, camera))
        folder_name = 's_{:02d}_act_{:02d}_subact_{:02d}_ca_{:02d}'.format(subject, action, subaction, camera)
        annot_file = IMG_PATH + folder_name + '/' + annot_name
        try:
          data = sio.loadmat(annot_file)
        except:
          print 'pass', "{}, {}".format(folder_name, annot_file)
          continue
        meta_Y2d = data['Y2d'].reshape(17, 2, -1)
        d = meta_Y2d[inds,:,0:-1:5]
        n = d.shape[-1]
        if maxLen > n:
          z = np.zeros((d.shape[0], d.shape[1], maxLen-n))
          d= np.append(d, z, axis=2)
        instances.append(d.transpose(2,0,1))
        length.append(n)
        subjects.append(subject)
        actions.append(action)
        subactions.append(subaction)
        cameras.append(camera)
# Save Data
h5name = SAVE_PATH + '{}_2D.h5'.format(splitNames[0])
f = h5py.File(h5name, 'w')
f['instances'] = instances
f["lengths"] = length
f["subjects"] = subjects
f["actions"] = actions
f["subactions"] = subactions
f["cameras"] = cameras
f.close()    
