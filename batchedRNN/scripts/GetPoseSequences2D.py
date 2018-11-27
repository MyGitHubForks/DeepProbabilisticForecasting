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

SEQUENCE_LENGTH = 12
inds = [3, 2, 1, 4, 5, 6, 0, 7, 8, 10, 16, 15, 14, 11, 12, 13]
subject_list = [[1, 5, 6, 7, 8], [9], [11]]
splitNames = ["train", "val", "test"]
if args.only_train:
  subject_list = [subject_list[0]]
  splitNames = [splitNames[0]]
action_list = np.arange(2, 17)
subaction_list = np.arange(1, 3)
camera_list = np.arange(1, 5)
IMG_PATH = '/Users/danielzeiberg/Documents/Human3.6/images/'
SAVE_PATH = '/Users/danielzeiberg/Documents/Human3.6/Processed/'
annot_name = 'matlab_meta.mat'

if not os.path.exists(SAVE_PATH):
  os.mkdir(SAVE_PATH)


# Split 0: train
# Split 1: test

for split in range(len(splitNames)):
  inputId = []
  targetId = []
  input_2d = []
  target_2d = []
  subjects = []
  actions = []
  subactions = []
  cameras = []
  istrain = []
  num = 0
  for subject in subject_list[split]:
    for action in action_list:
      print("subject: {} action: {}".format(subject, action))
      for subaction in subaction_list:
        for camera in camera_list:
          folder_name = 's_{:02d}_act_{:02d}_subact_{:02d}_ca_{:02d}'.format(subject, action, subaction, camera)
          annot_file = IMG_PATH + folder_name + '/' + annot_name
          try:
            data = sio.loadmat(annot_file)
          except:
            print 'pass', "{}, {}".format(folder_name, annot_file)
            continue
          n = data['num_images'][0][0]
          meta_Y2d = data['Y2d'].reshape(17, 2, n)
          for i in range(0, data['num_images'] - 5 * SEQUENCE_LENGTH * 2, 5):
            # if test and image number is not a multiple of 200
            if split == 1 and i % 200 != 0:
                continue
            inputId.append(range(i+1, i+1 + 5*(SEQUENCE_LENGTH), 5))
            targetId.append(range(i+1+5*SEQUENCE_LENGTH, i+1+5*SEQUENCE_LENGTH*2, 5))
            input_2d.append(meta_Y2d[inds, :, i:i+5*SEQUENCE_LENGTH:5].transpose(2,0,1))
            target_2d.append(meta_Y2d[inds, :, i+5*SEQUENCE_LENGTH:i+5*SEQUENCE_LENGTH*2:5].transpose(2,0,1))
            subjects.append(subject)
            actions.append(action)
            subactions.append(subaction)
            cameras.append(camera)
            istrain.append(1 - split)
            num += 1
  h5name = SAVE_PATH + '{}_2D.h5'.format(splitNames[split])
  if args.down_sample and splitNames[split] == "train":
    h5name = SAVE_PATH + '{}_2D_down_sample_{}.h5'.format(splitNames[split], args.down_sample)
    nRows = len(inputId)
    selection = list(np.random.choice(range(nRows), size=np.ceil(nRows * args.down_sample).astype(int),
      replace=False).astype(int))
    selection = [np.asscalar(a) for a in selection]
    num = np.ceil(nRows * args.down_sample).astype(int)
    inputId = itemgetter(*selection)(inputId)
    targetId = itemgetter(*selection)(targetId)
    input_2d = itemgetter(*selection)(input_2d)
    target_2d = itemgetter(*selection)(target_2d)
    subjects = itemgetter(*selection)(subjects)
    actions = itemgetter(*selection)(actions)
    subactions = itemgetter(*selection)(subactions)
    cameras = itemgetter(*selection)(cameras)
    istrain = itemgetter(*selection)(istrain)
  print("split:{} - n instances: {}".format(splitNames[split], num))
  f = h5py.File(h5name, 'w')
  f['inputId'] = inputId
  f['targetId'] = targetId
  f['input2d'] = input_2d
  f['target2d'] = target_2d
  f['subject'] = subjects
  f['action'] = actions
  f['subaction'] = subactions
  f['camera'] = cameras
  f['istrain'] = istrain
  f.close()
    
