import scipy.io as sio
import numpy as np
import h5py
import os
SEQUENCE_LENGTH = 12
inds = [3, 2, 1, 4, 5, 6, 0, 7, 8, 10, 16, 15, 14, 11, 12, 13]
print("inds", sorted(inds))
subject_list = [[1, 5, 6, 7, 8], [9], [11]]
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
splitNames = ["train", "val", "test"]
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
          meta_Y3d_mono = data['Y3d_mono'].reshape(17, 3, n)
          bboxx = data['bbox'].transpose(1, 0)
          for i in range(0, data['num_images'] - 5 * SEQUENCE_LENGTH * 2, 5):
            # if test and image number is not a multiple of 200
            if split == 1 and i % 200 != 0:
                continue
            inputId.append(range(i+1, i+1 + 5*(SEQUENCE_LENGTH), 5))
            targetId.append(range(i+1+5*SEQUENCE_LENGTH, i+1+5*SEQUENCE_LENGTH*2, 5))
            inputSub = meta_Y2d[inds, :, i:i+5*SEQUENCE_LENGTH:5].transpose(2,0,1)
            inputSub = inputSub.reshape((list(inputSub.shape[:1])+[-1]))
            input_2d.append(inputSub)
            targetSub = meta_Y2d[inds, :, i+5*SEQUENCE_LENGTH:i+5*SEQUENCE_LENGTH*2:5].transpose(2,0,1)
            targetSub = targetSub.reshape((list(targetSub.shape[:1])+[-1]))
            target_2d.append(targetSub)
            subjects.append(subject)
            actions.append(action)
            subactions.append(subaction)
            cameras.append(camera)
            istrain.append(1 - split)
            num += 1
  print("split:{} - n instances: {}".format(splitNames[split], num))      
  h5name = SAVE_PATH + '{}_flat.h5'.format(splitNames[split])
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
    
