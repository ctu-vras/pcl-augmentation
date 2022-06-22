# Launching Real3D-Aug for semantic dataset

## Adjusting config file

1. Adjust paths
   - *dataset_path* - path to dataset files
   - *maps_path* - path to directory, where created map will be stored (path must end with: /maps/small/npz, but these directories is created automatically)
   - *annotation_path* - path to directory, where created bouding boxes of frame objects will be stored
   - *bbox_path* - path to directory, where cutted objects will be stored
   - *output_path* - path to directory, where outputs of the algorithm will be stored
2. Insertion paramethers
   - *classes* - set of classes indexes, which would augment frames (all these classes need to have filled: min_points, placement, labels_shortcut and labels)
   - *min_points* - minimal number of object points, which need to be visible to add object to frame
   - *random* - bool variable, if it is **True** method adds to frames as many objects as value of *number_of_object* is, however number of samples of each class will be randomly generated. If it is **False** algorithm uses *number_of_classes* as number of samples of each class, which adds to frames

Can run augmentation the procedure by running the python script TASK/Real3D-Aug/insertion.py with modified path to the original dataset.
