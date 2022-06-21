# Launching Real3D-Aug for semantic dataset

## Adjusting config file

1. Adjust paths
   - dataset_path - path to dataset files
   - maps_path - path to directory, where created map will be stored (path must end with: /maps/small/npz, but these directories is created automatically)
   - annotation_path - path to directory, where created bouding boxes of frame objects will be stored
   - bbox_path - path to directory, where cutted objects will be stored
   - output_path - path to directory, where outputs of the algorithm will be stored

Can run augmentation the procedure by running the python script TASK/Real3D-Aug/insertion.py with modified path to the original dataset.
