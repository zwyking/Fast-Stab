# The code of 《Fast-Full-frame-Video-Stabilization-with-Iterative-Optimization》


* # Synthetic Dataset
You can run ```python assets/save_training_video_to_disk.py``` to prepare your synthetic dataset. Of course, the corresponding datasets should be prepared firstly (illustrated in the main paper). Then, you should also adjust the directory path in ```save_training_video_to_disk.py```, including ```--image_data_path, --csv_path, --save_dir, coco_path```.

* # Generate Confidence Maps
You can run ``pre_video_flow_process.py`` to generate a confidence map sequence for the input video. Before running, please prepare the ``PDCNet`` code and make sure it runs successfully.

* # Core model codes
You can find the code for the network model corresponding to the paper in the ```core_model``` folder, including ```model.py```, ```dataset.py```, and ```loss.py```, etc.

* # Citation
If you find this code helpful, please cite:
