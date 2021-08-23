# gpu_voxel_docker
```bash
git clone https://github.com/tjdalsckd/gpu_voxel_docker.git
cd gpu_voxel_docker
bash build.sh
bash start.sh
cd /root/workspace/libfranka_gpu_voxel
bash build.sh
source export_MODEL_PATH.sh
./gvl_ompl_planner 0 0 0 0 0 0

```
terminal 2
```bash
cd gpu_voxel_docker
bash multi_terminal.sh 
cd ~/workspace
```


# multi-view calibration
```bash
cd ~/workspace/multi-view
pip install -r requirements.txt
./each_cam_calibration.sh
./stereo_cam_calibration.sh
source /opt/ros/kinetic/setup.bash
source /root/catkin_ws/devel/setup.bash
./start_cam.sh
```

# hand -eye calibration
```bash
cd ~/workspace/calibration_application/application
python3 setup.py install
cd samples
python3 start_calibrationSample1.py

docker cp <filename> gpu_voxels:/root/workspace/calibration_application/application/samples
```

# move calibraiton file
```bash
cp Result/TBaseToCam.txt cd /root/workspace/libfranka_gpu_voxel/TBaseToCamera.txt
```
#start 
```bash
./gvl_ompl_planner 0 0 0 0 0 0
```
# pvnet_docker
