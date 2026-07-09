# net_hole_detector
ROS Noetic Package to detect automatically broken holes in nets and estimate the real distance between the camera and the holes.

This system combines **YOLO-World** for object detections (broken holes) and **Classical Computer Vision** for the escale estimation of the net (meter/pixel) based on known shape and real size of the net.

Added Stereo Cameras pipeline for the escale estimation of the net by calculating the disparity manually.

## Project Structure

```text
.
├── config/                 # Calibration file and tuning parameters
│   ├── UJI_tuning_params.yaml  <-- ¡IMPORTANT! Main Configuration
│   └── right_laser.yaml        <-- Intrinsec Camera Calibration
├── launch/                 # ROS launches
├── msg/                    # Messages definition (NetStats, Detection3D...)
├── scripts/                # Executable Nodes (Python)
├── src/                    # Internal Libraries (Estimators, Geometry)
├── weights/                # YOLO models weights (.pt)
└── envs/                   # Conda environment .yaml info

```

## ROS Dependencies

This package has been tested in **Ubuntu 20.04** and **ROS Noetic** make sure to have standard packages installed:

```bash
sudo apt install ros-noetic-vision-opencv ros-noetic-cv-bridge ros-noetic-image-transport
```

## Conda environment

Due to dependency conflicts between ultralytics and ROS, is strongly recommended to use the provided environment to avoid system issues. Assuming Conda or MiniConda is installed:

```bash
# Create environment from the yaml file (Execute in the root of the package)
conda env create -f envs/tandem_yolo.yaml
```

## How to use?

Due to this conflicts, we will have this setup:

In one terminal, we will execute:

```bash
roscore
```

and in another window of the same terminal we will launch the Preprocessing nodes, in charge of decompressing the images and aplying CLAHE if we desire so:

```bash
roslaunch net_hole_detector preprocessing.launch is_compressed:=true
```

The different arguments can be modified in the .launch file or in the terminal. We must specifie if the images we are expecting are compressed or not, to know if we must decompress them or not.

In a **different terminal**, we activate the conda environment

```bash
conda activate tandem_yolo
```

and we launch the Detection and Inference nodes, in charge of managing YOLO detections, infering scale and distance and publishing the results.

```bash
roslaunch net_hole_detector detection.launch image_topic:=/your/camera/topic is_rectified:=true
```

We must specifie if the images we are expecting are rectified or not, since we will use matrix K or matrix P (both are callibration matrixes) depending on the case. 

The commands specified above will execute the nodes, but they will not activate the inference and hole information gathering. There are two services to enable and disable that, they are the service net_hole_detector/inference_activation_srv and net_hole_detector/inference_deactivation_srv.

## Configuration/Tuning

All numeric parameters are centered in a key yaml file in the *config/UJI_tuning_params.yaml* path. So, if we desire to adjust any of them, they will be more accessible, and modification of Python code will not be needed.

## Camera Calibration

This system needs the calibration matrix of the camera (otherwise it will not work). We have a two steps system: in *config/right_laser.yaml* (we can add another configuration yaml file for another camera if we want to) we store the backup calibration, so when the process starts we have some matrix to default to; additionally we will try to subsribe to the CameraInfo topic (if it is published by the robot) if we are able to subscribe to this topic we will overwrite the previous calibration info.

## Recorded Case

This is meant to be used live, but if we want to try it with already recorded bagfiles, we must do as follows:

```bash
roscore
```

```bash
roslaunch net_hole_detector preprocessing.launch sim:=true is_compressed:=true
```

(again in the conda environment terminal):
```bash
roslaunch net_hole_detector detection.launch sim:=true is_rectified:=true
```

And also (in the 'normal' terminal, in another window)
```bash
rosbag play --clock your/desired/bagfile.bag
```

put the *sim* argument as *True* inthe roslaunches and the *--clock* flag in the rosbag play.

## system.launch

Additionally we have a launch that can trigger the whole process without CLAHE. We will not use this launch anyway.

## Inputs

In order to work, this pipeline will need the *image_topic* (defined in the preprocessing.launch file, can be modified there if needed) and, as we discussed before, the *camera_info_topic* or otherwise its associated yaml file with ALL the calibration info.

## Publications (Outputs)

We will publish different information abot the detections. Some of the info will be visual (different topics to see what are we detecting) and some of the info will be numerical (via ROS messages) so the robot can act accodingly.

The 'informative' topics (Custom messages of type *BoundingBox(Array).msg*, *Detection3D(Array).msg*, *NetStats.msg*) will be:

**/net_hole_detector/bounding_boxes** will publish the info of the boundingboxes detected by YOLO (the score/probability of the detection, x of boxcenter, y of boxcenter, width, height) normalized (the rest of the pipeline will interpret this results).

**/net_hole_detector/detections_3d** will publish the X,Y,Z coordinates (in meters) of the center of the box according to **Camera Coordinates**, a further transform will be needed to have it in World Coordinates.

We will provide **ALL** detections and its coordinates, so the robot can choose what to do with this information.

**/net_hole_detector/net_stat** will publish the scale (meters/pixel) and the calculated median area (in pixels and meters) of the net square grid of every image.

The main 'visual' topics (messages of type *sensor_msgs/Image*) will be:

*Note*: In this context, we are refering to *blobs* as **ALL** the detected net squares of the grid in every image (not only the broken ones). And as *valid* we mean that we apply some filters to remove as much noise or missdetected blobs as possible, we do it by requiring a minimum and maximum area, and a certain aspect ratio height/height to the detected contours. This allows us to calculate the average size in pixels of the net square grid and the meters/pixel scale of every image.

**/hole_detector/image_processed** we will see the original image decompressed and with CLAHE applied (if asked so).

**/net_hole_detector/blobs/binary** an image with the 'valid' detected blobs.

**/net_hole_detector/blobs/overlay** the original image with the same 'valid' blobs painted over it

**/net_hole_detector/debug_image** the original image with the detected bounding boxes painted over it and the Z distance and width written.

**/net_hole_detector/net_mask_yolo_fused** A fusion of the blobs and the yolo: we will see only the valid blobs inside the detected bounding box.

**/net_hole_detector/blob_axis_orientation** the minor and major axis of the detected hole given by two points for each axis and the orientation of the hole major axis, and the length of both axis in meters.

# Stereo Case:

The dependencies and the CONDA environment is the same as the mono case.

## How to use?

Again in this case, we will have to separate the launches due to conflicts between OpenCV and Ultralytics. Due to this conflicts, we will have this setup:

In one terminal, we will execute:

```bash
roscore
```

and in another window of the same terminal we will launch the Disparity related nodes (in this launch file, we must modifiy the following parameters: "left_camera_info_topic", "right_camera_info_topic", to match with the CameraInfo topic of each stereo camera):

```bash
roslaunch net_hole_detector stereo_manual_processing.launch 
```

In a **different terminal**, we activate the conda environment

```bash
conda activate tandem_yolo
```

and we launch the Detection nodes. We launch the same detection node twice (one for each stereo camera, with its corresponding namespaces), this node returns the coordinates of each boundingbox detected by YOLO in each camera (in this launch file, we must modify the following arguments: "left_final_image_topic", "right_final_image_topic", to match with the Image topics of each stereo camera). 

```bash
roslaunch net_hole_detector stereo_bboxes.launch
```

## How this works?

This pipeline is simpler than the mono case.

First we apply YOLO to both stereo cameras at the same time, these detections are published in the **/net_hole_detector/stereo_left/bounding_boxes** and **/net_hole_detector/stereo_right/bounding_boxes** topics respectively.

NOTE: As in the mono case, the YOLO detection coordinates are normalized according to the size of the image.

Then the detections are used by the **manual_disparity_calculator** node. This node is the core of the stereo pipeline, as it calculates the distance between the camera and the net by computing manually the disparity between the centers of the two bounding boxes (one from each camera). It does so following the next process:

- To start it focuses on the left image and gets the bounding box with the higher score, we can call it 'best_box'.
- Then, tries to find its matching box in the right image, to do so it searches for bounding boxes at the same height as the left image (with a certain tolerance).
- If the search is successfull then we have the 'best_box' in the left image and its corresponding pair in the right image, so we can calculate the disparity between the two center of each bounding box. Basically, the disparity is the difference (in absolute value) in pixels between a point in the left image and a point in the right image. This difference is only computated in the X axes, as we suppose the two points to be aligned on the Y axes.
- Once the disparity is calculated we use the following formula to get the distance, Z:

```text
Z = focal length * Baseline / disparity
```

We can access to the focal length of each camera via the CameraInfo topic. We know the Baseline (the distance between each stereo camera) as we have installed them. And we have just calculated the disparity between the two centers. So we estimate Z, the distance between the camera frame and the detected hole (in x,y,z coordinates). This info is stored in the topic **/net_hole_detector/stereo_detections_3d**.

Then, we only have to compose the TFs to get the detected hole in World NED coordinates which will be published in the **/net_hole_detector/hole** topic (this process is done by the **detection_to_world_pose.py** node). This will be the topic to be accessed by the controller to adjust the path in the approach to the hole.

To debug and see what is happening we have created the **stereo_visualizer.py** node that allows us to watch in real time via the **/net_hole_detector/debug_image** topic the images of the left stereo camera with the detected bounding boxes drawn over it. The optical center (cx,cy) of the image is also shown. This node can be modified to draw more or less information depending o what are we debugging at the time. Is a nice visual tool to help us understand what is happening in the process.

## Important

- Check the TFs of the stereo cameras (that they correspond to the real position).
- Check the baseline (the distance between the two stereo cameras) also corresponds to the real distance, it is referenced at the beggining of the **manual_disparity_calculator.py** node. In the simulation is important to place both cameras according to this baseline in the corresponding **.scn** file.

All of this is critical for the pipeline to work (otherwise the measurements will not be trustworthy).

## Simulation related things:

In order to check if the detection is accurated or not we have to compare it with a ground truth. To do so in the simulation, we have got the World NED coordinates of the hole, and we have published a static TF os this coordinates named **hole_ground_truth** (this works as the hole does not move). Then we only need to compare this coordinates to the World NED coordinates of our detections (that is the **/net_hole_detector/hole** topic). Then we simply calculate the euclidean distance between the tow coordinates and we have the 3D error. We publish it in the **/net_hole_detector/error** topic. We have seen the error is more or less stable around 10 cm.

