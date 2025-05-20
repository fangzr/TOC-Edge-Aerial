# Task-Oriented Communications for Visual Navigation with Edge-Aerial Collaboration

The repository for paper 'Task-Oriented Communications for Visual Navigation with Edge-Aerial Collaboration in Low Altitude Economy'.

## Abstract

To support the Low Altitude Economy (LAE), precise unmanned aerial vehicles (UAVs) localization in urban areas where global positioning system (GPS) signals are unavailable is crucial. Vision-based methods offer a viable alternative but face severe bandwidth, memory and processing constraints on lightweight UAVs. 

Inspired by mammalian spatial cognition, we propose a task-oriented communication framework, where UAVs equipped with multi-camera systems extract compact multi-view features and offload localization tasks to edge servers. We introduce the **O**rthogonally-constrained **V**ariational **I**nformation **B**ottleneck encoder (O-VIB), which incorporates automatic relevance determination (ARD) to prune non-informative features while enforcing orthogonality to minimize redundancy. This enables efficient and accurate localization with minimal transmission cost.

Extensive evaluation on a dedicated LAE UAV dataset shows that O-VIB achieves high-precision localization under stringent bandwidth budgets.

## System Model

![System Architecture](https://raw.githubusercontent.com/fangzr/TOC-Edge-Aerial/refs/heads/main/figure/system_model_00.jpg)

Our system operates in a UAV-edge collaborative framework designed for GPS-denied urban environments. The model consists of:

- **Multi-camera UAV System**: Captures multi-directional views (Front, Back, Left, Right, Down) for comprehensive spatial awareness
- **Edge Server Infrastructure**: Maintains a geo-tagged feature database, enabling efficient localization
- **Communication-Efficient Design**: Optimizes the trade-off between localization accuracy and bandwidth consumption

The UAV captures multi-view images at each time step, extracts high-dimensional features through a feature extractor, and transmits compressed representations to edge servers. Our objective is to minimize localization error while keeping communication costs below a specified threshold.

## Multi-View UAV Dataset

![Simulation Environment](https://raw.githubusercontent.com/fangzr/TOC-Edge-Aerial/refs/heads/main/figure/simulation_00.jpg)

We collected a comprehensive dataset using the CARLA simulator to facilitate research on multi-view UAV visual navigation in GPS-denied environments:

### Dataset Specifications:
- **Environments**: 8 representative urban maps in CARLA (Town01, Town02, Town03, Town04, Town05, Town06, Town07, Town10HD)
- **Collection Method**: UAV flying at constant height following road-aligned waypoints with random direction changes
- **Camera Configuration**: 5 onboard cameras capturing different angles and directions (Front, Back, Left, Right, Down)
- **Image Types**: RGB, semantic, and depth images at 400×300 pixel resolution
- **Scale**: 357,690 multi-view frames with precise localization and rotation labels
- **Hardware**: Collected using 4×RTX 5000 Ada GPUs
- **URL**: [Hugging Face](https://huggingface.co/datasets/Peter341/Multi-View-UAV-Dataset)

### Dataset Structure:
```
Dataset_CARLA/Dataset_all/
├── town01_20241217_215934.tar
├── town02_20241218_153549.tar
├── town03_20241217_222228.tar
├── town04_20241217_225428.tar
├── town05_20241218_092919.tar
├── town06_20241217_233050.tar
├── town07_20241218_153942.tar
└── town10hd_20241218_151215.tar

town05_20241218_092919/
├── calibration/
│   └── camera_calibration.json    # Contains parameters for all 5 UAV onboard cameras
├── depth/                         # Depth images from all cameras
│   ├── Back/
│   │   ├── 000000.npy             # Depth data in NumPy format
│   │   ├── 000000.png             # Visualization of depth data
│   │   └── ...
│   ├── Down/
│   ├── Front/
│   ├── Left/
│   └── Right/
├── metadata/                      # UAV position, rotation angles and timestamps
│   ├── 000000.json
│   ├── 000001.json
│   └── ...
├── rgb/                           # RGB images from all cameras (PNG format only)
│   ├── Back/
│   ├── Down/
│   ├── Front/
│   ├── Left/
│   └── Right/
└── semantic/                      # Semantic segmentation images (PNG format only)
    ├── Back/
    ├── Down/
    ├── Front/
    ├── Left/
    └── Right/
```

### Data Format Details:
- **RGB Images**: Standard PNG format (400×300 pixels)
- **Semantic Images**: Pixel-wise semantic segmentation in PNG format
- **Depth Images**: Available in both PNG (visualization) and NumPy (.npy) formats for precise depth values
- **Metadata**: JSON files containing UAV position coordinates, rotation angles, and timestamps
- **Calibration**: JSON file with intrinsic and extrinsic parameters for all five cameras

### Dataset Visualization:

#### RGB Camera View
![RGB Visualization](https://raw.githubusercontent.com/fangzr/TOC-Edge-Aerial/refs/heads/main/figure/rgb_animation.gif)

#### Semantic Segmentation View
![Semantic Visualization](https://raw.githubusercontent.com/fangzr/TOC-Edge-Aerial/refs/heads/main/figure/semantic_animation.gif)

#### Depth Map View
![Depth Visualization](https://raw.githubusercontent.com/fangzr/TOC-Edge-Aerial/refs/heads/main/figure/depth_animation.gif)

The dataset provides a realistic simulation of UAV flight in urban environments where GPS signals might be compromised or unavailable. It enables researchers to develop and test novel algorithms for visual navigation, localization, and perception tasks in GPS-denied environments.

## Feature Extraction (UAV-side)

![Encoder Architecture](https://raw.githubusercontent.com/fangzr/TOC-Edge-Aerial/refs/heads/main/figure/encoder_00.jpg)

Our feature extraction pipeline is designed for robust multi-view feature extraction under limited bandwidth:

### Key Components:
- **CLIP-based Vision Backbone**: Utilizes CLIP Vision Transformer (ViT-B/32) pretrained on large-scale natural image-text pairs
- **Feature Processing**: Each image undergoes preprocessing (resize, normalize, tokenize) before feature extraction
- **Normalization**: Features are normalized to lie on the unit hypersphere, improving numerical stability and facilitating cosine similarity-based retrieval
- **Multi-view Feature Tensor**: Final representation constructed by concatenating view-wise embeddings, capturing a rich panoramic representation of the UAV's surroundings

This pipeline creates a memory base for the visual navigation system, enabling efficient localization with minimal communication overhead.

## Position Prediction (Edge Server-side)

![Decoder Architecture](https://raw.githubusercontent.com/fangzr/TOC-Edge-Aerial/refs/heads/main/figure/decoder_00.jpg)

The edge server receives compressed representations from the UAV and estimates the UAV's position using a sophisticated multi-view attention fusion mechanism:

### Position Inference:
- **Multi-view Attention Fusion**: Integrates information from multiple camera views
- **Hybrid Estimation Method**: Combines direct regression and retrieval-based inference
- **Adaptive Weighting**: Balances regression and retrieval estimates based on confidence scores
- **Geo-tagged Database**: Utilized for querying position information

This end-to-end pipeline optimizes the trade-off between localization accuracy and communication efficiency, enabling precise UAV navigation in GPS-denied environments with constrained wireless bandwidth.

## Hardware Implementation

![Hardware Setup](https://raw.githubusercontent.com/fangzr/TOC-Edge-Aerial/refs/heads/main/figure/hardware_00.jpg)

We validated our approach using a physical testbed with real hardware components:

### Hardware Configuration:
- **UAV Compute**: Jetson Orin NX 8GB for encoding five camera streams
- **Communication**: IEEE 802.11 wireless transmission to nearby roadside units (RSUs)
- **Relay RSU**: Raspberry Pi 5 16GB that forwards data via Gigabit Ethernet to cloud edge servers when overloaded
- **Edge RSU**: Jetson Orin NX Super 16GB performing on-board inference

This hardware implementation allowed us to evaluate algorithm encoding/decoding complexity and latency in real-world conditions, confirming that our O-VIB framework delivers high-precision localization with minimal bandwidth usage.

### Position Prediction Demonstration:

![Position Prediction Demo](https://raw.githubusercontent.com/fangzr/TOC-Edge-Aerial/refs/heads/main/figure/prediction_video_random.gif)

The green dot represents the Ground Truth (GT), which is the actual coordinate of the UAV. The red dot represents the Top 1 prediction (Pred), which is the most accurate prediction. However, Top 2 and Top 3 are alternative prediction locations provided by the algorithm, but their accuracy is usually much lower than the Top 1 prediction.

## Paper

Our research is detailed in the paper: [Task-Oriented Communications for Visual Navigation with Edge-Aerial Collaboration in Low Altitude Economy](https://www.researchgate.net/publication/391159895_Task-Oriented_Communications_for_Visual_Navigation_with_Edge-Aerial_Collaboration_in_Low_Altitude_Economy)


## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

This work of Y. Fang was supported in part by the Hong Kong SAR Government under the Global STEM Professorship and Research Talent Hub,  the Hong Kong Jockey Club under the Hong Kong JC STEM Lab of Smart City (Ref.: 2023-0108). This work of J. Wang was partly supported by the National Natural Science Foundation of China under Grant No. 62222101 and No. U24A20213, partly supported by the Beijing Natural Science Foundation under Grant No. L232043 and No. L222039, partly supported by the Natural Science Foundation of Zhejiang Province under Grant No. LMS25F010007. The work of S. Hu was supported in part by the Hong Kong Innovation and Technology Commission under InnoHK Project CIMDA. The work of Y. Deng was supported in part by the National Natural Science Foundation of China under Grant No. 62301300. 

## Contact

For any questions or discussions, please open an issue or contact us at zhefang4-c [AT] my [DOT] cityu [DOT] edu [DOT] hk.
