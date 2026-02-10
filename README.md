# Controllable Human-Object Interaction Synthesis (ECCV 2024 Oral)
This is the official implementation for the ECCV 2024 [paper](https://arxiv.org/pdf/2312.03913). For more information, please check the [project webpage](https://lijiaman.github.io/projects/chois/).

![CHOIS Teaser](chois_teaser.png)

## Environment Setup
> Note: This code was developed on Ubuntu 20.04 with Python 3.8, CUDA 11.3 and PyTorch 1.11.0.

Clone the repo.
```
git clone https://github.com/lijiaman/chois_release.git
cd chois_release/
```
Create a virtual environment using Conda and activate the environment. 
```
conda create -n chois_env python=3.8
conda activate chois_env 
```
Install PyTorch. 
```
conda install pytorch==1.11.0 torchvision==0.12.0 torchaudio==0.11.0 cudatoolkit=11.3 -c pytorch
```
Install PyTorch3D. 
```
conda install -c fvcore -c iopath -c conda-forge fvcore iopath
conda install -c bottler nvidiacub
pip install --no-index --no-cache-dir pytorch3d -f https://dl.fbaipublicfiles.com/pytorch3d/packaging/wheels/py38_cu113_pyt1110/download.html
```
Install human_body_prior. 
```
git clone https://github.com/nghorbani/human_body_prior.git
pip install tqdm dotmap PyYAML omegaconf loguru
cd human_body_prior/
python setup.py develop
```
Install BPS.
```
pip install git+https://github.com/otaheri/chamfer_distance
pip install git+https://github.com/otaheri/bps_torch
```
Install other dependencies. 
```
pip install -r requirements.txt 
```

## Docker (GPU)
This repo includes a `Dockerfile` + `docker-compose.yml` that match the paper environment (Ubuntu 20.04, Python 3.8, CUDA 11.3, PyTorch 1.11) and also install Blender for the provided demo scripts.

### Requirements
- Docker
- NVIDIA Container Toolkit (Linux) to run the demos with GPU (`chois-gpu` service). On macOS/Windows you can still build the image and run prereq checks with the `chois` service, but the model itself requires CUDA.

### Data / models (required)
You need these folders on your host machine:
- `processed_data/` (download link in Prerequisites below)
- `pretrained_models/` (download link below)
- SMPL models available under `processed_data/smpl_all_models/`:
  - `smplx/SMPLX_MALE.npz`, `smplx/SMPLX_FEMALE.npz`, `smplx/SMPLX_NEUTRAL.npz`
  - `smplh_amass/male/model.npz` (used to read the kinematic tree)

If your data/models live elsewhere, set environment variables when running compose:
- `PROCESSED_DATA_HOST=/abs/path/to/processed_data`
- `PRETRAINED_MODELS_HOST=/abs/path/to/pretrained_models`
- `SMPL_ALL_MODELS_HOST=/abs/path/to/smpl_all_models`

### Build
```
docker compose build
```

### Run the provided demos
Single-window:
```
python tools/check_demo_prereqs.py --demo single_window
docker compose run --rm chois-gpu sh scripts/test_chois_single_window.sh
```

Smoke test (no Blender rendering; first 2 sequences only):
```
docker compose run --rm chois-gpu sh scripts/test_chois_single_window.sh --compute_metrics --max_test_seqs=2
```

Long sequence in scene:
```
python tools/check_demo_prereqs.py --demo long_seq
docker compose run --rm chois-gpu sh scripts/test_chois_long_seq_in_scene.sh
```

### Prerequisites 
Please download [SMPL-X](https://smpl-x.is.tue.mpg.de/index.html).

This code expects SMPL models under a `smpl_all_models/` folder. By default it will look in:
- `./processed_data/smpl_all_models/` (recommended for the provided scripts)
- `./data/smpl_all_models/`

You can also override paths via environment variables:
- `SMPL_ALL_MODELS_DIR=/abs/path/to/smpl_all_models`
- `SMPLH_PATH=/abs/path/to/smpl_all_models/smplh_amass` (must contain `male/model.npz`)

If you would like to generate visualizations, please install [Blender](https://www.blender.org/download/).
Blender paths can be configured via environment variables (no code edits required):
- `BLENDER_PATH` (defaults to `blender`)
- `BLENDER_UTILS_ROOT_FOLDER` (defaults to `manip/vis`)
- `BLENDER_SCENE_FOLDER` (defaults to `processed_data/blender_files`)

Please download all the [data](https://drive.google.com/file/d/1ZG-9--RfUWj5oWYnvcONNuRuxaH_Zpw1/view?usp=sharing) and put ```processed_data``` to your desired location ```your_path/processed_data```.  

### Testing: Generating single-window interaction sequences for OMOMO objects.  
Please download pretrained [model](https://drive.google.com/drive/folders/1gqw3EHiEMqw1OXqH92Axoc5FJntA_E5x?usp=sharing) and put ```pretrained_models/``` to the root folder. If you'd like to test on 3D-FUTURE objects, please add ```--unseen_objects```. For quantitative evaluation please add ```--for_quant_eval```.
```
sh scripts/test_chois_single_window.sh 
```

### Testing: Generating scene-aware long sequence for OMOMO objects.  
Note that this following command will generate long sequence visualizations in an empty floor. If you'd like to visualize in a 3D scene, please check next instruction below. 
```
sh scripts/test_chois_long_seq_in_scene.sh 
```

### Run visualization for interaction sequence and 3D scenes 
Please run the command to generate long sequence first. This will save human and object meshes to .ply file. If you want to skip the visualization for long sequence in an empty floor, you can change line 2449 in ```trainer_chois.py```, set ```save_obj_only``` to True. 
```
sh scripts/test_chois_long_seq_in_scene.sh 
```
To visualize the generated sequence in the given 3D scene with Blender:
```
cd utils/vis_utils
python render_res_w_blender.py
```
No code edits are required: configure Blender via environment variables (for example `BLENDER_PATH=blender`). If you are running via `docker compose`, Blender is already installed in the image and these env vars are set in `docker-compose.yml`.

### Training 
Train CHOIS (generating object motion and human motion given text, object geometry, and initial states). Please replace ```--entity``` with your account name. Note that when you first run this script, it need to extract BPS representation for all the sequences and may take more than 1 hour to finish the data processing. It requires about 32G disk space. 
```
sh scripts/train_chois.sh
```

### Computing Evaluation Metrics (FID, R-precision)
We followed prior work on human motion generation (Generating Diverse and Natural 3D Human Motions from Text. CVPR 2022.) for evaluating human motion quality and text-motion consistency. Since the motion distribution in our paper is different from HumanML3D, we trained corresponding feature extractors on OMOMO dataset (Object Motion Guided Human Motion Synthesis. SIGGRAPH Asia 2023.). We provided trained feature extractors [here](https://drive.google.com/drive/folders/1hGDYEy91Tk7FC1U_8BouhlSp5RQkEJuY?usp=sharing). To run this evaluation, please check the code and modify corresponding paths for feature extractors and the results that you need to evaluate. 
```
cd t2m_eval/
python final_evaluations.py 
```

### Generating new data for scene-aware interaction synthesis.  
We used the following code for generating waypoints in 3D scenes. If you'd like to generate more data, please check ```create_eval_data.py```. 
```
cd utils/create_eval_data/
python create_eval_data.py 
```

### Citation
```
@inproceedings{li2023controllable,
  title={Controllable human-object interaction synthesis},
  author={Li, Jiaman and Clegg, Alexander and Mottaghi, Roozbeh and Wu, Jiajun and Puig, Xavier and Liu, C. Karen},
  booktitle={ECCV},
  year={2024}
}
```

### Related Repos
We adapted some code from other repos in data processing, learning, evaluation, etc. Please check these useful repos. 
```
https://github.com/lijiaman/omomo_release
https://github.com/lijiaman/egoego_release
https://github.com/EricGuo5513/text-to-motion 
https://github.com/lucidrains/denoising-diffusion-pytorch
https://github.com/jihoonerd/Conditional-Motion-In-Betweening 
https://github.com/lijiaman/motion_transformer 
``` 
