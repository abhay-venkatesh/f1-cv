# f1-cv
F1 normalization with applications in computer vision.

## Installation

### Windows

```bash
conda env create -f environment.yml
conda activate f1-cv
pip install "git+https://github.com/philferriere/cocoapi.git#egg=pycocotools&subdirectory=PythonAPI"
pip install slidingwindow
```

### Ubuntu

```bash
conda env create -f environment.yml
conda activate f1-cv
pip install pycocotools
pip install slidingwindow
```

## Running

```bash
(f1-cv) $ mv configs/_deprecated/full_coco_stuff/coco_stuff_official_eval.yml configs/
(f1-cv) $ python main.py configs/coco_stuff_official_eval.yml
```
