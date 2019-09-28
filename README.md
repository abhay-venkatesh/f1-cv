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

### Replicating an Experiment

Find the experiment you are interested in under `experiments/archive`, and then just read the config `.yml` file to replicate. For example, consider `experiments/archive/coco_stuff/coco_stuff_full/coco_stuff_f1_deep_lab_amazon/coco_stuff_f1_deep_lab_amazon.yml`:
```yaml
agent: COCOStuffF1Trainer

# Dataset
dataset path: /home/ubuntu/data/filtered_datasets/full_stuff_amazon

...
```
This experiment uses the `COCOStuffF1Trainer` in file `coco_stuff_f1_trainer.py` with the `full_stuff_amazon` dataset.
You can generate the dataset from https://github.com/abhay-venkatesh/coco-stuff-tools.

You can directly use the `coco_stuff_f1_deep_lab_amazon.yml` file as an argument into `main.py` to replicate.
