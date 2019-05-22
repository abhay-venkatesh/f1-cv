from pathlib import Path
from PIL import Image
import numpy as np

if __name__ == "__main__":
    gt = np.array(Image.open(Path("000000053626_gt.png")))
    print(np.unique(gt))
    raise RuntimeError
    gt_colored = np.zeros((3, gt.shape[0], gt.shape[1]))
    output = Image.open(Path("000000053626.png"))
    output_colored = np.zeros((3, output.shape[0], output.shape[1]))