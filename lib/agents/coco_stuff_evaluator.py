from PIL import Image
from lib.agents.agent import Agent
from lib.datasets.coco_stuff import COCOStuffEval
from pathlib import Path
from pycocotools import mask
from scipy.stats import mode
from torch import nn
from torchvision import transforms
from tqdm import tqdm
import importlib
import numpy as np
import os
import simplejson as json
import slidingwindow
import torch


class COCOStuffEvaluator(Agent):
    N_CLASSES = 92
    WINDOW_SIZE = 320
    WINDOW_OVERLAP_PERCENT = 0.50

    def run(self):
        testset = COCOStuffEval(self.config["dataset path"])

        net_module = importlib.import_module(
            ("lib.models.{}".format(self.config["model"]))
        )
        net = getattr(net_module, "build_" + self.config["model"])

        model = net(
            n_classes=self.N_CLASSES,
            size=(self.config["img width"], self.config["img height"]),
        ).to(self.device)
        model = nn.DataParallel(model)
        model.load_state_dict(torch.load(Path(self.config["checkpoint path"])))

        # For checking if output already produced
        output_names = [
            f
            for f in os.listdir(self.config["outputs folder"])
            if os.path.isfile(Path(self.config["outputs folder"], f))
        ]

        model.eval()
        coco_result = []
        eval_start = round(
            self.config["split number"]
            * (len(testset) / self.config["eval split divisor"])
        )
        eval_end = round(
            (self.config["split number"] + 1)
            * (len(testset) / self.config["eval split divisor"])
        )
        with torch.no_grad():
            for i in tqdm(range(eval_start, eval_end)):
                img, img_name = testset[i]

                seg_name = img_name.replace(".jpg", ".png")
                if seg_name in output_names:
                    seg_img = Image.open(Path(self.config["outputs folder"], seg_name))
                    seg_array = np.array(seg_img)
                    anns = segmentationToCocoResult(
                        seg_array, int(img_name.replace(".jpg", "")), stuffStartId=0
                    )
                    coco_result.extend(anns)
                else:

                    img_ = self._resize(img)

                    img_windows, windows = self._get_img_windows(img_)

                    X = torch.stack(img_windows).to(self.device)
                    Y_, _ = model(X)
                    _, predicted = torch.max(Y_.data, 1)

                    seg_array = self._get_seg_array(predicted, windows, img_)

                    # Write segmentation as PNG output
                    seg_img = Image.fromarray(seg_array)
                    if seg_array.shape != img.shape[1:]:
                        seg_img = Image.fromarray(seg_array)
                        seg_img = seg_img.resize(
                            (img.shape[2], img.shape[1]), Image.NEAREST
                        )

                    seg_img.save(Path(self.config["outputs folder"], seg_name))

                    anns = segmentationToCocoResult(
                        seg_array, int(img_name.replace(".jpg", "")), stuffStartId=0
                    )
                    coco_result.extend(anns)

        with open(Path(self.config["outputs folder"], "coco_result.json"), "w+") as f:
            json.dump(coco_result, f)

    def _get_img_windows(self, img):
        img = np.array(img)
        windows = slidingwindow.generate(
            img,
            slidingwindow.DimOrder.ChannelHeightWidth,
            self.WINDOW_SIZE,
            self.WINDOW_OVERLAP_PERCENT,
        )

        img_windows = []
        for window in windows:
            rect = (window.x, window.y, window.w, window.h)
            square = slidingwindow.fitToSize(
                rect,
                self.WINDOW_SIZE,
                self.WINDOW_SIZE,
                (0, 0, img.shape[2], img.shape[1]),
            )
            window = slidingwindow.SlidingWindow(
                square[0],
                square[1],
                square[2],
                square[3],
                slidingwindow.DimOrder.ChannelHeightWidth,
            )
            img_windows.append(torch.tensor(img[window.indices()]))
        return img_windows, windows

    def _get_seg_array(self, predicted, windows, img):
        _, h, w = img.shape
        n_predictions, _, _ = predicted.shape
        seg = torch.full((h, w), self.N_CLASSES).float()
        pred_stack = torch.full((n_predictions, h, w), self.N_CLASSES).float().cuda()
        for i, window in enumerate(windows):
            indice = (slice(i, i + 1), window.indices()[1], window.indices()[2])
            pred_stack[indice] = predicted[i, :, :]

        if n_predictions > 1:
            # If only a single prediction is made for a pixel, take that
            # prediction
            pred_stack = pred_stack.cpu()
            twothvalue, _ = torch.kthvalue(pred_stack, 2, dim=0)
            # If the 2th smallest value is self.N_CLASSES, then only a single
            # prediction was made for that pixel
            # So we take the single prediction for that pixel
            single_predicted, _ = torch.min(pred_stack, dim=0)
            seg[twothvalue == self.N_CLASSES] = single_predicted[
                twothvalue == self.N_CLASSES
            ]
            pred_stack = pred_stack.numpy()
            seg_array = seg.numpy()
            twothvalue = twothvalue.numpy()
            # For the rest pixels, i.e. those pixels with more than one
            #  prediction, take the majority vote among those predictions
            pred_stack[pred_stack == self.N_CLASSES] = np.nan
            # torch.mode() not working
            majority_vote, _ = mode(pred_stack, axis=0, nan_policy="omit")
            majority_vote = np.squeeze(majority_vote, axis=0)
            seg_array[twothvalue != self.N_CLASSES] = majority_vote[
                twothvalue != self.N_CLASSES
            ]
        else:
            # All predictions are single predictions
            seg = torch.squeeze(pred_stack, dim=0)
            seg_array = seg.cpu().numpy()

        seg_array = seg_array.astype(np.uint8)

        return seg_array

    def _display_tensor(self, img_tensor):
        img_tensor = img_tensor.cpu()
        img = transforms.ToPILImage()(img_tensor)
        img.show()
        input("Press enter to continue...")

    def _display_np_array(self, np_array):
        img = Image.fromarray(np_array)
        img.show()
        input("Press enter to continue...")

    def _resize(self, img):
        img = transforms.ToPILImage()(img)

        w, h = img.size
        if w < self.WINDOW_SIZE:
            img = img.resize((self.WINDOW_SIZE, h), Image.BILINEAR)

        w, h = img.size
        if h < self.WINDOW_SIZE:
            img = img.resize((w, self.WINDOW_SIZE), Image.BILINEAR)

        return transforms.ToTensor()(img)


def segmentationToCocoResult(labelMap, imgId, stuffStartId=92):
    """
    Convert a segmentation map to COCO stuff segmentation result format.
    :param labelMap: [h x w] segmentation map that indicates the label of each
     pixel
    :param imgId: the id of the COCO image (last part of the file name)
    :param stuffStartId: (optional) index where stuff classes start
    :return: anns    - a list of dicts for each label in this image
       .image_id     - the id of the COCO image
       .category_id  - the id of the stuff class of this annotation
       .segmentation - the RLE encoded segmentation of this class
    """

    # Get stuff labels
    shape = labelMap.shape
    if len(shape) != 2:
        raise Exception(
            (
                "Error: Image has %d instead of 2 channels! Most likely you "
                "provided an RGB image instead of an indexed image (with or"
                "without color palette)."
            )
            % len(shape)
        )
    [h, w] = shape
    assert h > 0 and w > 0
    labelsAll = np.unique(labelMap)
    labelsStuff = [i for i in labelsAll if i >= stuffStartId]

    # Add stuff annotations
    anns = []
    for labelId in labelsStuff:

        # Create mask and encode it
        Rs = segmentationToCocoMask(labelMap, labelId)

        # Create annotation data and add it to the list
        anndata = {}
        anndata["image_id"] = int(imgId)
        anndata["category_id"] = int(labelId)
        anndata["segmentation"] = Rs
        anns.append(anndata)
    return anns


def segmentationToCocoMask(labelMap, labelId):
    """
    Encodes a segmentation mask using the Mask API.
    :param labelMap: [h x w] segmentation map that indicates the label of each
     pixel
    :param labelId: the label from labelMap that will be encoded
    :return: Rs - the encoded label mask for label 'labelId'
    """
    labelMask = labelMap == labelId
    labelMask = np.expand_dims(labelMask, axis=2)
    labelMask = labelMask.astype("uint8")
    labelMask = np.asfortranarray(labelMask)
    Rs = mask.encode(labelMask)
    assert len(Rs) == 1
    Rs = Rs[0]

    return Rs
