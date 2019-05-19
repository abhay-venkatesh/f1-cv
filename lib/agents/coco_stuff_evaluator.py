from PIL import Image
from lib.agents.agent import Agent
from lib.datasets.coco_stuff import COCOStuffEval
from math import floor
from pathlib import Path
from pycocotools import mask
from torch import nn
from torchvision import transforms
from tqdm import tqdm
import importlib
import numpy as np
import simplejson as json
import torch


class COCOStuffEvaluator(Agent):
    N_CLASSES = 92
    WINDOW_SIZE = 320

    def run(self):
        testset = COCOStuffEval(self.config["dataset path"])

        net_module = importlib.import_module(
            ("lib.models.{}".format(self.config["model"])))
        net = getattr(net_module, "build_" + self.config["model"])

        model = net(
            n_classes=self.N_CLASSES,
            size=(self.config["img width"],
                  self.config["img height"])).to(self.device)
        model = nn.DataParallel(model)
        model.load_state_dict(torch.load(Path(self.config["checkpoint path"])))

        model.eval()
        coco_result = []
        with torch.no_grad():
            for img, img_name in tqdm(testset):
                img_ = self._resize(img)
                _, h, w = img_.shape

                windows, h_overlaps, w_overlaps = self._get_windows(img_)
                X = torch.stack(windows).to(self.device)
                Y_, _ = model(X)
                _, predicted = torch.max(Y_.data, 1)
                seg = self._construct_mask(predicted, h, w)

                if len(h_overlaps) != 0:
                    X = torch.stack(h_overlaps).to(self.device)
                    Y_, _ = model(X)
                    _, predicted = torch.max(Y_.data, 1)
                    predicted = predicted.float()
                    seg = self._apply_h_overlaps(predicted, seg, h, w)

                if len(w_overlaps) != 0:
                    X = torch.stack(w_overlaps).to(self.device)
                    Y_, _ = model(X)
                    _, predicted = torch.max(Y_.data, 1)
                    predicted = predicted.float()
                    seg = self._apply_w_overlaps(predicted, seg, h, w)

                # Write segmentation as PNG output
                seg_array = seg.cpu().numpy()
                seg_array = seg_array.astype(np.uint8)
                if seg.shape != img.shape[1:]:
                    seg = Image.fromarray(seg_array)
                    seg = seg.resize((img.shape[2], img.shape[1]),
                                     Image.NEAREST)
                    seg_array = np.array(seg)

                seg_img = Image.fromarray(seg_array)
                seg_img.save(
                    Path(self.config["outputs folder"],
                         img_name.replace(".jpg", ".png")))

                anns = segmentationToCocoResult(
                    seg_array,
                    int(img_name.replace(".jpg", "")),
                    stuffStartId=0)
                coco_result.extend(anns)

        with open(
                Path(self.config["outputs folder"], "coco_result.json"),
                "w+") as f:
            json.dump(coco_result, f)

    def _resize(self, img):
        img = transforms.ToPILImage()(img)

        w, h = img.size
        if w < self.WINDOW_SIZE:
            img = img.resize((self.WINDOW_SIZE, h), Image.BILINEAR)

        w, h = img.size
        if h < self.WINDOW_SIZE:
            img = img.resize((w, self.WINDOW_SIZE), Image.BILINEAR)

        return transforms.ToTensor()(img)

    def _construct_mask(self, predicted, h, w):
        seg = torch.zeros((h, w)).float().cuda()
        num_h_fits = h / self.WINDOW_SIZE
        num_w_fits = w / self.WINDOW_SIZE
        k = 0
        for i in range(0, floor(num_h_fits)):
            for j in range(0, floor(num_w_fits)):
                h1, h2 = i * self.WINDOW_SIZE, (i + 1) * self.WINDOW_SIZE
                w1, w2 = j * self.WINDOW_SIZE, (j + 1) * self.WINDOW_SIZE
                seg[h1:h2, w1:w2] = predicted[k, :, :]
                k += 1
        return seg

    def _apply_h_overlaps(self, predicted, seg, h, w):
        num_w_fits = w / self.WINDOW_SIZE
        for j in range(0, floor(num_w_fits)):
            h1, h2 = h - self.WINDOW_SIZE, h
            w1, w2 = j * self.WINDOW_SIZE, (j + 1) * self.WINDOW_SIZE
            seg[h1:h2, w1:w2] = torch.round(
                (seg[h1:h2, w1:w2] + predicted[j, :, :]) / 2)
        return seg

    def _apply_w_overlaps(self, predicted, seg, h, w):
        num_h_fits = h / self.WINDOW_SIZE
        for i in range(0, floor(num_h_fits)):
            h1, h2 = i * self.WINDOW_SIZE, (i + 1) * self.WINDOW_SIZE
            w1, w2 = w - self.WINDOW_SIZE, w
            seg[h1:h2, w1:w2] = torch.round(
                (seg[h1:h2, w1:w2] + predicted[i, :, :]) / 2)
        return seg

    def _get_windows(self, img):
        _, h, w = img.shape
        num_h_fits = h / self.WINDOW_SIZE
        num_w_fits = w / self.WINDOW_SIZE
        windows = []
        for i in range(0, floor(num_h_fits)):
            for j in range(0, floor(num_w_fits)):
                h1, h2 = i * self.WINDOW_SIZE, (i + 1) * self.WINDOW_SIZE
                w1, w2 = j * self.WINDOW_SIZE, (j + 1) * self.WINDOW_SIZE
                windows.append(img[:, h1:h2, w1:w2])

        h_overlaps = []
        if not num_h_fits.is_integer():
            for j in range(0, floor(num_w_fits)):
                h1, h2 = h - self.WINDOW_SIZE, h
                w1, w2 = j * self.WINDOW_SIZE, (j + 1) * self.WINDOW_SIZE
                h_overlaps.append(img[:, h1:h2, w1:w2])

        w_overlaps = []
        if not num_w_fits.is_integer():
            for i in range(0, floor(num_h_fits)):
                h1, h2 = i * self.WINDOW_SIZE, (i + 1) * self.WINDOW_SIZE
                w1, w2 = w - self.WINDOW_SIZE, w
                w_overlaps.append(img[:, h1:h2, w1:w2])

        return windows, h_overlaps, w_overlaps


def segmentationToCocoResult(labelMap, imgId, stuffStartId=92):
    '''
    Convert a segmentation map to COCO stuff segmentation result format.
    :param labelMap: [h x w] segmentation map that indicates the label of each
     pixel
    :param imgId: the id of the COCO image (last part of the file name)
    :param stuffStartId: (optional) index where stuff classes start
    :return: anns    - a list of dicts for each label in this image
       .image_id     - the id of the COCO image
       .category_id  - the id of the stuff class of this annotation
       .segmentation - the RLE encoded segmentation of this class
    '''

    # Get stuff labels
    shape = labelMap.shape
    if len(shape) != 2:
        raise Exception(
            ('Error: Image has %d instead of 2 channels! Most likely you '
             'provided an RGB image instead of an indexed image (with or'
             'without color palette).') % len(shape))
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
        anndata['image_id'] = int(imgId)
        anndata['category_id'] = int(labelId)
        anndata['segmentation'] = Rs
        anns.append(anndata)
    return anns


def segmentationToCocoMask(labelMap, labelId):
    '''
    Encodes a segmentation mask using the Mask API.
    :param labelMap: [h x w] segmentation map that indicates the label of each
     pixel
    :param labelId: the label from labelMap that will be encoded
    :return: Rs - the encoded label mask for label 'labelId'
    '''
    labelMask = labelMap == labelId
    labelMask = np.expand_dims(labelMask, axis=2)
    labelMask = labelMask.astype('uint8')
    labelMask = np.asfortranarray(labelMask)
    Rs = mask.encode(labelMask)
    assert len(Rs) == 1
    Rs = Rs[0]

    return Rs