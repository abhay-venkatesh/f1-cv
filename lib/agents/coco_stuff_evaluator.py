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
import simplejson as json
import slidingwindow
import torch


class COCOStuffEvaluator(Agent):
    N_CLASSES = 92
    WINDOW_SIZE = 320
    WINDOW_OVERLAP_PERCENT = 0.5

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
            # for img, img_name in tqdm(testset):
            for img, img_name in tqdm([testset[1]]):
                """ testing_images: [0, 1] """
                img_ = self._resize(img)

                img_windows, windows = self._get_img_windows(img_)

                X = torch.stack(img_windows).to(self.device)
                Y_, _ = model(X)
                _, predicted = torch.max(Y_.data, 1)

                seg = self._get_seg(predicted, windows, img_)
                self._display_tensor(seg)

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

                raise RuntimeError

        with open(
                Path(self.config["outputs folder"], "coco_result.json"),
                "w+") as f:
            json.dump(coco_result, f)

    def _get_img_windows(self, img):
        img = np.array(img)
        windows = slidingwindow.generate(
            img, slidingwindow.DimOrder.ChannelHeightWidth, self.WINDOW_SIZE,
            self.WINDOW_OVERLAP_PERCENT)

        img_windows = []
        for window in windows:
            rect = (window.x, window.y, window.w, window.h)
            square = slidingwindow.fitToSize(
                rect, self.WINDOW_SIZE, self.WINDOW_SIZE,
                (0, 0, img.shape[2], img.shape[1]))
            window = slidingwindow.SlidingWindow(
                square[0], square[1], square[2], square[3],
                slidingwindow.DimOrder.ChannelHeightWidth)
            img_windows.append(torch.tensor(img[window.indices()]))
        return img_windows, windows

    def _get_seg(self, predicted, windows, img):
        _, h, w = img.shape
        n_predictions, _, _ = predicted.shape
        seg = torch.zeros((n_predictions, h, w)).float().cuda()
        for i, window in enumerate(windows):
            indice = (slice(i, i+1), window.indices()[1], window.indices()[2])
            seg[indice] = predicted[i, :, :]
            self._display_tensor(seg[indice])
            raise RuntimeError

        seg = seg.cpu().numpy()
        # torch.mode() not working
        seg, _ = mode(seg, axis=0)
        seg = np.squeeze(seg, axis=0)
        return torch.tensor(seg)

    def _display_tensor(self, img_tensor):
        img_tensor = img_tensor.cpu()
        img = transforms.ToPILImage()(img_tensor)
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