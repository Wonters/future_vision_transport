import numpy as np
import torch
import pickle
import base64
import warnings
from pathlib import Path
import torch.nn.functional as F
from torchmetrics import Precision, Recall
from fastapi.testclient import TestClient
from PIL import Image
from src.wrapper import SegmentedVgg16Wrapper, SegmentedDilatednetWrapper, SegmentedUnetWrapper
from src.utils import degrade_png_quality, group_mask, group_mask_v2, iou_score, CATEGORIES_MASK, compute_iou
from src.api import app


warnings.filterwarnings("ignore", category=UserWarning)

IMAGE = Image.open(Path(__file__).parent/'data/zurich_000000_000019_leftImg8bit.png')
MASK = Image.open(Path(__file__).parent/'data/zurich_000000_000019_gtFine_labelIds.png')

def test_degrade_png_quality():
    degraded_image = degrade_png_quality(IMAGE, quality=5)
    assert degraded_image.mode == 'RGB'


def test_degrade_png_quality_2():
    degraded_image = degrade_png_quality(MASK, quality=5, mode="L")
    assert degraded_image.mode == 'L'


def test_group_mask():
    """
    Test the mask if it contains human informations
    :return:
    """
    mask = torch.from_numpy(np.array(MASK))
    mask = mask.unsqueeze(0)
    mask_ = group_mask(mask)
    for cat_num, cat_value in enumerate(CATEGORIES_MASK.values()):
        assert np.array_equal(mask_.numpy()[0, ..., cat_num], np.isin(np.array(MASK), cat_value))


def test_group_mask_2():
    """
    Test show the mask on the original image
    :return:
    """
    mask = torch.from_numpy(np.array(MASK))
    mask = mask.unsqueeze(0)
    mask_ = group_mask_v2(mask, with_cat_number=True)
    model = SegmentedVgg16Wrapper(x_data=[IMAGE.filename], y_data=[MASK.filename], degradation=None)
    model.visualize(image=IMAGE, mask=mask_[0, ...].numpy())
    assert model.visualize(image=IMAGE, mask=mask_[0, ...]).size == (2048, 1024)


def test_group_mask_3():
    """
    Test the mask if it contains human informations
    :return:
    """
    mask = torch.from_numpy(np.array(MASK))
    mask = mask.unsqueeze(0)
    mask_ = group_mask_v2(mask)
    for cat_num, cat_value in enumerate(CATEGORIES_MASK.values()):
        assert np.array_equal(mask_.numpy()[0, ..., cat_num], np.isin(np.array(MASK), cat_value))

def test_precision_recall():
    prec = Precision(task='multiclass', num_classes=8, average='macro')
    recal = Recall(task='multiclass', num_classes=8, average='macro')
    mask = torch.from_numpy(np.array(MASK))
    mask = mask.unsqueeze(0)
    mask_ = group_mask_v2(mask)
    mock_pred = mask_.clone()
    mock_pred[0, :int(mask_.shape[1] / 2), :, :] = 0
    mask_ = mask_.permute(0, 3, 1, 2)
    mock_pred = mock_pred.permute(0, 3, 1, 2)
    mask_ = mask_.to(torch.float).cpu()
    mock_pred = mock_pred.to(torch.float).cpu()
    pred = torch.argmax(mock_pred, dim=1)
    target = torch.argmax(mask_, dim=1)
    assert round(prec(pred, target).item(),4) == 0.1400
    assert round(recal(pred, target).item(),4) == 0.2385

def test_iou_score():
    """
    Testing iou score equal to 1
    """
    mask1 = torch.from_numpy(np.array(MASK))
    mask1 = mask1.unsqueeze(0)
    mask1 = group_mask_v2(mask1)
    iou = iou_score(mask1, mask1)
    assert iou.item() == 1


def test_iou_score_2():
    """
    Testing iou score equal to 1 per class
    """
    mask1 = torch.from_numpy(np.array(MASK))
    mask1 = mask1.unsqueeze(0)
    iou = {}
    for index, (name, values) in enumerate(CATEGORIES_MASK.items()):
        mask = mask1[mask1 == index]
        iou[name] = iou_score(mask, mask)
        assert iou[name].item() == 1


def test_iou_score_3():
    """
    Testing the iou_score per class changing a part of the original mask
    """
    mask1 = torch.from_numpy(np.array(MASK))
    mask1 = mask1.unsqueeze(0)
    mask = mask1.clone()
    mask[..., int(mask.shape[1] / 2):, int(mask.shape[2] / 2):] = 0
    iou = {}
    for index, (name, values) in enumerate(CATEGORIES_MASK.items()):
        target = mask1 == index
        pred = mask == index
        iou[name] = np.round(iou_score(pred, target).item(), 5)
    assert iou == {'void': 6e-05, 'flat': 0.32005,
                   'construction': 1.0, 'object': 0.75801,
                   'nature': 1.0, 'sky': 1.0,
                   'human': 0.3488, 'vehicle': 0.54515}

def test_iou_score_4():
    """
    Testing the compute iou as train algo use it
    Test the iou per class mocking a prediction mask replacing the half bottom of the mask per 0
    Keeping a part of void and flat
    """
    mask = torch.from_numpy(np.array(MASK))
    mask = mask.unsqueeze(0)
    mask_ = group_mask_v2(mask)
    mock_pred = mask_.clone()
    mock_pred[0, :int(mask_.shape[1] / 2), :, :] = 0
    iou = compute_iou(mock_pred.to(torch.float).permute(0,3,1,2),
                      mask_.to(torch.float).permute(0,3,1,2),
                      mean=False,
                      softmax=False)
    iou = {k: np.round(v.item(), 5) for k, v in iou.items()}
    assert iou =={'void': 0.12023, 'flat': 0.90839, 'construction': 0.0,
                  'object': 0.0, 'nature': 0.0, 'sky': 0.0, 'human': 0.0,
                  'vehicle': 0.0}

def test_fastapi_server():
    """
    Test the api server for a prediction
    """
    client = TestClient(app)
    rep = client.post("/predict", json={'pickle_data': base64.b64encode(pickle.dumps(IMAGE)).decode('utf-8')})
    assert isinstance(pickle.loads(base64.b64decode(rep.text)), Image.Image)


class BaseWrapperTest:
    model_class = SegmentedVgg16Wrapper

    @classmethod
    def setup_class(cls):
        """"""
        cls.model = cls.model_class(x_data=[IMAGE.filename], y_data=[MASK.filename], degradation=None, frac=1)
        cls.model.epochs = 1


class TestSegmentedVgg16Wrapper(BaseWrapperTest):

    def test_dataset(self):
        """"""
        for image, mask, idx in self.model.dataloader:
            image = F.interpolate(image, mask.shape[2:], mode='bilinear', align_corners=False)
            image = image.permute(0, 2, 3, 1).detach().cpu().numpy()[0, ...]
            mask = mask.permute(0, 2, 3, 1)
            arg_index = torch.argmax(mask.cpu(), dim=-1)
            image = Image.fromarray(image.astype(np.uint8), mode='RGB')
            self.model.visualize(image, arg_index.detach().cpu().numpy()[0, ...], logits=False)
            break

    def test_train(self):
        """"""
        self.model.train()
        assert Path(self.model.checkpoint_path).is_file()

    def _test_features(self):
        print(self.model.model.encoder)

    def test_predict(self):
        """"""
        output = self.model.predict([IMAGE])
        output = output.detach().cpu().numpy()
        assert output.shape == (1, 1024, 2048, 8)
        self.model.visualize(IMAGE, output[0, ...])


class TestSegmentedUnetWrapper(BaseWrapperTest):
    model_class = SegmentedDilatednetWrapper

    def test_train(self):
        """"""
        self.model.train()


class TestUnetWrapper(BaseWrapperTest):
    model_class = SegmentedUnetWrapper

    def test_train(self):
        """"""
        self.model.train()
