import numpy as np
import torch
import pickle
import base64
from pathlib import Path
from fastapi.testclient import TestClient
from PIL import Image
from src.wrapper import SegmentedVgg16Wrapper, SegmentedDilatednetWrapper, SegmentedUnetWrapper
from src.utils import degrade_png_quality, group_masked, iou_score, CATEGORIES_MASK
from src.api import app

IMAGE = Image.open('zurich/annotat/zurich_000000_000019_leftImg8bit.png')
MASK = Image.open('zurich/segmented/zurich_000000_000019_gtFine_labelIds.png')


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
    mask_ = group_masked(mask)
    for cat_num, cat_value in enumerate(CATEGORIES_MASK.values()):
        assert np.array_equal(mask_ == cat_num, np.isin(np.array(MASK), cat_value))

def test_group_mask_2():
    """
    Test show the mask on the original image
    :return:
    """
    mask = torch.from_numpy(np.array(MASK))
    mask = mask.unsqueeze(0)
    mask_ = group_masked(mask, with_cat_number=True)
    model = SegmentedVgg16Wrapper()
    #model.visualize(image=IMAGE, mask=mask_).show()
    assert model.visualize(image=IMAGE, mask=mask_).size == (2048, 1024)

def test_iou_score():
    mask1 = torch.from_numpy(np.array(MASK))
    mask1 = mask1.unsqueeze(0)
    iou = iou_score(mask1, mask1)
    assert iou.item() == 1

def test_fastapi_server():
    client = TestClient(app)
    rep = client.post("/predict", json={'pickle_data':base64.b64encode(pickle.dumps(IMAGE)).decode('utf-8')})
    assert isinstance(pickle.loads(base64.b64decode(rep.text)), Image.Image)


class BaseWrapperTest:
    model_class = SegmentedVgg16Wrapper
    @classmethod
    def setup_class(cls):
        """"""
        cls.model = cls.model_class(x_data=[IMAGE.filename], y_data=[MASK.filename], degradation=5)

class TestSegmentedVgg16Wrapper(BaseWrapperTest):

    def test_dataset(self):
        """"""
        for image, mask in self.model.dataloader:
            break

    def test_train(self):
        """"""
        self.model.train()
        assert Path(self.model.checkpoint_path).is_file()

    def test_features(self):
        print(self.model.model.encoder)

    def test_predict(self):
        """"""
        output = self.model.predict([IMAGE])
        output = output.detach().cpu().numpy()
        assert output.shape == (1, 1024, 2048, 8)
        self.model.visualize(IMAGE, output[0,...]).show()

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