import numpy as np
from PIL import Image
from glob import glob
from src.train import SegmentedVgg16Wrapper
from src.utils import degrade_png_quality, group_masked


def test_degrade_png_quality():
    image = Image.open('zurich/annotat/zurich_000000_000019_leftImg8bit.png')
    degraded_image = degrade_png_quality(image, quality=5)
    assert degraded_image.mode == 'RGB'

def test_degrade_png_quality_2():
    image = Image.open('zurich/segmented/zurich_000000_000019_gtFine_labelIds.png')
    degraded_image = degrade_png_quality(image, quality=5, mode="L")
    assert degraded_image.mode == 'L'

def test_group_mask():
    img = Image.open("zurich/annotat/zurich_000069_000019_leftImg8bit.png")
    mask = Image.open("zurich/segmented/zurich_000069_000019_gtFine_labelIds.png")
    mask = np.array(mask)
    print(mask.shape)
    np.unique(mask)
    print(mask[(mask == 25) | (mask == 24)].shape)
    mask_ = group_masked(mask)
    print(mask_[(mask_ == 25) | (mask_ == 24)].shape)


class TestWrapper:
    @classmethod
    def setup_class(cls):
        """"""
        images = glob('zurich/annotat/*.png')
        masks = glob('zurich/segmented/*labelIds.png')
        cls.model = SegmentedVgg16Wrapper(x_data=images, y_data=masks, degradation=5)

    def test_dataset(self):
        """"""
        for image, mask in self.model.dataloader:
            masks = mask.cpu().numpy()
            print(masks.shape, np.where(masks > 0))
            print(mask[mask==7].shape)
            break

    def test_train(self):
        """"""
        self.model.train()

    def test_predict(self):
        """"""
        output = self.model.predict(['zurich/annotat/zurich_000000_000019_leftImg8bit.png'])
        output = output.detach().cpu().numpy()
        assert output.shape == (1, 1024, 2048, 8)

        Image.fromarray(output[0]).show()
