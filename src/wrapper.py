import logging
import os
import random
import time
import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import mlflow
from typing import List
from PIL import Image
from PIL.PngImagePlugin import PngImageFile
from torch.utils.data import DistributedSampler, DataLoader, Subset
from .dataset import DatasetVGG16
from .deeplearning import DilatedNet, SegmentedVGG16, UNet
from .utils import iou_score, CATEGORIES_MASK, LAYER_COLORS

logger = logging.getLogger(__name__)

class SegmentedModelWrapper:
    """
    Model wrapper to train and predict model with pytorch
    """
    model_class = SegmentedVGG16
    dataset_class = DatasetVGG16
    epochs: int = 5
    device: str = "mps"
    batch_size: int = 10
    shuffle: bool = True
    # For development
    mlflow_register : str = "./mlruns_dev"

    @property
    def model_params(self)->dict:
        return dict(width=2048,
                    height=1024,
                    num_classes=8)

    def __init__(self,x_data:list=None, y_data:list=None, degradation: int = 10, distributed: bool=False):
        """"""
        if os.environ.get("DEV", False):
            mlflow.set_tracking_uri(self.mlflow_register)
        self.distributed = distributed
        if self.distributed:
            dist.init_process_group("nccl")
        if dist.is_initialized():
            self.local_rank = dist.get_rank()
            torch.cuda.set_device(self.local_rank)
        self.model = self.model_class(**self.model_params)
        self.model.to(self.device)
        if x_data and y_data:
            if dist.is_initialized():
                self.dataloader, self.sampler = self.get_ddp_dataloader()
                logger.info(f"Rank {dist.get_rank()} using DDP")
            else:
                self.dataset = self.dataset_class(x_data,
                                                  y_data,
                                                  degradation=degradation)
                self.dataloader = DataLoader(self.dataset,
                                             batch_size=self.batch_size,
                                             shuffle=self.shuffle)
        self.criterion = torch.nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(),
                                          lr=1.e-5)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer,
                                                         step_size=1,
                                                         gamma=0.2)

    def sample_dataset(self, frac=0.1):
        """
        Retrieve a subsample of the original dataset
        :param frac:
        :return:
        """
        dataset_size = len(self.dataset)
        sample_size = int(frac * dataset_size)
        indices = random.sample(range(dataset_size), sample_size)
        sampled_dataset = Subset(self.dataset, indices)
        return sampled_dataset

    def parallel_model(self):
        """
        Configure the model in parallel mode for Cuda
        Allow to run on a multi GPU instance
        :return:
        """
        self.model = self.model.cuda(f"cuda:{self.local_rank}")
        self.model = nn.parallel.DistributedDataParallel(self.model,
                                                            device_ids=[self.local_rank],
                                                            output_device=self.local_rank,
                                                            find_unused_parameters=True)
    def get_ddp_dataloader(self, frac=1.0):
        """
        Create a distribute dataloader with a fraction of the original dataset
        :param frac:
        :return:
        """
        sampled_dataset = self.sample_dataset(frac=frac)
        sampler = DistributedSampler(sampled_dataset)
        dataloader = DataLoader(sampled_dataset, batch_size=self.batch_size, sampler=sampler)
        return dataloader, sampler

    @property
    def checkpoint_path(self):
        return f"{self.model.__class__.__name__}.pth"

    def train(self):
        """"""
        if not hasattr(self,'dataset'):
            raise Exception("No data had been configured, can't train. To Train the model, give data in input "
                            f"as {self.__class__.__name__}(x_data=[...], y_data=[...])")
        self.model.train()
        with mlflow.start_run():
            for epoch in range(self.epochs):
                for images, masks in self.dataloader:
                    self.optimizer.zero_grad()
                    # Output shape (B, H, W, C)
                    output = self.model(images)
                    logger.debug(f"Train shapes: pred {output.shape}, true {masks.shape}")
                    loss = self.criterion(output, masks)
                    output = output.cpu().detach()
                    masks = masks.cpu().detach()
                    output = output.permute(0,2,3,1)
                    masks = masks.permute(0,2,3,1)
                    argmax_indices = torch.argmax(output, dim=-1)
                    output_mask = torch.zeros_like(output)
                    output_mask.scatter_(dim=3,
                                         index=argmax_indices.unsqueeze(-1),
                                         src=torch.ones_like(argmax_indices,
                                         dtype=output_mask.dtype).unsqueeze(-1))
                    output_mask = output_mask.to(torch.uint8)
                    iou = iou_score(output_mask, masks.to(torch.uint8))
                    mlflow.log_metric("iou", float(iou))
                    mlflow.log_metric("loss", loss.item())
                    mlflow.log_metric("time", time.time())
                    mlflow.log_metric("epoch", epoch)
                    logger.info(f"Epoch {epoch} : Loss {loss.item()}")
                    logger.info(f"IOU {iou}")
                    loss.backward()
                    self.optimizer.step()
                self.scheduler.step(epoch)
            torch.save(self.model.state_dict(), self.checkpoint_path)

    def visualize(self,image:PngImageFile, mask: np.ndarray):
        """
        Visualize the mask plotting each categories on the original image
        :param mask: mask
        :param image: original image from the mask is predicted
        :return:
        """

        layers = {}
        # The mask dimension in output of the NN is (B,H,W,C)
        # The line transform dimension C (8 canals) in the most probable category
        mask = np.argmax(mask, axis=-1)
        for i, cat_name in enumerate(CATEGORIES_MASK):
            if any(mask[mask == i]):
                sub_mask = np.zeros_like(np.array(mask))
                sub_mask[mask==i] = 225
                layers[cat_name] = sub_mask
        # Transform in gray image object, PIL take only uint8 format
        layers = {name:Image.fromarray(layer.astype(np.uint8), mode='L') for name, layer in layers.items()}
        img = Image.new('RGB', image.size, (0,0,0))
        for name, layer in layers.items():
            img.paste(LAYER_COLORS[name], mask=layer)
        blended = Image.blend(image,img, alpha=0.4)
        return blended


    def predict(self, images: List[str|PngImageFile]):
        """"""
        state_dict = torch.load(self.checkpoint_path)
        self.model.load_state_dict(state_dict)
        self.model.eval()
        logger.info(f"Predict {images}")
        for i in range(0, len(images), self.batch_size):
            x_vgg16 = torch.stack([DatasetVGG16.transform_rgb(Image.open(img)
                                                              if isinstance(img, str)
                                                              else img)
                                   for img in images[i:i+self.batch_size]])
            x_vgg16 = x_vgg16.to(self.device)
            with torch.no_grad():
                start= time.time()
                output = self.model(x_vgg16)
                logger.info(f"Predicition done in {time.time()-start}")
        return output.permute(0,2,3,1)


class SegmentedVgg16Wrapper(SegmentedModelWrapper):
    """
    Wrapper for vgg16
    """
    model_class = SegmentedVGG16
    dataset_class = DatasetVGG16

class SegmentedDilatednetWrapper(SegmentedModelWrapper):
    """
    Wrapper for unet
    """
    model_class = DilatedNet
    dataset_class = DatasetVGG16

class SegmentedUnetWrapper(SegmentedModelWrapper):
    batch_size = 1
    model_class = UNet
    dataset_class = DatasetVGG16

    @property
    def model_params(self) ->dict:
        return dict(

        )