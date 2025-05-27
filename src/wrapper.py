import logging
from tqdm import tqdm
import random
import time
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
import torch.distributed as dist
import torch.nn as nn
from torchmetrics import Precision, Recall
import mlflow
from typing import List
from PIL import Image
from PIL.PngImagePlugin import PngImageFile
from torch.utils.data import DistributedSampler, DataLoader, Subset
from .dataset import DatasetVGG16, DatasetDeepLabV3
from .deeplearning import DilatedNet, SegmentedVGG16, UNet, DeepLabV3, CrossEntropyDiceLoss
from .utils import compute_iou, CATEGORIES_MASK, LAYER_COLORS
from .config import DEVICE, DEV
import warnings

warnings.filterwarnings("ignore", category=UserWarning)
random.seed(43)

logger = logging.getLogger(__name__)


class SegmentedModelWrapper:
    """
    Model wrapper to train and predict model with pytorch
    """
    model_class = SegmentedVGG16
    dataset_class = DatasetVGG16
    epochs: int = 20
    device: str = DEVICE
    batch_size: int = 10
    shuffle: bool = True
    lr = 1.e-5
    weight_decay = 1e-2
    weights = torch.tensor([1.0000, 0.1957, 0.3598, 4.2297, 0.4897, 2.0718, 5.8432, 0.9945],
                           dtype=torch.float32,
                           device=DEVICE)
    train_head = False
    # For development
    mlflow_register: str = "./mlruns_dev"
    num_class = 8
    checkpoint_dir: Path = Path("checkpoints")
    height: int = 1024
    width: int = 2048

    @property
    def model_params(self) -> dict:
        return dict(width=self.width,
                    height=self.height,
                    num_classes=self.num_class)

    def __init__(self, x_data: list = None, y_data: list = None,
                 degradation: int = 10,
                 crop: int = None,
                 experiment_name: str = "Segmentation",
                 frac=1,
                 distributed: bool = False):
        """"""
        self.experiment_name = experiment_name
        if x_data is None and y_data is None:
            logger.warning("No data had been configured, can't train.")
        else:
            self.width, self.height = Image.open(x_data[0]).size
        if self.width > self.height and crop:
            self.width = crop
        if DEV:
            mlflow.set_tracking_uri(self.mlflow_register)
        self.distributed = distributed
        if self.distributed:
            dist.init_process_group("nccl")
        if dist.is_initialized():
            self.local_rank = dist.get_rank()
            torch.cuda.set_device(self.local_rank)
        self.model = self.model_class(**self.model_params)
        self.load_checkpoint()
        self.model.to(self.device)
        self.dataset = self.dataset_class(x_data,
                                          y_data,
                                          degradation=degradation,
                                          crop=crop)
        if x_data and y_data:
            if dist.is_initialized():
                self.dataloader, self.sampler = self.get_ddp_dataloader(frac=frac)
                logger.info(f"Rank {dist.get_rank()} using DDP")
            else:
                self.sampler = self.sample_dataset(frac=frac)
                self.dataloader = DataLoader(self.sampler,
                                             batch_size=self.batch_size,
                                             shuffle=self.shuffle)

        self.criterion = torch.nn.CrossEntropyLoss(weight=self.weights)
        # self.criterion = CrossEntropyDiceLoss(alpha=0.5, weight=self.weights, reduction='mean')
        # Training only the head.
        if self.train_head:
            for name, param in self.model.named_parameters():
                if name.startswith("vgg16."):
                    param.requires_grad = False
            params = filter(lambda p: p.requires_grad, self.model.parameters())
        else:
            params = self.model.parameters()
        self.optimizer = torch.optim.AdamW(params,
                                           lr=self.lr,
                                           weight_decay=self.weight_decay)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer,
                                                                    mode='min',
                                                                    factor=0.5,
                                                                    patience=3,
                                                                    verbose=True)
        self.prec = Precision(task='multiclass', num_classes=self.num_class, average='macro')
        self.rec = Recall(task='multiclass', num_classes=self.num_class, average='macro')

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
        Create distribute dataloader with a fraction of the original dataset
        :param frac:
        :return:
        """
        sampled_dataset = self.sample_dataset(frac=frac)
        sampler = DistributedSampler(sampled_dataset)
        dataloader = DataLoader(sampled_dataset, batch_size=self.batch_size, sampler=sampler)
        return dataloader, sampler

    @property
    def checkpoint_path(self):
        """
        Return the checkpoint directory to store different checkpoints
        """
        directory = self.checkpoint_dir / f"{self.model.__class__.__name__}"
        directory.mkdir(parents=True, exist_ok=True)
        return directory

    def _prediction(self, images):
        """
        Return results of a prediction
        :return:
        """
        return self.model(images)

    def train(self):
        """"""
        if not hasattr(self, 'dataset'):
            raise Exception("No data had been configured, can't train. To Train the model, give data in input "
                            f"as {self.__class__.__name__}(x_data=[...], y_data=[...])")
        self.model.train()
        logger.info(f"Start training on {len(self.dataset)} images")
        mlflow.set_experiment(self.experiment_name)
        experiment = mlflow.get_experiment_by_name(self.experiment_name)
        if hasattr(self.model, "aux_head"):
            logger.info("aux loss activate")
        with mlflow.start_run(experiment_id=experiment.experiment_id, run_name=self.__class__.__name__):
            mlflow.log_param('lr', self.lr)
            mlflow.log_param('weight_decay', self.weight_decay)
            for epoch in range(self.epochs):
                self.model.train()
                self.dataset.random_crop = None #random.randrange(0, 100), random.randrange(0, 512)
                self.dataset.random_rotation = random.randint(-10, 10)
                train_loss = np.array([])
                indexes = list()
                for images, masks, idx in tqdm(self.dataloader):
                    indexes.append(idx.numpy().tolist())
                    self.optimizer.zero_grad()
                    self.model.input_height, self.model.input_width = masks.shape[2:]
                    self.height, self.width = masks.shape[2:]
                    # Output shape (B, C, H, W)
                    if hasattr(self.model, "aux_head"):
                        logits, aux_logits = self._prediction(images)
                        loss = self.criterion(logits, masks)
                        aux_loss = self.criterion(aux_logits, masks)
                        train_loss = np.concatenate((train_loss, [loss.item() + 0.1 * aux_loss.item()]))
                    else:
                        try:
                            logits, _ = self._prediction(images)
                        except ValueError:
                            logits = self._prediction(images)
                        loss = self.criterion(logits, masks)
                        train_loss = np.concatenate((train_loss, [loss.item()]))
                    iou = compute_iou(logits, masks, mean=False)
                    preds = torch.argmax(torch.softmax(logits, dim=1).cpu().detach(), dim=1)
                    target = torch.argmax(masks.cpu().detach(), dim=1)
                    mlflow.log_metric("precision", self.prec(preds, target))
                    mlflow.log_metric("recall", self.rec(preds, target))
                    for name, value in iou.items():
                        mlflow.log_metric(f"iou_{name}", float(value))
                    mlflow.log_metric("loss", loss.item())
                    mlflow.log_metric("time", time.time())
                    mlflow.log_metric("epoch", epoch)
                    mlflow.log_metric('current_lr', self.optimizer.param_groups[0]['lr'])
                    logger.info(f"Epoch {epoch} : Loss {loss.item()}")
                    loss.backward()
                    self.optimizer.step()
                logger.info(f" Epoch MeanLoss {train_loss.mean()}")
                self.scheduler.step(train_loss.mean())
                tmp_file = Path(self.checkpoint_path / f'epoch{epoch}.pth')
                torch.save(self.model.state_dict(), tmp_file)
                mlflow.log_artifact(str(tmp_file), artifact_path="checkpoints")
                self.model.eval()
                with torch.no_grad():
                    image = Image.open(self.dataset.images[0])
                    #image = Image.fromarray(self.dataset.square_crop(image, left=512).numpy())
                    logger.info(self.dataset.images[0])
                    output = self.predict([image])
                    blended_img = self.visualize(image, output.cpu().detach().numpy()[0, ...])
                    mlflow.log_image(blended_img, artifact_file=f"visualisation/visualization-epoch{epoch}.png")

    def visualize(self, image: PngImageFile, mask: np.ndarray, logits: bool = True):
        """
        Visualize the mask plotting each categories on the original image
        :param mask: mask
        :param logits: if masks is in logits format
        :param image: original image from the mask is predicted
        :return:
        """
        layers = {}
        # The mask dimension in output of the NN is (B,H,W,C)
        # The line transform dimension C (8 canals) in the most probable category
        if logits:
            mask = np.argmax(mask, axis=-1)
        for i, cat_name in enumerate(CATEGORIES_MASK):
            if any(mask[mask == i]):
                sub_mask = np.zeros_like(np.array(mask))
                sub_mask[mask == i] = 225
                layers[cat_name] = sub_mask
        # Transform in gray image object, PIL take only uint8 format
        layers = {name: Image.fromarray(layer.astype(np.uint8), mode='L') for name, layer in layers.items()}
        img = Image.new('RGB', image.size, (0, 0, 0))
        logger.info(f"{img.size}, {image.size}")
        if mask.size != image.size:
            x = (image.width - mask.shape[1]) // 2
            y = (image.height - mask.shape[0]) // 2
        for name, layer in layers.items():
            img.paste(LAYER_COLORS[name], (x, y), mask=layer)
        blended = Image.blend(image, img, alpha=0.4)
        return blended

    def get_most_recent_checkpoint(self):
        return sorted(self.checkpoint_path.glob("*.pth"),
                      key=lambda f: f.stat().st_ctime,
                      reverse=True)[0]

    def load_checkpoint(self):
        try:
            state_dict = torch.load(self.get_most_recent_checkpoint(), map_location="mps")
            self.model.load_state_dict(state_dict, strict=False)
            logger.info("Checkpoint loaded")
        except IndexError:
            logger.warning(f"No checkpoint found at {self.checkpoint_path}")

    def predict(self, images: List[str | PngImageFile]):
        """"""

        self.model.eval()
        logger.info(f"Predict {images}")
        for i in range(0, len(images), self.batch_size):
            x_vgg16 = torch.stack([self.dataset.transform_rgb(Image.open(img)
                                                              if isinstance(img, str)
                                                              else img)
                                   for img in images[i:i + self.batch_size]])
            x_vgg16 = x_vgg16.to(self.device)
            with torch.no_grad():
                start = time.time()
                try:
                    output, _ = self._prediction(x_vgg16)
                except ValueError:
                    output = self._prediction(x_vgg16)
                logger.info(f"Predicition done in {time.time() - start}")
        return output.permute(0, 2, 3, 1)


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
    model_class = UNet
    dataset_class = DatasetVGG16

    @property
    def model_params(self) -> dict:
        return dict(

        )

    def _prediction(self, images):
        return super()._prediction(images)


class SegmentedDeeplabV3(SegmentedModelWrapper):
    model_class = DeepLabV3
    dataset_class = DatasetDeepLabV3

    @property
    def model_params(self) -> dict:
        return dict(

        )

    def _prediction(self, images):
        logits = self.model(images)['out']
        logits = F.interpolate(logits, (self.height, self.width), mode='bilinear', align_corners=False)
        return logits
