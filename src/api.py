import base64
import logging
import pickle
from PIL.PngImagePlugin import PngImageFile
from pydantic import BaseModel
from fastapi import FastAPI, Request
from src.utils import load_module
from src.config import WRAPPER_CONFIG, WRAPPER_CLASS

wrapper = load_module('src.wrapper',WRAPPER_CLASS)(**WRAPPER_CONFIG)

logger = logging.getLogger(__name__)
app = FastAPI()
@app.get("/")
def readme():
    return "ok"

class ImageModel(BaseModel):
    """
    Schema of the input image
    """
    pickle_data: str

    def decode(self) -> PngImageFile:
        """
        Unpickle the image to get a PIL Image object
        :return:
        """
        return pickle.loads(base64.b64decode(self.pickle_data.encode()))

@app.post("/predict")
async def predict(request: Request, image: ImageModel):
    """
    Receive a pickled image and compute a segmentation
    :param request:
    :param image:
    :return:
    """
    logger.info("Launch prediction")
    pil_image = image.decode()
    output = wrapper.predict([pil_image])
    return base64.b64encode(pickle.dumps(wrapper.visualize(pil_image, output.cpu().detach().numpy()[0,...])))



