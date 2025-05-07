import base64
import logging
import pickle
from PIL.PngImagePlugin import PngImageFile
from pydantic import BaseModel
from fastapi import FastAPI, Request
from src.wrapper import SegmentedVgg16Wrapper

logger = logging.getLogger(__name__)


app = FastAPI()
@app.get("/")
def readme():
    return "ok"

class Image(BaseModel):
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
async def predict(request: Request, image: Image):
    """
    Receive a pickled image and compute a segmentation
    :param request:
    :param image:
    :return:
    """
    logger.info("Launch prediction")
    wrapper = SegmentedVgg16Wrapper(x_data=[], y_data=[])
    output = wrapper.predict([image.decode()])
    return base64.b64decode(pickle.dumps(wrapper.visualize(image, output)))



