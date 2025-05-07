import requests
import gradio as gr
import base64
import pickle

API_ENDPOINT = "http://127.0.0.1:8001/predict"

def post_api(image):
    """
    Send a pickled image to the api
    :param image:
    :return:
    """
    payload = pickle.dumps([image])
    rep = requests.post(API_ENDPOINT,data={"images":base64.b64encode(payload).decode('utf-8')})
    return pickle.loads(base64.b64decode(rep.text)), "Image segmentée"

# GRADIO API
interface = gr.Interface(
    fn=post_api,
    inputs=gr.Image(type="pil"),
    outputs=[gr.Image(type="pil", label="Image annotée"), gr.Textbox(label="Résultat JSON")],
    title="Prédicteur de ",
    description="Uploadez une image puis cliquez sur Predict."
)

if __name__ == "__main__":
    interface.launch()