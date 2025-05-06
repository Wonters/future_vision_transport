import gradio as gr


# Exemple de fonction de prédiction (à remplacer par votre propre modèle)
def predict_image(image):
    # Par exemple, on retourne la taille de l'image comme "prédiction"
    return f"Taille de l'image : {image.size}"

# Création de l'interface
interface = gr.Interface(
    fn=predict_image,
    inputs=gr.Image(type="pil"),
    outputs=[gr.Image(type="pil", label="Image annotée"), gr.Textbox(label="Résultat JSON")],
    title="Prédicteur de ",
    description="Uploadez une image puis cliquez sur Predict."
)

interface.launch()