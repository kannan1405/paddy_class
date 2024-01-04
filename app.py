import gradio as gr
import skimage
from fastai.vision.all import *




learn = load_learner('model.pkl')

labels = learn.dls.vocab
def predict(image):
    # Resize the input image


    # Convert the resized image to a PIL Image
    pil_image = PILImage.create(image)

    # Make the prediction
    pred, pred_idx, probs = learn.predict(pil_image)

    return {labels[i]: float(probs[i]) for i in range(len(labels))}


title = "paddy Classifier"
examples = ['203446.jpg','203447.jpg','203448.jpg']
interpretation='default'
enable_queue=True

gr.Interface(fn=predict,inputs=gr.Image(),outputs=gr.Label(num_top_classes=10),title=title,examples=examples).launch()