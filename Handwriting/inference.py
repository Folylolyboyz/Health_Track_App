import onnxruntime as rt
import numpy as np
from PIL import Image
from itertools import groupby
from spellchecker import SpellChecker

class InferenceModel:
    def __init__(self, path = "Handwriting/handwritingModel.onnx"):
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"] if rt.get_device() == "GPU" else ["CPUExecutionProvider"]
        self.model = rt.InferenceSession(path, providers = providers)
        
        self.input_shape = [meta.shape for meta in self.model.get_inputs()]
        self.input_name = self.model.get_inputs()[0].name
        # print(self.input_name)
        self.output_name = self.model.get_outputs() #[meta.name for meta in self.model._outputs_meta]
        
        self.metadata = self.model.get_modelmeta().custom_metadata_map
        # print(self.metadata)
    
    def predict(self, img):
        converted_img = self.imageConvert(img)
        del img
        
        preds = self.model.run(None , {self.input_name: converted_img})[0]
        text = self.ctc_decoder(preds, self.metadata["vocab"])[0]
        
        text = self.spellcheck(text)
        del preds, self.model, self.input_shape, self.input_name, self.output_name, self.metadata
        return text
    
    def imageConvert(self, img) -> np.ndarray:
        img = img.resize(self.input_shape[0][1:3][::-1])
        
        # img has to be sent as Image.open().convert("RGB")
        img = np.array(img)
        img = img[:, :, ::-1]   # Convert from RGB to BGR
        
        img = np.expand_dims(img, axis=0).astype(np.float32)
        # print(type(img))
        return img
    
    def ctc_decoder(self, preds: np.ndarray, vocab: list) -> list:
        argmax_preds = np.argmax(preds, axis=-1)
        grouped_preds = [[k for k,_ in groupby(preds)] for preds in argmax_preds]
        text = ["".join([vocab[k] for k in group if k < len(vocab)]) for group in grouped_preds]
        return text
    
    def spellcheck(self, text: str):
        spell = SpellChecker()
        text = spell.correction(text)
        del spell
        return text
    
# Uncomment to test

model = InferenceModel()

# Word
# img = Image.open("Handwriting/Dataset/b01-000-00-05.png").convert("RGB")
# print(model.predict(img))

# Whole directory
import os
for i in os.listdir("Handwriting/Dataset/"):
    model = InferenceModel()
    print(f"Handwriting/Dataset/{i}")
    img = Image.open(f"Handwriting/Dataset/{i}").convert("RGB")
    print(model.predict(img))