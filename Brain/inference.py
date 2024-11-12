import onnxruntime as rt
import numpy as np
from PIL import Image
import random

class CustomTransform:
    def __init__(self, size=(224, 224), mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], horizontal_flip=True):
        self.size = size
        self.mean = np.array(mean).reshape(3, 1, 1).astype(np.float32)
        self.std = np.array(std).reshape(3, 1, 1).astype(np.float32)
        self.horizontal_flip = horizontal_flip

    def resize(self, img):
        """Resize the image to the specified size."""
        return img.resize(self.size, Image.BILINEAR)

    def random_horizontal_flip(self, img):
        """Randomly flip the image horizontally."""
        if self.horizontal_flip and random.random() > 0.5:
            return img.transpose(Image.FLIP_LEFT_RIGHT)
        return img

    def to_numpy_array(self, img):
        """Convert a PIL Image to a numpy array in float32 format and scale pixel values to [0, 1]."""
        img_array = np.array(img).astype(np.float32) / 255.0
        return img_array.transpose((2, 0, 1))  # Change HWC to CHW

    def normalize(self, array):
        """Normalize a numpy array image with mean and standard deviation."""
        return (array - self.mean) / self.std

    def __call__(self, img):
        """Apply all transformations in sequence to the image."""
        img = self.resize(img)
        img = self.random_horizontal_flip(img)
        img_array = self.to_numpy_array(img)
        return self.normalize(img_array).astype(np.float32)

def onnxPredictData(imagereal, path = "Brain/brainModel.onnx") -> np.int64:
    test_transform = CustomTransform(horizontal_flip=False)
    
    all_classes = ["notumor", "glioma", "meningioma", "pituitary"]
    # imagereal = Image.open(image).convert("RGB")
    
    image = test_transform(imagereal)
    image = image[0]
    input_data = image.numpy()
    
    sess = rt.InferenceSession(path, providers=["CPUExecutionProvider"])
    input_name = sess.get_inputs()[0].name #input_names=['input']
    
    del imagereal, image, test_transform
    pred_onx = sess.run(None , {input_name : input_data})[0]
    
    # print(input_name)
    # print(image)
    # print(type(pred_onx))
    # print(pred_onx)
    predicted_class = np.argmax(pred_onx, axis=1)[0] # Finds the index of the highest item in the list or numpy array
    # print(predicted_class)
    # print(type(predicted_class))
    del sess
    return all_classes[predicted_class]

# print(onnxPredictData("Brain/Dataset/Testing/notumor/Te-noTr_0001.jpg"))