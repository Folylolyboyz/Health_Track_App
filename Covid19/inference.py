import onnxruntime as rt
import numpy as np
from PIL import Image
import torchvision

def onnxPredictData(imagereal, path = "Covid19/ViralOrCovid.onnx") -> np.int64:
    test_transform = torchvision.transforms.Compose([
        torchvision.transforms.Resize(size=(224, 224)),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    
    
    
    class_names = ['normal', 'viral', 'covid']
    # imagereal = Image.open(image).convert("RGB")
    
    image = test_transform(imagereal)
    image = image.unsqueeze(0)
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
    return class_names[predicted_class]

# print(onnxPredictData("Covid19/Dataset/test/covid/COVID-251.png"))