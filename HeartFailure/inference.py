import onnxruntime as rt
import numpy as np
import pandas

def onnxPredictData(data, path = "HeartFailure/heartfailureModel.onnx") -> list:
    sess = rt.InferenceSession(path, providers=["CPUExecutionProvider"])
    input_name = sess.get_inputs()[0].name
    if type(data) == list:
        pred_onx = sess.run(None , {input_name: np.float32(data)})[0]
        return pred_onx.tolist()
    if type(data) == pandas.core.frame.DataFrame:
        pred_onx = sess.run(None , {input_name: data.values.astype(np.float32)})[0]
        return pred_onx.tolist()

print(onnxPredictData([[52.0, 1.0, 2.0, 0.0, 0.0, 180.0, 0.0, 3.0, 2.0]])[0])