import onnxruntime as rt
import numpy as np
import pandas

def onnxPredictData(data, path = "Diabetes/diabetesModel.onnx") -> list:
    sess = rt.InferenceSession(path, providers=["CPUExecutionProvider"])
    input_name = sess.get_inputs()[0].name
    if type(data) == list:
        pred_onx = sess.run(None , {input_name: np.float32(data)})[0]
        return pred_onx.tolist()
    if type(data) == pandas.core.frame.DataFrame:
        pred_onx = sess.run(None , {input_name: data.values.astype(np.float32)})[0]
        return pred_onx.tolist()

print(onnxPredictData([[0,43.1,1,1,1,21.37,6.5,130]])[0])
# 0,67.0,0,1,1,27.32,6.5,200