import bentoml
import numpy as np
from bentoml.io import NumpyNdarray

# runner = bentoml.sklearn.get("mlzoomcamp_homework:qtzdz3slg6mwwdu5").to_runner()
runner = bentoml.sklearn.get("mlzoomcamp_homework:jsi67fslz6txydu5").to_runner()

service = bentoml.Service("homework", runners=[runner])

@service.api(input=NumpyNdarray(), output=NumpyNdarray())
def predict(input: np.ndarray):
    result = runner.predict.run(input)
    return result