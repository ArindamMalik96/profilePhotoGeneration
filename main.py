from flask import Flask, request
import json
app = Flask(__name__)
from dist_bw_faces_multi1 import *
from caffeFaceDetection import *

@app.route("/")
def hello():
    return "Gender detection is up and running. Hello World from Flask in a uWSGI Nginx Docker container with \
     Python 3.6 (from the example template)"

@app.route('/api/detectface', methods=['GET', 'POST'])
def detect_face():
    content = request.json
    # requestUrl = "http://highresolution.photography/images/girl-face-with-freckles-main.jpg";
    requestUrl = content.get("imgUrl")
    if requestUrl:
        print(requestUrl,"\n")
        dimensionFace=mark_download(requestUrl)
        return json.dumps(dimensionFace, cls=NumpyEncoder)
    return "unsufficient data"  

@app.route('/api/detectCaffeFace', methods=['GET', 'POST'])
def detectCaffeface():
    content = request.json
    # requestUrl = "http://highresolution.photography/images/girl-face-with-freckles-main.jpg";
    requestUrl = content.get("imgUrl")
    if requestUrl:
        print(requestUrl,"\n")
        dimensionFace = getFaceFromCaffe(requestUrl)
        return json.dumps(dimensionFace, cls=NumpyEncoder)
    return "unsufficient data"  


class NumpyEncoder(json.JSONEncoder):
    """ Special json encoder for numpy types """
    def default(self, obj):
        if isinstance(obj, (np.int_, np.intc, np.intp, np.int8,
            np.int16, np.int32, np.int64, np.uint8,
            np.uint16, np.uint32, np.uint64)):
            return int(obj)
        elif isinstance(obj, (np.float_, np.float16, np.float32,
            np.float64)):
            return float(obj)
        elif isinstance(obj,(np.ndarray,)): #### This is the fix
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True, port=80)
