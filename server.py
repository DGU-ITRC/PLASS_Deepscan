import os
import uuid
import numpy as np
from pathlib import Path
from PIL import Image
from flask import Flask, request, render_template
from feature_extractor import FeatureExtractor

app = Flask(__name__)
featureExtractor = FeatureExtractor()

def get_features():
    features = []
    img_paths = []
    for feature_path in Path("./static/feature").glob("*.npy"):
        features.append(np.load(feature_path))
        img_paths.append(Path("./static/original") / (feature_path.stem + ".jpg"))
    features = np.array(features)
    return features, img_paths

def update_features():
    global _features, _img_paths
    for img_path in sorted(Path("./static/resized").glob("*.jpg")):
        if img_path.stem + ".npy" not in os.listdir("./static/feature"): 
            feature = featureExtractor.extract(img=Image.open(img_path))
            feature_path = Path("./static/feature") / (img_path.stem + ".npy")
            np.save(feature_path, feature)
    _features, _img_paths = get_features()

def save_image(file):
    img = Image.open(file.stream).convert("RGB")
    resized_img = img.resize((256, 256))
    filename = str(uuid.uuid4()) + ".jpg"
    origianl_img_path = "static/original/" + filename
    resized_img_path = "static/resized/" + filename
    img.save(origianl_img_path)
    resized_img.save(resized_img_path)
    return resized_img, origianl_img_path

_features, _img_paths = get_features()
update_features()

@app.route('/', methods=['GET', 'POST'])
def demo():
    global _features, _img_paths
    if request.method == 'POST':
        file = request.files['upload_image']

        # Save query image
        img, filepath = save_image(file)

        # Run search
        query = featureExtractor.extract(img)

        _features, _img_paths = get_features()

        if _features.size == 0:
            update_features()
            return render_template('index.html', query_path=filepath, scores=[])

        dists = np.linalg.norm(_features-query, axis=1)  # L2 distances to features
        ids = np.argsort(dists)[:30]  # Top 30 results
        scores = [((1-float(dists[id]))*100, _img_paths[id]) for id in ids]
        
        update_features()
        return render_template('index.html',
                                query_path=filepath,
                                scores=scores)
    else:
        return render_template('index.html')
    
@app.route('/inference', methods=['POST'])
def inference():
    global _features, _img_paths

    file = request.files['upload_image']

    # Save query image
    img, filepath = save_image(file)

    # Run search
    query = featureExtractor.extract(img)

    _features, _img_paths = get_features()

    if _features.size == 0:
        update_features()
        return render_template('index.html', query_path=filepath, scores=[])

    dists = np.linalg.norm(_features-query, axis=1)  # L2 distances to features
    ids = np.argsort(dists)[:30]  # Top 30 results
    scores = [((1-float(dists[id]))*100, _img_paths[id]) for id in ids]
    
    update_features()
    return {
        "query": filepath,
        "similarity": [{"score": score[0], "path": str(score[1])} for score in scores[:3]]
    }

if __name__=="__main__":
    app.run("0.0.0.0")
