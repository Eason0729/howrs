YuNet Usage

```python
import cv2
import onnxruntime as ort
import numpy as np

# Paths
MODEL_PATH = "face_detection_yunet_2023mar.onnx"  # from opencv_zoo repo [web:8][web:15]
IMAGE_PATH = "test.jpg"

# Load image
img = cv2.imread(IMAGE_PATH)
h, w = img.shape[:2]

# YuNet default input size in OpenCV examples is 320x320 [web:5]
inp_w, inp_h = 320, 320

# Preprocess: resize, BGR->RGB, normalize to [0,1], NCHW
img_resized = cv2.resize(img, (inp_w, inp_h))
blob = img_resized[:, :, ::1].astype(np.float32) / 255.0  # keep BGR if model trained that way
blob = np.transpose(blob, (2, 0, 1))        # HWC -> CHW
blob = np.expand_dims(blob, axis=0)         # NCHW

# Load ONNX model with onnxruntime
sess = ort.InferenceSession(MODEL_PATH, providers=["CPUExecutionProvider"])
input_name = sess.get_inputs()[0].name
outputs = sess.run(None, {input_name: blob})

# YuNet ONNX from OpenCV Zoo outputs bounding boxes and scores/landmarks
# Typical output layout per detection:
# [x, y, w, h, score, l0x, l0y, l1x, l1y, l2x, l2y, l3x, l3y, l4x, l4y] [web:6][web:8]
pred = outputs[0][0]  # shape: [N, 15]

score_thresh = 0.9
detections = pred[pred[:, 4] > score_thresh]

# Scale boxes back to original image size
scale_x = w / inp_w
scale_y = h / inp_h

for det in detections:
    x, y, bw, bh, score = det[:5]
    # bbox in resized image space -> original image
    x1 = int(x * scale_x)
    y1 = int(y * scale_y)
    x2 = int((x + bw) * scale_x)
    y2 = int((y + bh) * scale_y)

    # draw bbox
    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
    cv2.putText(img, f"{score:.2f}", (x1, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    # landmarks (5 points)
    landmarks = det[5:].reshape(-1, 2)
    for (lx, ly) in landmarks:
        lx = int(lx * scale_x)
        ly = int(ly * scale_y)
        cv2.circle(img, (lx, ly), 2, (0, 0, 255), -1)

cv2.imshow("YuNet Face Detection", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
```


SFace Usage
```python
import onnxruntime as ort
import numpy as np
import cv2
from numpy.linalg import norm

# Paths
SFACE_MODEL_PATH = "face_recognition_sface.onnx"  # e.g. from opencv/face_recognition_sface
IMG1_PATH = "person1.jpg"
IMG2_PATH = "person2.jpg"

# Create ONNX Runtime session
sess = ort.InferenceSession(SFACE_MODEL_PATH, providers=["CPUExecutionProvider"])

# Inspect model input
input_cfg = sess.get_inputs()[0]
input_name = input_cfg.name
# typical SFace input is [1, 3, 112, 112], BGR, float32
_, c, h, w = input_cfg.shape

def preprocess_face(img_bgr, target_size=(w, h)):
    # Assume img_bgr is already a cropped & aligned face
    img = cv2.resize(img_bgr, target_size)
    img = img.astype(np.float32)
    # SFace expects 0–255 with mean/std = 127.5 (OpenCV dnn style)
    mean = 127.5
    std = 127.5
    img = (img - mean) / std
    # HWC -> CHW -> add batch
    img = np.transpose(img, (2, 0, 1))
    img = np.expand_dims(img, axis=0)
    return img

def get_embedding(face_bgr):
    blob = preprocess_face(face_bgr)
    feat = sess.run(None, {input_name: blob})[0]  # shape: [1, 128] or [1, 256] depending on model
    feat = feat[0]
    # L2-normalize
    feat = feat / (norm(feat) + 1e-10)
    return feat

def cosine_similarity(f1, f2):
    return float(np.dot(f1, f2) / (norm(f1) * norm(f2) + 1e-10))

# Load example face images (already cropped/aligned)
img1 = cv2.imread(IMG1_PATH)
img2 = cv2.imread(IMG2_PATH)

emb1 = get_embedding(img1)
emb2 = get_embedding(img2)

sim = cosine_similarity(emb1, emb2)
print("Cosine similarity:", sim)

# Example threshold (you should tune this on your data)
THRESHOLD = 0.363
if sim > THRESHOLD:
    print("Same person")
else:
    print("Different people")
```
