import cv2
import numpy as np
import calendar
import torch
import torch.nn as nn
from ultralytics import YOLO
from info import parse_egyptian_id, GOVERNORATES


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

card_model   = YOLO('Models/card_detector.pt')
nid_model     = YOLO('Models/nid_detector.pt')

# CRNN Model
CHARS = "0123456789"
BLANK_IDX = 10

idx_to_char = {i: c for i, c in enumerate(CHARS)}

class BidirectionalLSTM(nn.Module):
    def __init__(self, in_size, hidden, out_size):
        super().__init__()
        self.rnn = nn.LSTM(in_size, hidden, bidirectional=True, batch_first=True)
        self.fc = nn.Linear(hidden * 2, out_size)

    def forward(self, x):
        out, _ = self.rnn(x)
        return self.fc(out)


class CRNN(nn.Module):
    def __init__(self, num_classes, rnn_hidden=256, rnn_layers=2, dropout=0.3):
        super().__init__()

        def conv_bn_relu(ci, co):
            return nn.Sequential(
                nn.Conv2d(ci, co, 3, 1, 1, bias=False),
                nn.BatchNorm2d(co),
                nn.ReLU(inplace=True),
            )

        self.cnn = nn.Sequential(
            conv_bn_relu(1, 64),
            nn.MaxPool2d(2, 2),

            conv_bn_relu(64, 128),
            nn.MaxPool2d(2, 2),

            conv_bn_relu(128, 256),
            conv_bn_relu(256, 256),
            nn.MaxPool2d((2,1),(2,1)),

            conv_bn_relu(256, 512),
            conv_bn_relu(512, 512),
            nn.MaxPool2d((2,1),(2,1)),

            conv_bn_relu(512, 512),
            nn.Dropout2d(dropout),
        )

        self.pool = nn.AdaptiveAvgPool2d((1, None))

        layers = []
        for i in range(rnn_layers):
            in_size = 512 if i == 0 else rnn_hidden * 2
            out_size = num_classes if i == rnn_layers - 1 else rnn_hidden * 2
            layers.append(BidirectionalLSTM(in_size, rnn_hidden, out_size))

        self.rnn = nn.Sequential(*layers)

    def forward(self, x):
        x = self.cnn(x)
        x = self.pool(x)
        x = x.squeeze(2).permute(0, 2, 1)
        x = self.rnn(x)
        return x.permute(1, 0, 2)


crnn_model = CRNN(
    num_classes=len(CHARS) + 1,
    rnn_hidden=256,
    rnn_layers=2,
    dropout=0.3
).to(DEVICE)

checkpoint = torch.load("Models/best_crnn.pth", map_location=DEVICE)
crnn_model.load_state_dict(checkpoint["model"])
crnn_model.eval()


#detect and crop the card from the input image
def crop_card(image):
    prediction_results = card_model.predict(image, conf=0.5, iou=0.45 , verbose=False)
    prediction_result  = prediction_results[0]

    if len(prediction_result.boxes) == 0:
        return None

    boxes = prediction_result.boxes
    boxes_xyxy = boxes.xyxy.cpu().numpy()
    confs = boxes.conf.cpu().numpy()
    
    areas = (boxes_xyxy[:, 2] - boxes_xyxy[:, 0]) * (boxes_xyxy[:, 3] - boxes_xyxy[:, 1])

    scores = confs * areas
    best_idx = np.argmax(scores)

    h, w = image.shape[:2]

    x1, y1, x2, y2 = boxes_xyxy[best_idx].astype(int)

    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(w, x2)
    y2 = min(h, y2)

    image_cropped = image[y1:y2, x1:x2].copy()
    
    return image_cropped


#correct the orientation of the cropped card
def correct_orientation(image_cropped):

    small = cv2.resize(image_cropped, (128, 128))

    rotations = [
        small,
        cv2.rotate(small, cv2.ROTATE_90_CLOCKWISE),
        cv2.rotate(small, cv2.ROTATE_180),
        cv2.rotate(small, cv2.ROTATE_90_COUNTERCLOCKWISE),
    ]

    results = nid_model.predict(
        rotations,
        conf=0.4,
        verbose=False
    )

    angles = [0, 90, 180, 270]

    best_angle = 0
    best_score = 0

    for i, result in enumerate(results):

        if len(result.boxes) > 0:

            confs = result.boxes.conf.cpu().numpy()

            score = confs.sum()

            if score > best_score:
                best_score = score
                best_angle = angles[i]

    if best_angle == 90:
        return cv2.rotate(image_cropped, cv2.ROTATE_90_CLOCKWISE)

    elif best_angle == 180:
        return cv2.rotate(image_cropped, cv2.ROTATE_180)

    elif best_angle == 270:
        return cv2.rotate(image_cropped, cv2.ROTATE_90_COUNTERCLOCKWISE)

    return image_cropped


#correct the skew of the orientation corrected image
def correct_skew(image):

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    gray = cv2.GaussianBlur(gray, (3,3), 0)

    edges = cv2.Canny(gray, 50, 150)

    lines = cv2.HoughLinesP(
        edges,
        1,
        np.pi / 180,
        threshold=80,
        minLineLength=100,
        maxLineGap=10
    )

    if lines is None:
        return image

    angles = []

    for line in lines:

        x1, y1, x2, y2 = line[0]

        angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))

        if -20 < angle < 20:
            angles.append(angle)

    if len(angles) == 0:
        return image

    median_angle = np.median(angles)

    if abs(median_angle) < 1:
        return image

    h, w = image.shape[:2]

    M = cv2.getRotationMatrix2D(
        (w // 2, h // 2),
        median_angle,
        1.0
    )

    rotated = cv2.warpAffine(
        image,
        M,
        (w, h),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REPLICATE
    )

    return rotated


#detect and crop the ID region from the corrected image
def crop_id_box(image, pad=5):
    result = nid_model.predict(image, conf=0.5, verbose=False)[0]

    if len(result.boxes) == 0:
        return None

    boxes = result.boxes
    confs = boxes.conf.cpu().numpy()
    best_idx = np.argmax(confs)
    
    bbox = boxes.xyxy[best_idx].cpu().numpy().astype(int)

    x1, y1, x2, y2 = bbox
    x1, y1 = max(0, x1-pad), max(0, y1-pad)
    x2, y2 = min(image.shape[1], x2+pad), min(image.shape[0], y2+pad)
    id_cropped = image[y1:y2, x1:x2].copy()

    return id_cropped


#Function to preprocess the cropped image for OCR
def preprocess_image(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (256, 64))

    norm = resized.astype(np.float32) / 255.0
    norm = (norm - 0.5) / 0.5

    tensor = torch.tensor(norm).unsqueeze(0).unsqueeze(0)
    return tensor.to(DEVICE)


#Function to detect the national ID number using YOLO
def detect_national_id(id_cropped):
    img = preprocess_image(id_cropped)

    with torch.no_grad():
        output = crnn_model(img)

    return decode_crnn(output)


#Decoding the CRNN output to  the predicted ID number
def decode_crnn(output):

    output = output.permute(1, 0, 2)
    preds = output.argmax(2)

    pred = preds[0]

    chars = []
    prev = None

    for p in pred:

        p = p.item()

        if p != BLANK_IDX and p != prev:
            chars.append(idx_to_char[p])

        prev = p

    return "".join(chars)



def validate_egyptian_id(nid):
    parsed = parse_egyptian_id(nid)
    if parsed is None:
        return False

    if not (1 <= parsed['month'] <= 12):
        return False

    try:
        max_day = calendar.monthrange(parsed['full_year'], parsed['month'])[1]
    except ValueError:
        return False

    if not (1 <= parsed['day'] <= max_day):
        return False

    if parsed['governorate_code'] not in GOVERNORATES:
        return False

    return True


'''full pipeline from image to national ID number'''
def extract_national_id(input_img):

    img_array = np.frombuffer(input_img, np.uint8)
    image     = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

    if image is None:                       
        raise ValueError("Invalid Image")
 
    image_cropped = crop_card(image)
    
    if image_cropped is None:
       raise ValueError("Card Not Detected")

    corrected_orientation = correct_orientation(image_cropped)

    corrected_skew = correct_skew(corrected_orientation)

    id_cropped = crop_id_box(corrected_skew)

    if id_cropped is None:
       raise ValueError("ID Not Detected")

    nid = detect_national_id(id_cropped)

    if not validate_egyptian_id(nid):
        raise ValueError("Please Retake a clearer image")

    return corrected_skew, id_cropped, nid
