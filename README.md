# Egyptian National ID Extractor

An AI-powered system for extracting structured information from Egyptian National ID cards using object detection, OCR, and image processing techniques.

The system detects the ID card, corrects its orientation and skew, extracts the 14-digit National ID number, validates it, and decodes the user's birth date, governorate, and gender.

---

## Demo

Try the application online:

**Streamlit App:**  
[Egyptian-Id-Extractor](https://egyptian-id-extractor-hayoma.streamlit.app/)

---

## Features

- Card detection using YOLO
- Automatic orientation correction
- Skew correction using OpenCV
- National ID region detection
- OCR using a CRNN model
- Egyptian National ID validation
- Decode:
  - Birth Date
  - Governorate
  - Gender
- Interactive Streamlit interface

---

## Pipeline

```text
Input Image
      │
      ▼
Card Detection (YOLO)
      │
      ▼
Orientation Correction
      │
      ▼
Skew Correction
      │
      ▼
National ID Detection
      │
      ▼
CRNN OCR
      │
      ▼
Validation & Decoding
      │
      ▼
Display Results
```

---

## Tech Stack

- Python
- PyTorch
- Ultralytics YOLO
- OpenCV
- Streamlit
- NumPy

---

## Project Structure

```text
NID-Extractor/
│
├── app.py
├── utils.py
├── info.py
├── requirements.txt
├── README.md
└── Models/
    ├── card_detector.pt
    ├── nid_detector.pt
    └── best_crnn.pth
```

---

## Installation

Install the required packages:

```bash
pip install -r requirements.txt
```

Create a `Models` folder and place the model files inside it.

### Model Files

The trained model weights are not included in this repository because of GitHub file size limits.

Download them from:

- **CRNN Model:** [Drive](https://drive.google.com/file/d/1qpxysgUfDOstEo1ZSpp5EGyTZaTMMUBB/view?usp=sharing)
- **YOLO Models:** (or another Drive link if needed)

After downloading, place them inside:

```text
Models/
├── card_detector.pt
├── nid_detector.pt
└── best_crnn.pth
```

---

## Run

```bash
streamlit run app.py
```

---

## Performance

- High-accuracy card detection
- Reliable National ID localization
- CRNN OCR Accuracy: **≈98%** on the evaluation dataset

