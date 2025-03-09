# Audio Copy-Move Forgery Detection

## Overview
This project detects audio copy-move forgeries, where a segment of an audio file is copied and pasted elsewhere within the same file to alter its content. It transforms audio into spectrograms, extracts keypoints, identifies high-frequency ranges, generates graph images using advanced graph techniques, and classifies them as forged or genuine using a Convolutional Neural Network (CNN).

## Features
- Converts audio to super-resolution spectrograms using Short-Time Fourier Transform (STFT).
- Extracts keypoints with SIFT and identifies high-frequency ranges.
- Applies bandpass filtering and spiral pattern extraction.
- Constructs visibility graphs and converts them to images.
- Uses a fine-tuned MobileNetV2 CNN for forgery classification.

## Requirements
- Python 3.8+
- Libraries: `tensorflow`, `librosa`, `opencv-python`, `networkx`, `scipy`, `numpy`, `pandas`, `joblib`, `community`, `tqdm`, `matplotlib`, `scikit-learn`

Usage
- Clone the repository
```bash
git clone https://github.com/pavank-v/Audio-Forgery-Detection.git
```
- Create a new python environment
```bash
python3 -m venv env
# Activate the environment
source env/bin/activate
```
- CD to Backend
```bash
cd Backend
```
- Start the Project
```bash
python manage.py runserver
```
- Preview
![audio_forgery_project](https://github.com/user-attachments/assets/5bee94e9-c21a-44fa-b7db-0fcc324e323a)

File Structure
```bash
.
├── Backend
│   ├── api
│   │   ├── admin.py
│   │   ├── apps.py
│   │   ├── __init__.py
│   │   ├── models.py
│   │   ├── templates
│   │   │   └── api
│   │   │       └── index.html
│   │   ├── tests.py
│   │   ├── urls.py
│   │   └── views.py
│   ├── Backend
│   │   ├── asgi.py
│   │   ├── __init__.py
│   │   ├── settings.py
│   │   ├── urls.py
│   │   └── wsgi.py
│   ├── manage.py
│   └── static
├── notebooks
│   ├── ACM_MODEL.ipynb
│   └── audio_forgery_detection_model.joblib
├── README.md
└── requirements.txt
```

## Contributing

Contributions are welcome! Please fork the repository and submit pull requests. For major changes, open an issue to discuss potential improvements.  

### How to Contribute
1. Fork the repository.  
2. Create a new branch (`git checkout -b feature-branch`).  
3. Make your changes and commit them (`git commit -m "Improved the model accuracy"`).  
5. Push to the branch (`git push origin feature-branch`).  
6. Open a pull request.  

---

## Acknowledgments

- **Libraries Used**: This project leverages several libraries such as TensorFlow, Pandas, Librosa, OpenCV, and more.  
- **Inspiration**: Special thanks to the open-source community for providing valuable resources and documentation.  
