🩺 Ultrasound Image Classification using CNN Models

This repository contains a deep learning-based image classification project using ultrasound images collected from a Kaggle dataset (link to dataset can be added). The goal is to classify ultrasound images using various CNN architectures and compare their performance.

📁 Dataset

The dataset contains labeled ultrasound images used for training and evaluating multiple deep learning models. It has been downloaded from Kaggle.

Format: .jpg or .png
Labels: [Insert labels like "Normal", "Abnormal", etc.]
Total Samples: [Insert total number]
Train/Test Split: [e.g., 80/20 or K-fold]
🧠 Models Used

The following convolutional neural network (CNN) architectures were trained and tested on the dataset:

Model	Accuracy
🏆 Ensemble	0.9240
ShuffleNet v2 x1.0	0.9146
MobileNet v2	0.8829
EfficientNet B0	0.8766
DenseNet121	0.8766
ResNet18	0.8386
ResNet50	0.7943
SqueezeNet1.0	0.7595
ConvNeXt Tiny	0.5665
📊 Results Summary

The ensemble model achieved the highest accuracy of 92.40%, outperforming individual models like ShuffleNet and MobileNet. Lightweight models like ShuffleNet v2 and MobileNet v2 performed surprisingly well, indicating their suitability for mobile or edge devices.

⚙️ Installation

cd ultrasound-cnn-classification
pip install -r requirements.txt
🚀 How to Run

python train.py --model mobilenet_v2 --epochs 25 --batch_size 32
To evaluate:

python evaluate.py --model mobilenet_v2
For ensemble:

python ensemble.py
🧾 Folder Structure

📂ultrasound-cnn-classification
 ┣ 📁data/
 ┃ ┣ 📁train/
 ┃ ┣ 📁test/
 ┃ ┗ 📁val/
 ┣ 📁models/
 ┣ 📁outputs/
 ┣ 📄train.py
 ┣ 📄evaluate.py
 ┣ 📄ensemble.py
 ┣ 📄utils.py
 ┣ 📄requirements.txt
 ┗ 📄README.md
🛠️ Features

Preprocessing: Data augmentation, normalization
Training loop with early stopping and learning rate scheduler
Evaluation metrics: Accuracy, Confusion Matrix
Visualization: Training curves, Predictions
📉 Visualization Example

Add sample charts like:

Accuracy & loss over epochs
Confusion matrix
Sample predictions
🔮 Future Work

Model optimization for deployment on mobile
Real-time inference using webcam or ultrasound feed
Web app interface using Streamlit or Flask
📚 References

Kaggle Dataset
PyTorch Documentation
Papers: ShuffleNet, MobileNet, EfficientNet
benogn ultrasound images
malignant ultrasound images
normal ultrasound images


🤝 Contributing

Pull requests are welcome. For major changes, please open an issue first to discuss.

🧑‍💻 Author

Md. Mostofa Nayon
CSE Student | Deep Learning Enthusiast | Ultrasound AI Projects
📧 [mostofanayon2001@gmail.com]

