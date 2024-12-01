import streamlit as st
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
from io import BytesIO

# Sınıf isimlerini tanımlayın
CLASS_NAMES = ['Tomato___Late_blight', 'Tomato___healthy', 'Grape___healthy', 'Orange___Haunglongbing_(Citrus_greening)',
               'Soybean___healthy', 'Squash___Powdery_mildew', 'Potato___healthy', 'Corn_(maize)___Northern_Leaf_Blight',
               'Tomato___Early_blight', 'Tomato___Septoria_leaf_spot', 'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot',
               'Strawberry___Leaf_scorch', 'Peach___healthy', 'Apple___Apple_scab', 'Tomato___Tomato_Yellow_Leaf_Curl_Virus',
               'Tomato___Bacterial_spot', 'Apple___Black_rot', 'Blueberry___healthy', 'Cherry_(including_sour)___Powdery_mildew',
               'Peach___Bacterial_spot', 'Apple___Cedar_apple_rust', 'Tomato___Target_Spot', 'Pepper,_bell___healthy',
               'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)', 'Potato___Late_blight', 'Tomato___Tomato_mosaic_virus',
               'Strawberry___healthy', 'Apple___healthy', 'Grape___Black_rot', 'Potato___Early_blight',
               'Cherry_(including_sour)___healthy', 'Corn_(maize)___Common_rust_', 'Grape___Esca_(Black_Measles)',
               'Raspberry___healthy', 'Tomato___Leaf_Mold', 'Tomato___Spider_mites Two-spotted_spider_mite',
               'Pepper,_bell___Bacterial_spot', 'Corn_(maize)___healthy']
class_size = len(CLASS_NAMES)

# Cihazı ayarlayın (GPU varsa kullan, yoksa CPU kullan)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Model için gerekli sınıflar
class ImageClassificationBase(nn.Module):
    def training_step(self, batch):
        images, labels = batch
        out = self(images)
        loss = nn.functional.cross_entropy(out, labels)
        return loss
    
    def validation_step(self, batch):
        images, labels = batch
        out = self(images)
        loss = nn.functional.cross_entropy(out, labels)
        acc = accuracy(out, labels)
        return {"val_loss": loss.detach(), "val_accuracy": acc}
    
    def validation_epoch_end(self, outputs):
        batch_losses = [x["val_loss"] for x in outputs]
        batch_accuracies = [x["val_accuracy"] for x in outputs]
        epoch_loss = torch.stack(batch_losses).mean()
        epoch_accuracy = torch.stack(batch_accuracies).mean()
        return {"val_loss": epoch_loss.item(), "val_accuracy": epoch_accuracy.item()} 

def accuracy(outputs, labels):
    _, preds = torch.max(outputs, dim=1)
    return torch.tensor(torch.sum(preds == labels).item() / len(preds))

def ConvBlock(in_channels, out_channels, pool=False):
    layers = [nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
              nn.BatchNorm2d(out_channels),
              nn.ReLU(inplace=True)]
    if pool:
        layers.append(nn.MaxPool2d(4))
    return nn.Sequential(*layers)

class ResNet9(ImageClassificationBase):
    def __init__(self, in_channels, num_classes):
        super().__init__()
        self.conv1 = ConvBlock(in_channels, 64)
        self.conv2 = ConvBlock(64, 128, pool=True)
        self.res1 = nn.Sequential(ConvBlock(128, 128), ConvBlock(128, 128))
        self.conv3 = ConvBlock(128, 256, pool=True)
        self.conv4 = ConvBlock(256, 512, pool=True)
        self.res2 = nn.Sequential(ConvBlock(512, 512), ConvBlock(512, 512))
        self.classifier = nn.Sequential(
            nn.MaxPool2d(4),
            nn.Flatten(),
            nn.Linear(512, num_classes)
        )

    def forward(self, xb):
        out = self.conv1(xb)
        out = self.conv2(out)
        out = self.res1(out) + out
        out = self.conv3(out)
        out = self.conv4(out)
        out = self.res2(out) + out
        out = self.classifier(out)
        return out

# Modeli yükle
model_path = './plant-disease-model.pth'  # Sadece ağırlıklar
try:
    model = ResNet9(in_channels=3, num_classes=class_size)
    model.load_state_dict(torch.load(model_path, map_location=device))
    st.success("Model başarıyla yüklendi!")
except Exception as e:
    st.error(f"Model yüklenirken hata oluştu: {e}")
    st.stop()

model = model.to(device)
model.eval()

def preprocess_image(img):
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    img = transform(img).unsqueeze(0)
    return img

def predict_image(img):
    img = preprocess_image(img).to(device)
    with torch.no_grad():
        outputs = model(img)
    probabilities = torch.nn.functional.softmax(outputs, dim=1)
    confidence, predicted = torch.max(probabilities, 1)
    return predicted.cpu().numpy()[0], confidence.cpu().numpy()[0]

# Streamlit arayüzü
st.title("Bitki Hastalığı Tespit Uygulaması")
camera_input = st.camera_input('Kameradan resim çek')
gallery_input = st.file_uploader('VEYA Resim Yükle', accept_multiple_files=False)

if camera_input is not None:
    img_bytes = camera_input.getvalue()
    img = Image.open(BytesIO(img_bytes))
    img_cv2 = np.array(img)

    predicted_class, confidence = predict_image(img_cv2)
    st.write(f"Tahmin Edilen Sınıf: {CLASS_NAMES[predicted_class]}")
    st.write(f"İnanılırlık Yüzdesi: {confidence*100:.2f}%")

elif gallery_input is not None:
    img_bytes = gallery_input.getvalue()
    img = Image.open(BytesIO(img_bytes))
    img_cv2 = np.array(img)

    predicted_class, confidence = predict_image(img_cv2)
    st.write(f"Tahmin Edilen Sınıf: {CLASS_NAMES[predicted_class]}")
    st.write(f"İnanılırlık Yüzdesi: {confidence*100:.2f}%")

else:
    st.write("Lütfen bir resim yükleyin veya kamera kullanarak bir resim çekin.")
