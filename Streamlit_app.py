import streamlit as st
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
from io import BytesIO

# Sınıf isimlerini tanımlayın
CLASS_NAMES = ["healthy", "angular_leaf_spot", "bean_rust"]
class_size = len(CLASS_NAMES)

# Cihazı ayarlayın (GPU varsa kullan, yoksa CPU kullan)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Model için gerekli sınıflar
class ImageClassificationBase(nn.Module):
    def training_step(self, batch):
        images, labels = batch
        out = self(images)                  # Generate predictions
        loss = nn.functional.cross_entropy(out, labels) # Calculate loss
        return loss
    
    def validation_step(self, batch):
        images, labels = batch
        out = self(images)                   # Generate predictions
        loss = nn.functional.cross_entropy(out, labels)  # Calculate loss
        acc = accuracy(out, labels)          # Calculate accuracy
        return {"val_loss": loss.detach(), "val_accuracy": acc}
    
    def validation_epoch_end(self, outputs):
        batch_losses = [x["val_loss"] for x in outputs]
        batch_accuracies = [x["val_accuracy"] for x in outputs]
        epoch_loss = torch.stack(batch_losses).mean()       # Combine loss  
        epoch_accuracy = torch.stack(batch_accuracies).mean()   # Combine accuracies
        return {"val_loss": epoch_loss.item(), "val_accuracy": epoch_accuracy.item()} 

    def epoch_end(self, epoch, result):
        print("Epoch [{}], train_loss: {:.4f}, val_loss: {:.4f}, val_accuracy: {:.4f}".format(
            epoch, result['train_loss'], result['val_loss'], result['val_accuracy']))

def accuracy(outputs, labels):
    _, preds = torch.max(outputs, dim=1)
    return torch.tensor(torch.sum(preds == labels).item() / len(preds))

# convolution block with BatchNormalization
def ConvBlock(in_channels, out_channels, pool=False):
    layers = [nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
             nn.BatchNorm2d(out_channels),
             nn.ReLU(inplace=True)]
    if pool:
        layers.append(nn.MaxPool2d(4))
    return nn.Sequential(*layers)

# ResNet-9 modelini tanımlayın
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

# Modeli oluşturun ve yükleyin
model = ResNet9(in_channels=3, num_classes=class_size)
model.load_state_dict(torch.load("mymodel.pth", map_location=device))
model = model.to(device)
model.eval()

# Görüntüyü işleme fonksiyonu
def preprocess_image(img):
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    img = transform(img).unsqueeze(0)  # Batch boyutuna dönüştür
    return img

# Tahmin yapma fonksiyonu
def predict_image(img):
    img = preprocess_image(img)
    img = img.to(device)
    with torch.no_grad():
        outputs = model(img)
        st.write(f"Çıktılar: {outputs}")
    probabilities = torch.nn.functional.softmax(outputs, dim=1)
    confidence, predicted = torch.max(probabilities, 1)
    st.write(f"Tahmin Edilen Sınıf: {predicted}")
    return predicted.cpu().numpy()[0], confidence.cpu().numpy()[0]

# Streamlit arayüzü
st.title("Fasulye Hastalığı Tespit Uygulaması")

camera_input = st.camera_input('Kameradan resim çek')
gallery_input = st.file_uploader('VEYA Fasulye Fotoğrafı Ekleyin', accept_multiple_files=False)

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
