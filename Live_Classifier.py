import cv2
import torch
import torchvision.transforms as transforms
import PIL.Image as Image
from AlexNet import AlexNet

# transformers
image_transforms = transforms.Compose([
    transforms.Resize((227, 227)),
    transforms.ToTensor(),
    transforms.Normalize((0.5287, 0.4877, 0.4666), (0.3204, 0.3142, 0.3182))])

# Loading the model
model = AlexNet()
model.to("cpu")
model.load_state_dict(torch.load('./models/alexnet_mask_cnn_50.pth'))
model.eval()

# Given an image predict the type of mask
def predict(model, image, image_transforms):
    classes = ["Cloth", "N95", "NoMask", "Surgical"]

    # converting to pil image
    color_coverted = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(color_coverted)

    image = image_transforms(image).float()
    image = image.unsqueeze(0)
    output = model(image)
    _, predicted = torch.max(output.data, 1)
    pred = classes[predicted.item()]
    return pred


# opening webcam
cap = cv2.VideoCapture(0)

while True:

    ret, frame = cap.read()

    # prediction
    prediction = predict(model, frame, image_transforms)

    # write prediction
    cv2.putText(img=frame, text= prediction, org=(150, 100), fontFace=cv2.FONT_HERSHEY_TRIPLEX, fontScale=3,
                color=(0, 255, 0), thickness=3)

    cv2.imshow("Mask detector", frame)
    if cv2.waitKey(1) == 13:
        break

cap.release()
cv2.destroyAllWindows()
