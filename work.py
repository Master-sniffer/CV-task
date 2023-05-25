import streamlit as st
from PIL import Image
import requests
from io import BytesIO
import torch
from torchvision import transforms
from torchvision.models.detection import faster_rcnn
from PIL import Image, ImageDraw, ImageFont
import os
import torchvision.transforms as T
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from PIL import Image, ImageDraw
from pyngrok import ngrok

try:
    ngrok.kill()
except:
    pass

# Запуск NGROK
ngrok.set_auth_token("2QG5HdWTGLE21LL8MjWLzLIq7YO_mU2eQNTB1Yy4ToVvWL5k") # это секретный ключ и лучше его не палить


ngrok_tunnel = ngrok.connect(8501)
public_url = ngrok_tunnel.public_url

# Можно все эти обработки положить в класс, но чуть позже
# нахождение изображений на гербе
def analyze_image(image, threshold):

    model = faster_rcnn.fasterrcnn_resnet50_fpn(pretrained=True) #Используем преобученную модель и всегда для картинок будем ее так вызывать на случай, есл картиннки вообще будут
    model.eval()

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # по возможности используем видео-карту
    model = model.to(device) # закидываем модель в сам девайс

    transform = transforms.Compose([transforms.ToTensor()]) # нужно для перевода картинки в тензор

    image_tensor = transform(image).unsqueeze(0).to(device)

    
    with torch.no_grad(): # используем модель для предсказаний
        predictions = model(image_tensor)

    
    filtered_predictions = [p for p in predictions[0]["scores"] if p > threshold] # убираем предсказания, которые вышли за границу значения фильтра

    if len(filtered_predictions) > 0:
        draw = ImageDraw.Draw(image) # нужно, если хотим рисовать на картинке
        for i in range(len(filtered_predictions)):
            bbox = predictions[0]["boxes"][i].tolist() # объявляем квадрат, который будем рисовать
            label = predictions[0]["labels"][i].item() # и обозначение того, что нашли
            label_name = f"Label: {label}"  #  название того,  что нашшли
            draw.rectangle(bbox, outline="red", width=3) # здесь мы уже рисуем то, что нашли (точнее обводим)
            draw.text((bbox[0], bbox[1] - 20), label_name, fill="red") # а здесь пишем текст
        print("Bounding boxes added to image")
        return image # возвращаем картинку
    else:
        return "No symbols found" # или сообщение

def alt_way(image):
    model = fasterrcnn_resnet50_fpn(pretrained=True)
    model.eval()

    transform = T.Compose([T.ToTensor()])
    image_tensor = transform(image)

    # Make predictions
    with torch.no_grad():
        predictions = model([image_tensor])

    # тут немного иной подход, который можно использовать тоже для обнаружения
    boxes = predictions[0]["boxes"].tolist()
    labels = predictions[0]["labels"].tolist()
    scores = predictions[0]["scores"].tolist()

    
    coat_of_arms_index = -1
    max_score = 0
    for i, label in enumerate(labels): # находим индекс объекта с самым высоким значением обнаружения
        if label == 1:  # 1 для нас это герб
            if scores[i] > max_score:
                coat_of_arms_index = i
                max_score = scores[i]


    if coat_of_arms_index != -1: # если мы видим, что наше значение не равно -1, значит все необходимое мы нашли и можем рисовать
        coat_of_arms_box = boxes[coat_of_arms_index]
        draw = ImageDraw.Draw(image)
        draw.rectangle(coat_of_arms_box, outline="red")
        return image
    else:
        return("This is not a coat of arms.")


def third_way(image):

    model = fasterrcnn_resnet50_fpn(pretrained=True)
    model.eval()

    transform = T.Compose([T.ToTensor()])
    image_tensor = transform(image)

    # Make predictions
    with torch.no_grad():
        predictions = model([image_tensor])


    labels = predictions[0]["labels"].tolist() # здесь мы сначала проверяем на наличие герба, а потом заполняем
    if 1 in labels:
        boxes = predictions[0]["boxes"].tolist()
        labels = predictions[0]["labels"].tolist()

        draw = ImageDraw.Draw(image)
        for box, label in zip(boxes, labels):
            draw.rectangle(box, outline="red")
            draw.text((box[0], box[1]), str(label), fill="red")
        return image
    else:
        return("This is not a coat of arms.")
    

# данный метод я использую для того, чтобы точно понять, есть ли герб на картинке
def final_way(image, threshold):
    model = faster_rcnn.fasterrcnn_resnet50_fpn(pretrained=True)
    model.eval()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    transform = transforms.Compose([transforms.ToTensor()])
    coat_of_arms_label = 1

    image_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        predictions = model(image_tensor)

    # фильтруем тут все так, чтобы предсказания были максимально точными
    filtered_predictions = [pred for pred in predictions[0]["scores"] if pred > threshold and predictions[0]["labels"][pred.argmax()] == coat_of_arms_label]

    if len(filtered_predictions) > 0:
        draw = ImageDraw.Draw(image)
        for i in range(len(predictions[0]["labels"])):
            if predictions[0]["labels"][i] == coat_of_arms_label and predictions[0]["scores"][i] > threshold:
                bbox = predictions[0]["boxes"][i].tolist()
                label = predictions[0]["labels"][i].item()
                label_name = f"Label: {label}" 
                draw.rectangle(bbox, outline="red", width=3)
                draw.text((bbox[0], bbox[1] - 20), label_name, fill="red")
        return image
    else:
        return "None found"


st.set_option('deprecation.showfileUploaderEncoding', False)
st.title("Тестовое в Вулкан")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"]) # какой формат изображений принимается

if uploaded_file is not None:
    img = Image.open(uploaded_file).convert("RGB")

    st.image(img, caption="Uploaded Image", use_column_width=True)

    analysis = final_way(img, 0.9)
    print(type(analysis))
    try: # тут идет очень стремная конструкция, но так как просили сделать как можно раньше, то вот
        st.image(analysis, caption="Reworked Image", use_column_width=True)
    except:
        st.text("ДАЛЬШЕ ИДЁТ ИСКЛЮЧИТЕЛЬНО ОЗНАКОМИТЕЛЬНАЯ ЧАСТЬ НА СЛУЧАЙ,ЕСЛИ ЧТО-ТО ПОШЛО НЕ ТАК :)")
        try:
            analysis = alt_way(img)
            st.image(analysis, caption="Reworked Image", use_column_width=True)
        except:
            st.text("Попробуем другой threshold")
            try:
                analysis =third_way(img)
                st.image(analysis, caption="Reworked Image", use_column_width=True)
            except:
                st.text("это точно сработает !")
                try:
                    analysis = analyze_image(img, 0.1)
                    st.image(analysis, caption="Reworked Image", use_column_width=True)
                except:
                    st.text("Ничего у меня не получается...")

ngrok.kill()




