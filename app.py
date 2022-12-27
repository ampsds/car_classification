import io
import json
import os

from linebot import LineBotApi, WebhookHandler
from linebot.exceptions import InvalidSignatureError
from linebot.models import MessageEvent, TextMessage, TextSendMessage, ImageMessage

import torch
import torch.nn as nn

import torchvision.models as models
import torchvision.transforms as transforms
from torchvision.models import resnet50, ResNet50_Weights

from PIL import Image
from flask import Flask, jsonify, request, abort, render_template
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

##ラインボット
CHANNEL_ACCESS_TOKEN = os.getenv('CHANNEL_ACCESS_TOKEN')
CHANNEL_SECRET = os.getenv('CHANNEL_SECRET')

line_bot_api = LineBotApi(CHANNEL_ACCESS_TOKEN)
handler = WebhookHandler(CHANNEL_SECRET)

##推論

#imagenet_class_index = json.load(open('imagenet_class_index.json'))
classes = ['86', 'ハリアー', 'ハイエース', 'ノア', 'プリウス', 'シエンタ', 'ステップワゴン']
model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)

num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 7) #出力数に応じて変更

#model = models.densenet121(pretrained=True)               # Trained on 1000 classes from ImageNet
#model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)

model.load_state_dict(torch.load("./model50_weight_cpu.pth"))


model.eval()                                              # Turns off autograd and

img_class_map = None
mapping_file_path = 'index_to_name.json'                  # Human-readable names for Imagenet classes
if os.path.isfile(mapping_file_path):
    with open (mapping_file_path) as f:
        img_class_map = json.load(f)

# Transform input into the form our model expects
def transform_image(image_bytes):
    my_transforms = transforms.Compose([transforms.Resize(255),
                                        transforms.CenterCrop(224),
                                        transforms.ToTensor(),
                                        transforms.Normalize(
                                            [0.485, 0.456, 0.406],
                                            [0.229, 0.224, 0.225])])
    image = Image.open(io.BytesIO(image_bytes))
    return my_transforms(image).unsqueeze(0)

def get_prediction(image_bytes):
    tensor = transform_image(image_bytes=image_bytes)
    outputs = model.forward(tensor)
    _, y_hat = outputs.max(1)
    #predicted_idx = str(y_hat.item())
    #return imagenet_class_index[predicted_idx]
    return classes[y_hat.item()]

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.after_request
def after_request(response):
  response.headers.add('Access-Control-Allow-Origin', '*')
  response.headers.add('Access-Control-Allow-Headers', 'Content-Type')
  response.headers.add('Access-Control-Allow-Methods', 'GET,PUT,POST,DELETE,OPTIONS')
  return response

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        file = request.files['file']
        img_bytes = file.read()
        #class_id, class_name = get_prediction(image_bytes=img_bytes)
        class_name = get_prediction(image_bytes=img_bytes)
        return jsonify({'class_name': class_name})

#LINE BOTウェブフック
@app.route("/callback", methods=['POST'])
def callback():
  signature = request.headers['X-Line-Signature']
  body = request.get_data(as_text=True)
  app.logger.info("Request body: " + body)

  try:
    handler.handle(body, signature)
  except InvalidSignatureError:
    abort(400)

  return 'OK'

@handler.add(MessageEvent, message=TextMessage)
def handle_message(event):
  line_bot_api.reply_message(
    event.reply_token,
    TextSendMessage(text=event.message.text))

@handler.add(MessageEvent, message=ImageMessage)
def handle_image(event):
    content = line_bot_api.get_message_content(event.message.id)
    img = b""
    for chunk in content.iter_content():
        img += chunk
 
    class_name = get_prediction(image_bytes=img)

    line_bot_api.reply_message(
        event.reply_token,
        TextSendMessage(text=class_name))
