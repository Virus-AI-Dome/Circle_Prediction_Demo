
import os
import cv2
import numpy as np
import random
import torch
from torch.utils.data import Dataset, DataLoader
from model_trian import CircleCenterNet

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CircleCenterNet().to(device)
model.load_state_dict(torch.load("model_best_0.09984062297735363.pth"))
model.eval()

# 测试函数
def predict_circle_center(img, model):
    # img = cv2.imread(image_path)
    img_resized = cv2.resize(img, (128, 128))
    img_tensor = torch.tensor(img_resized, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0) / 255.0

    with torch.no_grad():
        prediction = model(img_tensor.to(device)).cpu().numpy()[0]

    x, y = int(prediction[0]), int(prediction[1])
    print(f"预测圆心坐标: ({x}, {y})")

    # 绘制预测结果
    cv2.circle(img, (x, y), 5, (0, 0, 255), -1)
    cv2.imshow("Prediction", img)
    cv2.waitKey(2)


# 示例调用

for i in range(50):
    img = np.ones((128, 128, 3), dtype=np.uint8) * 255
    x = random.randint(10, 128 - 20)
    y = random.randint(10, 128 - 20)
    r = random.randint(10, 40)

    color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
    cv2.circle(img, (x, y), r, color, -1)
    print(f"真实x:{x}, y:{y}")

    # filename = str("TEST") + ".png"
    # cv2.imwrite(os.path.join('./', filename), img)
    predict_circle_center(img, model)
