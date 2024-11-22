import cv2
import numpy as np
import random
import os

def generate_cricle_images(output_dir,num_images=1500,img_sieze=128):
    os.makedirs(output_dir,exist_ok=True)
    annotation = []
    for i in range(num_images):
        img = np.ones((img_sieze,img_sieze,3),dtype=np.uint8)*255
        x = random.randint(20,img_sieze-20)
        y = random.randint(20,img_sieze-20)
        r= random.randint(10,40)

        color = (random.randint(0,255),random.randint(0,255),random.randint(0,255))
        cv2.circle(img,(x,y),r,color,-1)

        filename = str(i)+".png"
        cv2.imwrite(os.path.join(output_dir,filename),img)
        annotation.append(f"{filename},{x},{y}")

    with open(os.path.join(output_dir,"annotations.cvs"),'w') as f:
          f.write("\n".join(annotation))

generate_cricle_images("circle_dataset", num_images=1000, img_sieze=128)
