import cv2
from ultralytics import YOLO
import torch 
import torchvision.models as models
import numpy as np
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(device)

model = models.resnet50(weights=None)
model.fc = torch.nn.Linear(model.fc.in_features, 14 * 2)
state_dict = torch.load("keypoints_model.pth", map_location=device)
model.load_state_dict(state_dict) 
model.to(device) 
model.eval()


def image_for_points(img):
        img_resized = cv2.resize(img, (224, 224))
        img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
        img_normalized = img_rgb / 255.0
        img_normalized = (img_normalized - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]

        tensor= torch.tensor(img_normalized, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0)
        return tensor.to(device)
ball  = YOLO('best.pt')
ball.to(device)
_ = ball(np.zeros((1080, 1920, 3), dtype=np.uint8), verbose=False)
            
video_source = "videos/video_input3.mp4"
cap = cv2.VideoCapture(video_source)
count=0

with torch.no_grad():
    ret,frame = cap.read()
    tensor = image_for_points(frame)
    output = model(tensor).squeeze().cpu().numpy() 
import time

while True:
    t0 = time.time()
    
    ret, frame = cap.read()
    if not ret:
        print('no way hosay')
        break
    
    image = frame
    orig_h, orig_w = image.shape[:2]
    


    
    if count%30 == 0:
        tensor = image_for_points(image)
         
        with torch.no_grad():
            
            output = model(tensor.to(device)).squeeze().cpu().numpy() 
            
    x=output[0:7:2]       
    y=output[1:8:2]

    if count%2 == 0:
        
        result = ball(image, verbose=False)[0]
        
    
        
        
    count+=1

    for i,(x,y) in enumerate(zip(x,y)):
        x_orig = int(x * (orig_w / 224.0))
        y_orig = int(y * (orig_h / 224.0)) 
        cv2.circle(image, (x_orig, y_orig), radius=5, color=(0, 255, 0), thickness=-1)
        cv2.putText(image, str(i), (x_orig + 10, y_orig - 5), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)

    

    for box in result.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        conf = float(box.conf[0]) 
        class_name = ball.names[int(box.cls[0])]
        cv2.circle(image, ((x1+x2)//2, (y1+y2)//2),6,(0, 255, 0), 3)

    cv2.imshow("Manually Annotated Original Frame", image)
    if cv2.waitKey(1) & 0xFF == ord('q'):
         break
    torch.cuda.empty_cache()
    print(time.time() - t0) 
cap.release()
cv2.destroyAllWindows()

