from __future__ import division
import warnings
import random
from Networks.HR_Net.seg_hrnet import get_seg_model
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
import dataset
import math             
import matplotlib.pyplot as plt
from image import *
from utils import *
from threading import Thread, Event
import logging
import nni
# from nni.utils import merge_parameter
from config import return_args, args

warnings.filterwarnings('ignore')
import time

logger = logging.getLogger('mnist_AutoML')

print(args)



def generate_trajectory_image(kpoint, trajectory_image=None):
    rate = 1
    pred_coor = np.nonzero(kpoint)
    
    # Initialize the trajectory image if not provided
    if trajectory_image is None:
        trajectory_image = np.zeros((int(kpoint.shape[0] * rate), int(kpoint.shape[1] * rate), 3), dtype="uint8") + 255  # White background

    coord_list = []
    for i in range(0, len(pred_coor[0])):
        hold = int(pred_coor[0][i] * rate)
        w = int(pred_coor[1][i] * rate)
        h = int(hold*1.35)
        coord_list.append([w, h])
        
        cv2.circle(trajectory_image, (w, h), 3, (0, 0, 0), -1)  # Draw circle on the persistent trajectory image

    return trajectory_image



def generate_point_map(kpoint):
    rate = 1
    pred_coor = np.nonzero(kpoint)
    point_map = np.zeros((int(kpoint.shape[0] * rate), int(kpoint.shape[1] * rate), 3), dtype="uint8") + 255  # 22
    # count = len(pred_coor[0])
    coord_list = []
    for i in range(0, len(pred_coor[0])):
        hold = int(pred_coor[0][i] * rate)
        w = int(pred_coor[1][i] * rate)
        h=int(hold*1.5)
        coord_list.append([w, h])
        cv2.circle(point_map, (w, h), 3, (0, 0, 0), -1)

    return point_map


def generate_bounding_boxes(kpoint, Img_data):
    '''generate sigma'''
    pts = np.array(list(zip(np.nonzero(kpoint)[1], np.nonzero(kpoint)[0])))
    leafsize = 2048
        
    if pts.shape[0] > 0: # Check if there is a human presents in the frame
        # build kdtree
        tree = scipy.spatial.KDTree(pts.copy(), leafsize=leafsize)

        distances, locations = tree.query(pts, k=4)
        for index, pt in enumerate(pts):
            pt2d = np.zeros(kpoint.shape, dtype=np.float32)
            pt2d[pt[1], pt[0]] = 1.
            if np.sum(kpoint) > 1:
                sigma = (distances[index][1] + distances[index][2] + distances[index][3]) * 0.1
            else:
                sigma = np.average(np.array(kpoint.shape)) / 2. / 2.  # case: 1 point
            sigma = min(sigma, min(Img_data.shape[0], Img_data.shape[1]) * 0.04)

            if sigma < 6:
                t = 2
            else:
                t = 2
            Img_data = cv2.rectangle(Img_data, (int(pt[0] - sigma), int(pt[1] - sigma)),
                                    (int(pt[0] + sigma), int(pt[1] + sigma)), (0, 255, 0), t)

    return Img_data


def show_fidt_func(input):
    input[input < 0] = 0
    input = input[0][0]
    fidt_map1 = input
    fidt_map1 = fidt_map1 / np.max(fidt_map1) * 255
    fidt_map1 = fidt_map1.astype(np.uint8)
    fidt_map1 = cv2.applyColorMap(fidt_map1, 2)
    return fidt_map1


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def counting(input):
    input_max = torch.max(input).item()
    keep = nn.functional.max_pool2d(input, (3, 3), stride=1, padding=1)
    keep = (keep == input).float()
    input = keep * input

    input[input < 120.0 / 255.0 * torch.max(input)] = 0
    input[input > 0] = 1

    '''negative sample'''
    if input_max<0.1:
        input = input * 0

    count = int(torch.sum(input).item())

    kpoint = input.data.squeeze(0).squeeze(0).cpu().numpy()

    return count, kpoint

transform=transforms.Compose([transforms.ToTensor(),transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),])
img_transform = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
tensor_transform = transforms.ToTensor()





def get_prediction_webcam(event: Event):  
    device = torch.device('cuda')
    model = get_seg_model()
    # model = nn.DataParallel(model, device_ids=[0])
    # model = model.cuda()

    checkpoint = torch.load('/Users/pratyushgupta/Desktop/Crowd-Counting-Platform-main/model_best_57_Shangai_Tech_FIDTM.pth')
    model.load_state_dict(checkpoint['state_dict'], strict=False)

    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    #fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')

    #cap = cv2.VideoCapture(args.video_path)
    #cap= cv2.VideoCapture(gstreamer_pipeline(flip_method=0), cv2.CAP_GSTREAMER)
    cap= cv2.VideoCapture(0)
    ret, frame = cap.read()
    print(frame.shape)

    '''out video'''
    width = frame.shape[1] #output size
    height = frame.shape[0] #output size
    out = cv2.VideoWriter('./demo.avi', fourcc, 30, (width, height))

    while True:
        try:
            ret, frame = cap.read()

            scale_factor = 0.5
            frame = cv2.resize(frame, (0, 0), fx=scale_factor, fy=scale_factor)
            ori_img = frame.copy()
        except:
            print("test end")
            cap.release()
            break
        frame = frame.copy()
        image = tensor_transform(frame)
        image = img_transform(image).unsqueeze(0)

        with torch.no_grad():
            d6 = model(image)

            count, pred_kpoint = counting(d6)
            point_map = generate_point_map(pred_kpoint)
            box_img = generate_bounding_boxes(pred_kpoint, frame)
            show_fidt = show_fidt_func(d6.data.cpu().numpy())
            #res = np.hstack((ori_img, show_fidt, point_map, box_img))
            res1 = np.hstack((ori_img, show_fidt))
            res2 = np.hstack((box_img, point_map))
            res = np.vstack((res1, res2))

            cv2.putText(res, "Count:" + str(count), (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            res = cv2.resize(res, (1500,720))
            cv2.imwrite('./demo3.jpg',box_img)
            cv2.imwrite('./demo.jpg', res)
            '''write in out_video'''
            out.write(res)
        
      
        
        cv2.imshow("dst",res)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        print("pred:%.3f" % count)
        
        if event.is_set():
            break
        
    cv2.destroyAllWindows()

def gstreamer_pipeline(
    capture_width=1280,
    capture_height=720,
    display_width=1280,
    display_height=720,
    framerate=60,
    flip_method=0,
):
    return (
        "nvarguscamerasrc ! "
        "video/x-raw(memory:NVMM), "
        "width=(int)%d, height=(int)%d, "
        "format=(string)NV12, framerate=(fraction)%d/1 ! "
        "nvvidconv flip-method=%d ! "
        "video/x-raw, width=(int)%d, height=(int)%d, format=(string)BGRx ! "
        "videoconvert ! "
        "video/x-raw, format=(string)BGR ! appsink"
        % (
            capture_width,
            capture_height,
            framerate,
            flip_method,
            display_width,
            display_height,
        )
    )




def get_prediction(file):  
    device = torch.device("cpu")
    trajectory_image = None
    model = get_seg_model()
    model = nn.DataParallel(model, device_ids=[0])
    model = model.cpu()

    checkpoint = torch.load('/Users/pratyushgupta/Desktop/Crowd-Counting-Platform-main/model_best_nwpu_FIDTM.pth', map_location=torch.device('cpu'))
    model.load_state_dict(checkpoint['state_dict'], strict=False)

# set your image path here
        
    print("------------------------------------")    
    print(file)
    if (file.endswith(".mp4")):
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        #fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')

        cap = cv2.VideoCapture(file)
        ret, frame = cap.read()
        print(frame.shape)
        n=0
        '''out video'''
        width = frame.shape[1] #output size
        height = frame.shape[0] #output size
        out = cv2.VideoWriter('./demo.avi', fourcc, 30, (width, height))

        while True:
            n=n+1
            try:
                
                ret, frame = cap.read()
                if(n%15!=0):
                    continue

                
                scale_factor = 0.5
                frame = cv2.resize(frame, (0, 0), fx=scale_factor, fy=scale_factor)
                ori_img = frame.copy()
            except:
                print("test end")
                cap.release()
                break
            frame = frame.copy()
            image = tensor_transform(frame)
            image = img_transform(image).unsqueeze(0)
            image = image.to(device)
            
            with torch.no_grad():
                d6 = model(image)
                
                count, pred_kpoint = counting(d6)
                point_map = generate_trajectory_image(pred_kpoint, trajectory_image)
                trajectory_image = point_map
                box_img = generate_bounding_boxes(pred_kpoint, frame)
                show_fidt = show_fidt_func(d6.data.cpu().numpy())
                #res = np.hstack((ori_img, show_fidt, point_map, box_img))
                res1 = np.hstack((ori_img, show_fidt))
                res2 = np.hstack((box_img, point_map))
                res = np.vstack((res1, res2))
    
                cv2.putText(res, "Count:" + str(count), (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                print(n)
                            
                cv2.imshow("dst",res)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                   break
    else :
        frame = cv2.imread(file)
        print(frame.shape)
        scale_factor = 0.5
        frame = cv2.resize(frame, (0, 0), fx=scale_factor, fy=scale_factor)
        ori_img = frame.copy()
        frame = frame.copy()
        image = tensor_transform(frame)
        image = img_transform(image).unsqueeze(0)




        with torch.no_grad():
            
            d6= model(image)
            d6_np=d6.data.cpu().numpy()
            segmentation_map = d6_np[0, 0]  # Taking the first class for visualization

            # Normalize the segmentation map to the range [0, 255]
            segmentation_map_normalized = (segmentation_map - np.min(segmentation_map)) / (np.max(segmentation_map) - np.min(segmentation_map))
            segmentation_map_normalized = (segmentation_map_normalized * 255).astype(np.uint8)

            # Save the segmentation map as an image
            cv2.imwrite('./segmentation_map.jpg', segmentation_map_normalized)

            count, pred_kpoint = counting(d6)
            point_map = generate_point_map(pred_kpoint)
            box_img = generate_bounding_boxes(pred_kpoint, frame)
            show_fidt = show_fidt_func(d6.data.cpu().numpy())
            #res = np.hstack((ori_img, show_fidt, point_map, box_img))
            res1 = np.hstack((ori_img, show_fidt))
            res2 = np.hstack((box_img, point_map))
            res = np.vstack((res1, res2))
         
            #cv2.putText(res, "Count:" + str(count), (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            density = './density_map.jpg' 
            cv2.imwrite('./demo3.jpg',box_img)
            cv2.imwrite('./demo.jpg', res)
            print("pred:%.3f" % count)
        
            x = random.randint(1,100000) 
            density = 'static/density_map'+str(x)+'.jpg' 
            plt.imsave(density,show_fidt) 
        
        
            return count , density

    
    
file_path = '/Users/pratyushgupta/Desktop/ UGP archives/drone4.mp4'  # Set your file path here
count, density_map_path = get_prediction(file_path)
print(f"Count: {count}, Density Map saved at: {density_map_path}")
