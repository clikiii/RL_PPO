import cv2
import torch
import numpy as np

def process_state_img(state_img):
    state_img = state_img[:84, :84] # crop the info bar
    state_img = cv2.cvtColor(state_img, cv2.COLOR_RGB2GRAY)

    # state_img = torch.Tensor(state_img).permute(2, 0, 1).unsqueeze(0)
    state_img = torch.Tensor(state_img).unsqueeze(0).unsqueeze(0)
    return state_img

def custom_reward(state_img):
    """
    state_img is a gray scale img whose size is [84, 84]
    """
    # state_img = state_img[61:77, 40:56] # 16*16 crop the car
    state_img = state_img.numpy()
    img_left = state_img[61:77, 40:45] # check 5 pixel on the left/right the car
    img_right = state_img[61:77, 51:56]
    std_avg = 0.5*(np.std(img_left) + np.std(img_right))

    if np.mean(img_left) > 115 or np.mean(img_right) > 115: return -0.5
    if std_avg > 5: return -0.5
    return 0

