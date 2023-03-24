#packages
import numpy as np
from PIL import Image
import argparse
import pdb
import imageio.v2 as iio
import torch.nn.functional as F
import cv2

def Clickchoose(target_file):
    #img1 = cv2.imread(source_file)
    img2 = cv2.imread(target_file)
    size = img2.shape
    cv2.namedWindow('img')
    x2, y2, w2, h2 = cv2.selectROI('roi',img2)
    cv2.destroyWindow('roi')
    w = size[1]  # 宽度
    h = size[0]  # 高度
   # left_top_source = (x1, y1)
   # right_bottom_source = (x1 + w1, y1 + h1)
    x_start,y_start = x2+(w2/2), y2+(h2/2)
   # right_bottom_source = (x2 + w2, y2 + h2)
    xs = max(w2,h2)
    ts1 = max(w,h)

    return x_start,y_start,int(xs/2),int(ts1)