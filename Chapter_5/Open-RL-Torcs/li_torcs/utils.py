import cv2
import numpy as np
from struct import pack, unpack
import os

IMG_SIZE = 640*480*3
F_OFFSET = 4
F_SIZE = 4
UCH_SIZE = 1

# mean_file = np.load("./mean_file.npy", "r")

# def prepross(img):
#     r = img[:,:,0].astype("float") - mean_file[0,:,:]
#     g = img[:,:,1].astype("float") - mean_file[1,:,:]
#     b = img[:,:,2].astype("float") - mean_file[2,:,:]
#     img_ = np.array([r, g, b]) # (C,H,W)
#     return np.transpose(img_, (1,2,0)) # (H, W, C)

class Render(object):
    """docstring for Render"""
    def __init__(self):
        self.window = cv2.namedWindow("TORCS", cv2.WINDOW_AUTOSIZE)

    def update(self, img):
        cv2.imshow("TORCS", img)
        cv2.waitKey(3)

# Read from shared memory
'''
struct shared_use_st
{
    // N.B. always put float behind uint8 to save memory.
    uint8_t data[image_width*image_height*3];
	uint8_t written;
	uint8_t restart;
    uint8_t relaunch;
    uint8_t trk_type;
    float steer; // only read
    float acc;
    float brake;
	float speed; // only write
	float to_track_middle;  // only write
    float angle;  // only write
    float dist_raced;  // only write
    float toMarking_L;
    float toMarking_M;
    float toMarking_R;
    float toMarking_LL;
    float toMarking_ML;
    float toMarking_MR;
    float toMarking_RR;
    float sl_x; // track seg start left x
    float sl_y;
    float sr_x; // track seg start right x
    float sr_y;
    float car_x; // car global x
    float car_y;
};
'''
def read_sl_x(shared):
    return unpack("f", shared[IMG_SIZE + 4*UCH_SIZE + 14*F_OFFSET:
    IMG_SIZE + 4 * UCH_SIZE + 14 * F_OFFSET + F_SIZE])[0]

def read_sl_y(shared):
    return unpack("f", shared[IMG_SIZE + 4*UCH_SIZE + 15*F_OFFSET:
    IMG_SIZE + 4 * UCH_SIZE + 15 * F_OFFSET + F_SIZE])[0]

def read_sr_x(shared):
    return unpack("f", shared[IMG_SIZE + 4*UCH_SIZE + 16*F_OFFSET:
    IMG_SIZE + 4 * UCH_SIZE + 16 * F_OFFSET + F_SIZE])[0]

def read_sr_y(shared):
    return unpack("f", shared[IMG_SIZE + 4*UCH_SIZE + 17*F_OFFSET:
    IMG_SIZE + 4 * UCH_SIZE + 17 * F_OFFSET + F_SIZE])[0]

def read_car_x(shared):
    return unpack("f", shared[IMG_SIZE + 4*UCH_SIZE + 18*F_OFFSET:
    IMG_SIZE + 4 * UCH_SIZE + 18 * F_OFFSET + F_SIZE])[0]

def read_car_y(shared):
    return unpack("f", shared[IMG_SIZE + 4*UCH_SIZE + 19*F_OFFSET:
    IMG_SIZE + 4 * UCH_SIZE + 19 * F_OFFSET + F_SIZE])[0]

def read_img(shared):
    img = np.fromstring(shared[:IMG_SIZE], dtype = np.uint8)
    img = img.reshape(480, 640, 3)
    # crop = img[480-320:, :, :]  # Cut out sky part
    res = cv2.resize(img, (280, 210)) # 134
    # res = cv2.cvtColor(res, cv2.COLOR_RGB2GRAY)
    # assert res.shape == (210, 280), "image shape must be (210, 280)"
    return res

def read_written(shared):
    return np.fromstring(shared[IMG_SIZE], dtype = np.uint8)[0]

def read_speed(shared):
    return unpack("f", shared[IMG_SIZE + 4*UCH_SIZE + 3*F_OFFSET:
    IMG_SIZE + 4 * UCH_SIZE + 3 * F_OFFSET + F_SIZE])[0]

def read_to_track_middle(shared):
    return unpack("f", shared[IMG_SIZE + 4*UCH_SIZE + 4*F_OFFSET:
        IMG_SIZE + 4*UCH_SIZE + 4*F_OFFSET + F_SIZE])[0]

def read_angle(shared):
    return unpack("f", shared[IMG_SIZE + 4*UCH_SIZE + 5*F_OFFSET:
        IMG_SIZE + 4*UCH_SIZE + 5*F_OFFSET + F_SIZE])[0]

def read_dist_raced(shared):
    return unpack("f", shared[IMG_SIZE + 4*UCH_SIZE + 6*F_OFFSET:
        IMG_SIZE + 4*UCH_SIZE + 6*F_OFFSET + F_SIZE])[0]

def write_written(memory):
    memory.write("0", offset = IMG_SIZE)

def write_restart(memory):
    memory.write("1", offset = IMG_SIZE + 1 * UCH_SIZE)

def write_relaunch(memory):
    memory.write("1", offset = IMG_SIZE + 2 * UCH_SIZE)

def write_steer(memory, s):
    s_ = pack("f", s)
    memory.write(s_, offset = IMG_SIZE + 4 * UCH_SIZE)

def write_acc(memory, acc=0.2):
    acc_ = pack('f', acc)
    memory.write(acc_, offset = IMG_SIZE + 4 * UCH_SIZE + 1 * F_OFFSET)

def write_brake(memory, brake=0.0):
    brake_ = pack('f', brake)
    memory.write(brake_, offset = IMG_SIZE + 4 * UCH_SIZE + 2 * F_OFFSET)
