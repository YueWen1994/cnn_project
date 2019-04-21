import cv2
import numpy as np
from keras.models import load_model
from utils.preprocess_utils import normalize_and_scale
import pickle
from tqdm import tqdm
from utils.preprocess_utils import *
train_img_size = (64, 64)
SIZE = 64
channel = 3


def detect_img_bbox(img, model):
    normalized_img = normalize_img(img, train_img_size)
    normalize_img_input = normalized_img.reshape(1, train_img_size[0], train_img_size[1], 1)
    preds = model.predict(normalize_img_input)[0]
    bbox = preds.astype(int)
    top, left, w, h = bbox[0], bbox[1], bbox[2], bbox[3]
    train_size_bbox = {'left': left, 'right': left + w, 'top': top, 'bottom': top + h}
    org_size_bbox = {
        'left':  max(0, int(left * img.shape[1] / train_img_size[1])),
        'right':  min(img.shape[1] - 1, int((left + w) * img.shape[1] / train_img_size[1])),
        'top': max(0, int(top * img.shape[0] / train_img_size[0])),
        'bottom':  min(img.shape[0] - 1, int((top + h) * img.shape[0] / train_img_size[0])),
    }
    return org_size_bbox


def detect_and_draw(img, detection_model, classification_model, fontScale=0.35, name='test', thickness=3, save_img=True):
    bbox = detect_img_bbox(img, detection_model)
    cropped_img = crop_img(img, bbox)
    normalized_img = normalize_and_scale(cropped_img.astype(float))
    predict_y = classification_model.predict(cv2.resize(normalized_img, (SIZE, SIZE)).reshape(1, SIZE, SIZE, channel))
    bbox_result = draw_bbox(bbox, predict_y, img, fontScale=fontScale, name=name, save_img=save_img, thickness=thickness)
    return bbox_result


def draw_bbox(bbox, preds_probs, img, name='test', fontScale=0.35, thickness=3, save_img=True):
    img_copy = np.copy(img)

    n_digit = np.argmax(preds_probs[0])
    if n_digit == 0:
        print('No digits detected!')
        return
    n_digit_prob = np.max(preds_probs[0])
    seq_probs = np.asarray(preds_probs[1:5]).squeeze()
    seq = np.argmax(seq_probs, axis=1)
    seq_prob = np.max(seq_probs, axis=1)
    avg_prob = (np.sum(seq_prob) + n_digit_prob) / 5.
    detected_bbox = bbox
    img_c = img.copy()
    cv2.rectangle(img_c, (detected_bbox['left'], detected_bbox['top']), (detected_bbox['right'], detected_bbox['bottom']), (0, 0, 255), thickness)
    sequence = seq[seq != 10]

    text1 = str(sequence)
    print(text1)
    conf = avg_prob * 100
    print(conf)
    text2 = 'pred_prob:' + str(('%2.3f' % conf)) + '%'
    org1 = (detected_bbox['left'], detected_bbox['top'] - 20)
    loc = (detected_bbox['left'] , detected_bbox['bottom'] + 5)
    w, d = img.shape[:2]
    font_size = int((w + d) / 20) / 20
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(img_c, text1, loc, font, fontScale=fontScale, color=(0, 255, 0), lineType=3, thickness=thickness)
    #cv2.putText(img_c, text2, org2, font, fontScale=0.5, color=(255, 255, 255), lineType=2, thickness=1)
    if save_img == True:
        cv2.imwrite('output/' + name + '.png', img_copy)
    return img_c


def read_and_detect_imgs():
    for i in range(1, 7, 1):
        imName = str(np.uint8(i))
        Img = cv2.imread('required/' + imName + '.jpg', 1)
        # runCNNDetection(Img, imgName = imName)
        detect_bbox(Img, img_name=imName, channel=3, trained_model=None)


def createCNNVideo():
    num = 1
    fourcc = cv2.VideoWriter_fourcc('M','J','P','G')
    CNNmodel = load_trained_model()

    cap = cv2.VideoCapture('input/video.mp4')
    outCap = cv2.VideoWriter('output/videoCNN.avi', fourcc, 10, (540, 960), True)

    while cap.isOpened():
          ret, frame = cap.read()
          if ret == True:
              currImg = frame
              print("Processing frame {}".format(num))
              num = num + 1
          else:
              currImg = None
              break

          currImg = np.copy(frame)
          outIm = detect_bbox(currImg, None, channel=3, save_img= False, trained_model= CNNmodel)
          outCap.write(outIm)

    cap.release()
    outCap.release()
    cv2.destroyAllWindows()
