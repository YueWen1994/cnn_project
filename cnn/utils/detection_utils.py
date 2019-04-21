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


def main_detect_for_five_pics(folder='input_images/'):
    imgs_name = ['1.png',
                 '2.png',
                 '3.png',
                 '4.png',
                 '5.png',
                 ]
    detection_model = load_model('saved_models/detection_model.hdf5')
    classification_model = load_model('saved_models/VGGPreTrained_classifier_corrected.hdf5')
    for img_name in imgs_name:
        img = cv2.imread(folder + img_name)
        name = img_name.split('.')[0]
        print(name)
        try:
            detected_result = detect_and_draw(img, detection_model, classification_model, name, save_img=True)
        except:
            print(name, ' failed.')


def mp4_video_writer(filename, frame_size, fps=20):
    """Opens and returns a video for writing.

    Use the VideoWriter's `write` method to save images.
    Remember to 'release' when finished.

    Args:
        filename (string): Filename for saved video
        frame_size (tuple): Width, height tuple of output video
        fps (int): Frames per second
    Returns:
        VideoWriter: Instance of VideoWriter ready for writing
    """
    fourcc = cv2.VideoWriter_fourcc(*'MP4V')
    return cv2.VideoWriter(filename, fourcc, fps, frame_size)


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
    seq_probs = np.asarray(preds_probs[1:5]).squeeze()
    seq = np.argmax(seq_probs, axis=1)
    detected_bbox = bbox
    img_c = img.copy()
    cv2.rectangle(img_c, (detected_bbox['left'], detected_bbox['top']), (detected_bbox['right'], detected_bbox['bottom']), (0, 0, 255), thickness)
    sequence = seq[seq != 10]

    text1 = str(sequence)
    print(text1)
    loc = (detected_bbox['left'] , detected_bbox['bottom'] + 5)
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(img_c, text1, loc, font, fontScale=fontScale, color=(0, 255, 0), lineType=3, thickness=thickness)
    if save_img == True:
        cv2.imwrite('output/' + name + '.png', img_copy)
    return img_c


def create_video(input_video, output_video, detection_model, classification_model):
    num = 1
    fourcc = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')
    cap = cv2.VideoCapture(input_video)
    outCap = cv2.VideoWriter(output_video, fourcc, 10, (540, 960), True)
    fps = 40
    w = 1020
    h = 1920
    video_out = mp4_video_writer(output_video, (w, h), fps)
    while cap.isOpened():
        ret, frame = cap.read()
        if ret == True:
            currImg = frame
            print("Processing frame {}".format(num))
            num = num + 1
        else:
            currImg = None
            break
        frame = np.rot90(np.rot90(np.rot90(currImg)))
        currImg = np.copy(frame)
        outIm = detect_and_draw(currImg, detection_model, classification_model, name='test',
                                fontScale=2, thickness=3, save_img=False)
        if outIm is not None:
            video_out.write(outIm)

    cap.release()
    outCap.release()
    cv2.destroyAllWindows()
