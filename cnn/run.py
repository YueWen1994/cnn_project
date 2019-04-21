from keras.models import load_model
from utils.detection_utils import detect_and_draw
import cv2

input_img_folder = 'input_images/'
output_folder = 'output/'
img_names_list = ['1.png',
                  '2.png',
                  '3.png',
                  '4.png',
                  '5.png']


def main_detect_five_imgs():
    detection_model = load_model('saved_models/detection_model.hdf5')
    classification_model = load_model('saved_models/VGGPreTrained_classifier_corrected.hdf5')
    for img_name in img_names_list:
        img = cv2.imread(input_img_folder + img_name)
        name = img_name.split('.')[0]
        detected_result = detect_and_draw(img, detection_model, classification_model, fontScale=0.5, name=name, thickness=1, save_img=True, output_folder=output_folder)


if __name__ == '__main__':
    main_detect_five_imgs()



