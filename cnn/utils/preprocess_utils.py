import cv2
import pickle
import h5py
import numpy as np
import os
from sklearn.model_selection import train_test_split

MAX_DIGIT_DETECTED = 4
train_folders = 'data/train'
test_folders = 'data/test'
train_data_name = 'data/train_data.pkl'
test_data_name = 'data/test_data.pkl'


def get_data_info(mat_file):
    data_generator = DataGenerator(mat_file)
    data_info = data_generator.get_final_result()
    return data_info


def generate_data_set_for_training(shape, channel):
    with open(train_data_name, 'rb') as f:
        train_data = pickle.load(f)
    with open(test_data_name, 'rb') as f:
        test_data = pickle.load(f)
    train_formatter = DataFormatter(train_data, train_folders, shape, channel=channel, add_neg=True)
    train_dataset, train_labels = train_formatter.get_formatted_data_for_modeling()
    test_formatter = DataFormatter(test_data, test_folders, shape, channel=channel, add_neg=False)
    test_dataset, test_labels = test_formatter.get_formatted_data_for_modeling()
    X_train, X_val, y_train, y_val = train_test_split(train_dataset, train_labels, test_size=0.2, random_state=0)
    data = {
        'X_train': X_train,
        'X_val': X_val,
        'y_train': y_train,
        'y_val': y_val,
        'X_test': test_dataset,
        'y_test': test_labels

    }
    return data


def normalize_and_scale(image_in, scale_range=(0, 1)):
    """Normalizes and scales an image to a given range [0, 255].

    Utility function. There is no need to modify it.

    Args:
        image_in (numpy.array): input image.
        scale_range (tuple): range values (min, max). Default set to [0, 255].

    Returns:
        numpy.array: output image.
    """
    image_out = np.zeros(image_in.shape)
    cv2.normalize(image_in, image_out, alpha=scale_range[0],
                  beta=scale_range[1], norm_type=cv2.NORM_MINMAX)

    return image_out


class DataGenerator:
    def __init__(self, mat_file):
        self.mat_file = h5py.File(mat_file, 'r')
        self.name = self.mat_file['digitStruct']['name']
        self.bbox = self.mat_file['digitStruct']['bbox']
        self.files_num = len(self.name)

    def get_name(self, n):
        name_chars = [chr(c[0]) for c in self.mat_file[self.name[n][0]].value]
        return ''.join(name_chars)

    def get_metric_value(self, metric):
        if len(metric) > 1:
            metric = [self.mat_file[metric.value[j].item()].value[0][0] for j in range(len(metric))]
        else:
            metric = [metric.value[0][0]]
        return metric

    def get_bbox(self, n):
        bb = self.bbox[n].item()
        bbox = {
            'height': self.get_metric_value(self.mat_file[bb]["height"]),
            'label': self.get_metric_value(self.mat_file[bb]["label"]),
            'left': self.get_metric_value(self.mat_file[bb]["left"]),
            'top': self.get_metric_value(self.mat_file[bb]["top"]),
            'width':  self.get_metric_value(self.mat_file[bb]["width"]),
        }
        return bbox

    def get_name_and_bbox(self, n):
        bbox_with_name = self.get_bbox(n)
        bbox_with_name['name'] = self.get_name(n)
        return bbox_with_name

    def get_all_names_and_bboxes(self):
        return [self.get_name_and_bbox(i) for i in range(self.files_num)]

    def get_final_result(self):
        all_name_and_bboxes = self.get_all_names_and_bboxes()
        result = []
        for i in range(self.files_num):
            ith_file = {'filename': all_name_and_bboxes[i]["name"]}
            ith_file_bbox = [
                {
                    'height': all_name_and_bboxes[i]['height'][j],
                    'label': all_name_and_bboxes[i]['label'][j],
                    'left':  all_name_and_bboxes[i]['left'][j],
                    'top': all_name_and_bboxes[i]['top'][j],
                    'width': all_name_and_bboxes[i]['width'][j],
                }
                for j in range(len(all_name_and_bboxes[i]['height']))
            ]
            ith_file['boxes'] = ith_file_bbox
            result.append(ith_file)
        return result


class DataFormatter:
    def __init__(self, data, folder, shape, channel, add_neg, max_digit_detected=MAX_DIGIT_DETECTED):
        self.data = data
        self.folder = folder
        self.shape = shape
        self.data_length = len(self.data)
        self.channel = channel
        self.add_neg = add_neg
        self.max_digit_detected = max_digit_detected
        NEG_LABEL = np.asarray([0] + [10] * self.max_digit_detected + [0], dtype='uint8')
        self.neg_label = NEG_LABEL

    def get_formatted_data_for_modeling(self):
        self.output_data = []
        self.labels = []
        NEG_LABEL = self.neg_label

        for i in range(self.data_length):

            filename_path = os.path.join(self.folder, self.data[i]['filename'])

            # read as three chanel rgb
            if self.channel == 1:
                img_org = cv2.imread(filename_path, 0)
            if self.channel == 3:
                img_org = cv2.imread(filename_path)

            img = img_org.copy()

            img_height, img_width = img.shape[:2]

            boxes = self.data[i]['boxes']
            # Get labels
            num_digits = len(boxes)
            if num_digits > self.max_digit_detected:
                # one for num of boxes, others for the 5 digits, None presents no digits
                continue
            else:
                ith_label = np.copy(self.neg_label)
                ith_label[0] = num_digits
                for j in range(1, num_digits + 1):
                    ith_label[j] = boxes[j-1]['label']
                if num_digits > 0:
                    ith_label[-1] = 1
                self.labels.append(ith_label)

            # Get bbox info as as list
            left_positions = [box['left'] for box in boxes]
            top_positions = [box['top'] for box in boxes]
            height_list = [box['height'] for box in boxes]
            width_list = [box['width'] for box in boxes]

            # crop and resize image
            bbox = get_digit_region_box_helper(img, left_positions, top_positions, height_list, width_list)
            cropped_img = crop_img(img, bbox)
            region = cv2.resize(cropped_img, self.shape)
            region = normalize_and_scale(region.astype(float), scale_range=(0, 1))
            self.output_data.append(region)

            # Add negative examples
            if self.add_neg:
                if bbox['top'] > 10 and bbox['left'] > 10:
                    negative_img = cv2.resize(img_org[0: bbox['top'], 0: bbox['left']], self.shape)
                    self.labels.append(NEG_LABEL)
                    self.output_data.append(negative_img)

                    negative_img = cv2.resize(img_org[bbox['top']: bbox['bottom'], 0: bbox['left']], self.shape)
                    self.labels.append(NEG_LABEL)
                    self.output_data.append(negative_img)

                    negative_img = cv2.resize(img_org[0: bbox['top'], bbox['left']: bbox['right']], self.shape)
                    self.labels.append(NEG_LABEL)
                    self.output_data.append(negative_img)

                # Add negative examples
                if img_height - bbox['bottom'] > 10 and img_width - bbox['right'] > 10:
                    negative_img = cv2.resize(img_org[bbox['bottom']:,  bbox['right']:], self.shape)
                    self.labels.append(NEG_LABEL)
                    self.output_data.append(negative_img)

                    negative_img = cv2.resize(img_org[bbox['bottom']:,  bbox['left']:bbox['right']], self.shape)
                    self.labels.append(NEG_LABEL)
                    self.output_data.append(negative_img)

                    negative_img = cv2.resize(img_org[bbox['top']: bbox['bottom'],  bbox['right']:], self.shape)
                    self.labels.append(NEG_LABEL)
                    self.output_data.append(negative_img)

        self.output_data = np.array(self.output_data)
        self.labels = np.array(self.labels)

        print('dataset:', self.output_data.shape)
        print('labels:', self.labels.shape)
        return self.output_data, self.labels


def get_digit_region_box_helper(img, left_positions, top_positions, height_list, width_list):
    height, width = img.shape[:2]
    region_left = min(left_positions)
    region_top = min(top_positions)
    region_height = max(top_positions) + max(height_list) - region_top
    region_width = max(left_positions) + max(width_list) - region_left

    region_top = region_top - region_height * 0.05  # a bit higher
    region_left = region_left - region_width * 0.05  # a bit wider
    region_bottom = min(height, region_top + region_height * 1.05)
    region_right = min(width, region_left + region_width * 1.05)

    return {'top': max(int(region_top), 0),
            'bottom': int(region_bottom),
            'left': max(int(region_left), 0),
            'right': int(region_right),}


def crop_img(img, bbox):
    left, top, right, bottom = bbox['left'], bbox['top'], bbox['right'], bbox['bottom']
    if len(img.shape) == 2:
        return img[top: bottom, left: right]
    if len(img.shape) == 3:
        return img[top: bottom, left: right, :]