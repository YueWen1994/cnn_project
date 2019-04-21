import cv2
import numpy as np
from keras.models import load_model
from utils.preprocess_utils import normalize_and_scale
import pickle
from tqdm import tqdm

def create_contour_msk(img, threshold=0.2):
    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    sobel_x = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=15)
    sobel_y = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=15)

    sobel_magtitude = np.sqrt(np.square(sobel_x) + np.square(sobel_y))
    edges = sobel_magtitude > threshold * np.max(sobel_magtitude)

    # Perform some smoothing to eliminate the spots
    smoothed = np.uint8(edges.copy())
    smoothed = cv2.morphologyEx(smoothed, cv2.MORPH_CLOSE, np.ones((3, 3)))
    smoothed = cv2.morphologyEx(smoothed, cv2.MORPH_OPEN, np.ones((5, 5)))
    contours, _ = cv2.findContours(smoothed, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours = np.asarray(contours)

    area = np.array([cv2.contourArea(contour) for contour in contours])
    bbox = np.array([cv2.boundingRect(contour) for contour in contours]).astype(float)
    aspect_ratio = bbox[:, 2].flatten() / bbox[:, 3].flatten()

    is_valid = (area > 200) & (aspect_ratio < 3) & (aspect_ratio > 0.25)

    mask = np.ones(smoothed.shape[:2], dtype='uint8') * 255
    for ix in range(0, len(is_valid), 1):
        if is_valid[ix] == False:
            cv2.drawContours(mask, [contours[ix]], -1, 0, -1)

    mask = cv2.morphologyEx(mask, cv2.MORPH_DILATE, np.ones((2, 2)))
    smoothed = cv2.bitwise_and(smoothed, mask)
    final_msk = cv2.morphologyEx(smoothed, cv2.MORPH_OPEN, np.ones((2, 2)))
    return final_msk


def load_trained_model():
    # CNNmodel = load_model('saved_models/designedBGRClassifier.hdf5')
    CNNmodel = load_model('saved_models/VGGPreTrained.classifier.hdf5')
    # CNNmodel = load_model('required/VGGPreTrained.classifier.hdf5')

    # CNNmodel = load_model('saved_models/VGGPreTrained.97val_classifier.hdf5')
    return CNNmodel


def substract_img_mean(img, channel):
    if channel == 1:
        img -= np.mean(img.flatten())
    if channel == 3:
        for i in range(3):
            img[:, :, i] -= np.mean(img[:, :, i].flatten())
    return img


def detect_bbox(img, cropped_size=32, steps=2, img_name=None, channel=3, save_img=True, trained_model=None):
    # steps = 2
    #cropped_size = 32
    SIZE = 64
    msks = create_contour_msk(img)

    if trained_model == None:
        model = load_trained_model()
    else:
        model = trained_model

    normalized_img = substract_img_mean(np.float64(np.copy(img)), channel)

    img_height, img_width = img.shape[:2]

    # initialize before pyramid
    scaled_height = np.copy(img_height)
    scaled_width = np.copy(img_width)

    scaled_images = [normalized_img]
    scaled_masks = [msks]
    height_positions = [np.int16(np.round(np.linspace(0, img_height - 1, scaled_height)))]
    weight_positions = [np.int16(np.round(np.linspace(0, img_width - 1, scaled_width)))]

    half_size = np.int16(np.round(cropped_size / 2.))

    for i in range(0, 10, 1):
        # stop when scaled height is too small
        if (scaled_height < half_size) | (scaled_width < half_size):
            break
        scaled_height = np.int(np.round(scaled_height * 0.8))
        scaled_width = np.int(np.round(scaled_width * 0.8))
        new_img = cv2.resize(scaled_images[i], (scaled_width, scaled_height), interpolation=cv2.INTER_AREA)
        new_msk = cv2.resize(scaled_masks[i], (scaled_width, scaled_height), interpolation=cv2.INTER_AREA)
        scaled_images.append(new_img)
        scaled_masks.append(new_msk)
        height_positions.append(np.int16(np.round(np.linspace(0, img_height - 1, scaled_height))))
        weight_positions.append(np.int16(np.round(np.linspace(0, img_width - 1, scaled_width))))

    bbox_locs, digits_probs_recorder = [], []

    half_size = np.int16(np.round(cropped_size / 2.))
    points_threshold = 500
    pyramid_nums = len(scaled_images)

    for pyramid_level in tqdm(range(pyramid_nums)):
        print('pyramid_level: ', pyramid_level)
        current_msk = scaled_masks[pyramid_level]
        current_img = scaled_images[pyramid_level]
        original_weight_positions = weight_positions[pyramid_level]
        original_height_positions = height_positions[pyramid_level]

        current_h, current_w, current_channel = current_img.shape

        if pyramid_level > 3:
            steps = max(int(0.5*steps), 1)
            points_threshold = 150

        for i in range(0, current_h - cropped_size, steps):
            top = np.max([0, i - half_size])
            bottom = np.min([current_h, i + half_size])
            if not np.any(current_msk[top:bottom, :]):
                continue

            for j in range(0, current_w - cropped_size, steps):
                left = np.max([0, j - half_size])
                right = np.min([current_w, j + half_size])

                if (bottom - top < cropped_size) | (right - left < cropped_size):
                    continue

                cropped_msk = current_msk[top:bottom, left: right]
                non_zero_pixels_count = np.count_nonzero(cropped_msk)
                if non_zero_pixels_count < points_threshold:
                    continue

                cropped_img = current_img[top:bottom, left:right, :].astype('float64')
                cropped_img = normalize_and_scale(cropped_img.astype(float))
                reshaped_cropped_img = cv2.resize(cropped_img, (SIZE, SIZE)).reshape(1, SIZE, SIZE, channel)
                predicted_result = model.predict(reshaped_cropped_img)  # predicting digits
                number_of_digits_probs = np.array(predicted_result[0].squeeze())
                digits_probs = np.array(predicted_result[1:5]).squeeze()
                is_digit_probs = np.array(predicted_result[5].squeeze())
                number_of_digits = np.argmax(number_of_digits_probs)

                is_highly_possible_digits = (is_digit_probs[1] > 0.9) & (number_of_digits > 0)

                if is_highly_possible_digits:
                    digits_predicted_result = np.argmax(digits_probs, axis=1)
                    probs = np.hstack((number_of_digits_probs[number_of_digits],
                                       digits_probs[0, digits_predicted_result[0]],
                                       digits_probs[1, digits_predicted_result[1]],
                                       digits_probs[2, digits_predicted_result[2]],
                                       digits_probs[3, digits_predicted_result[3]]))

                    if np.sum(probs) > 3.5:
                        bbox = np.array([original_weight_positions[left],
                                         original_height_positions[top],
                                         original_weight_positions[right],
                                         original_height_positions[bottom]], dtype='int16')
                        bbox_locs.append(bbox)
                        digits_probs_recorder.append(probs)

    bboxes = np.asarray(bbox_locs)
    bboxes_probs = np.asarray(digits_probs_recorder).squeeze()

    potential_msks = np.zeros((img_height, img_width), dtype='float64')
    print('start to find bbox.')
    for idx in range(0, len(bboxes), 1):
        b = bboxes[idx]
        blank = np.zeros_like(img, dtype='uint8')
        cv2.rectangle(blank, (b[0], b[1]), (b[2], b[3]), (255, 255, 255), -1)
        bw = np.float64(blank[:, :, 1]) / 255.
        potential_msks = potential_msks + (bw * np.sum(bboxes_probs[idx]))

    digits_highli_concentrated_areas = potential_msks > np.max(potential_msks) * .1  # 200
    digits_highli_concentrated_areas = cv2.morphologyEx(
        np.uint8(digits_highli_concentrated_areas), cv2.MORPH_CLOSE, np.ones((3, 3)),
        iterations=2
    )
    contours, _ = cv2.findContours(np.uint8(digits_highli_concentrated_areas), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # get bbox from contours
    bbox = []
    for cnt in contours:
        cB = cv2.boundingRect(cnt)
        cbox = [cB[0], cB[1], cB[0] + cB[2], cB[1] + cB[3]]
        bbox.append(cbox)
    # Get final boxes
    potencial_box = []
    preds_probs = []
    for idx in range(0, len(bbox), 1):
        b = bbox[idx]
        current_box = msks[b[1]:b[3], b[0]:b[2]]
        current_box = cv2.morphologyEx(current_box, cv2.MORPH_CLOSE, np.ones((5, 5)), iterations=5)
        bwcontours, tree = cv2.findContours(np.uint8(current_box), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        tmpbox = []
        for cnt in bwcontours:
            cB = cv2.boundingRect(cnt)
            cbox = np.add([cB[0], cB[1], cB[0] + cB[2], cB[1] + cB[3]],
                          [b[0], b[1], b[0], b[1]])
            tmpbox.append(cbox)
        if not tmpbox:
            continue
        cbox = np.asarray(tmpbox)
        mbox = [np.min(cbox[:, 0]), np.min(cbox[:, 1]), np.max(cbox[:, 2]), np.max(cbox[:, 3])]
        maybe = np.int16(np.round(np.mean([mbox, b], axis=0)))
        potencial_box.append(maybe)

        patch = cv2.resize(normalized_img[maybe[1]:maybe[3], maybe[0]:maybe[2], :], (cropped_size, cropped_size))
        patch = normalize_and_scale(patch)
        predict_y = model.predict(cv2.resize(patch, (SIZE, SIZE)).reshape(1, SIZE, SIZE, current_channel))
        preds_probs.append(predict_y)

    final_bbox = np.asarray(potencial_box)
    img_out = draw_bbox(final_bbox, preds_probs, img, img_name, save_img=save_img)
    return img_out


def draw_bbox(bbox, preds_probs, img, name='test', save_img=True):

    img_copy = np.copy(img)
    font = cv2.FONT_HERSHEY_SIMPLEX

    for ix in range(0, len(bbox), 1):
        nDig = np.argmax(preds_probs[ix][0])
        if nDig == 0:
           continue # not a sequence

        nDigProb = np.max(preds_probs[ix][0])
        tmp = np.asarray(preds_probs[ix][1:5]).squeeze()
        seq = np.argmax(tmp, axis=1)
        seqProb = np.max(tmp, axis=1)
        confidence = (np.sum(seqProb) + nDigProb)/5.

        if (nDigProb < 0.8): #| (confidence < 0.8):
           continue

        b = bbox[ix]
        cv2.rectangle(img_copy, (b[0], b[1]), (b[2], b[3]), (0,255,255), 2)
        sequence = seq[seq!=10]

        text1 = str(sequence)
        print(text1)
        conf = confidence*100
        print(conf)
        text2 = 'confidence:' + str(('%2.3f'%conf)) + '%'
        org1 = (b[0], b[1] - 5)
        org2 = (b[0], b[3] + 5)

        cv2.putText(img_copy, text1, org1, font,fontScale = 5, color =  (0, 255, 0),lineType = 3,thickness=3)
        cv2.putText(img_copy, text2, org2, font, fontScale = 5, color=(255, 255, 255),lineType =2,thickness=3)

    # cv2.imwrite('graded_images/' + name + '.png', oIm)

    if save_img == True:
       cv2.imwrite('output/' + name + '.png', img_copy)

    return img_copy


def loadAndDetectImages():
    for i in range(1,7,1):
        imName = str(np.uint8(i))
        Img = cv2.imread('required/'+ imName + '.jpg',1)
        # runCNNDetection(Img, imgName = imName)
        detect_bbox(Img, img_name=imName, channel=3, trained_model= None)


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
