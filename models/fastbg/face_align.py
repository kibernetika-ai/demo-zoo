from ml_serving.drivers import driver
import cv2
import os
from scipy import ndimage
import numpy as np
import json
import shutil
import logging
import argparse

LOG = logging.getLogger(__name__)

face_drv = driver.load_driver("model")()

mat_drv = driver.load_driver("model")()

face_input_name = ''
face_input_shape = None
face_output_name = ''

g_mean = np.array(([126.88, 120.24, 112.19])).reshape([1, 1, 3])
unknown_code = 128


def generate_trimap(alpha):
    trimap = np.copy(alpha)
    k_size = 20
    trimap[np.where((ndimage.grey_dilation(alpha[:, :], size=(k_size, k_size)) - ndimage.grey_erosion(alpha[:, :],
                                                                                                      size=(k_size,
                                                                                                            k_size))) != 0)] = unknown_code
    return trimap


def start_coco(args):
    face_drv.load_model('kuberlab-demo/openvino-face-detection:1.4.0-cpu', save_dir=args.model_dir)
    mat_drv.load_model('kuberlab-demo/deepmatting:1.0.0-38', save_dir=args.model_dir)
    global face_input_name, face_input_shape, face_output_name
    face_input_name, face_input_shape = list(face_drv.inputs.items())[0]
    face_output_name = list(face_drv.outputs)[0]
    with open(args.src_dir + '/annotations/instances_train2017.json') as f:
        data = json.load(f)
    data = data['annotations']
    shutil.rmtree(args.out_dir, True)
    os.mkdir(args.out_dir)
    images_dir = args.out_dir + '/images'
    mask_dir = args.out_dir + '/masks'
    preview_dir = args.out_dir + '/preview'
    os.mkdir(images_dir)
    os.mkdir(mask_dir)
    os.mkdir(preview_dir)
    limit_pictures = args.limit_pic
    logging.info('Number pictures: %d', limit_pictures)
    i = 0
    count = len(data)
    if limit_pictures < 0:
        limit_pictures = len(data)
    for a in data:
        if i >= limit_pictures:
            break
        if a['category_id'] == 1 and a['iscrowd'] == 0:
            name = '{:012d}.jpg'.format(a['image_id'])
            fname = args.src_dir + '/train2017/{}'.format(name)
            segmentation = a['segmentation']
            if len(segmentation) >= 4 or len(segmentation) < 1:
                continue
            area = a['area']
            if not os.path.exists(fname):
                continue
            img = cv2.imread(fname, cv2.IMREAD_COLOR)
            img_area = img.shape[0] * img.shape[1]
            if area < img_area * 0.1:
                continue
            mask = np.zeros((img.shape[0], img.shape[1]), np.uint8)
            for s in segmentation:
                p = np.array(s, np.int32)
                p = np.reshape(p, (1, int(p.shape[0] / 2), 2))
                mask = cv2.fillPoly(mask, p, color=(255, 255, 255))
            name = name.replace('.jpg', '')
            process_image('/tmp/' + name, name, img, mask, args.out_dir)
            if i % 1000 == 0:
                print(f'Proccesing: {i} of {count}')
            i += 1


files_counter = {}


def process_image(face_bboxes_file, out_file_name, img, mask, save_dir):
    if os.path.exists(face_bboxes_file + '.npy'):
        boxes = np.load(face_bboxes_file + '.npy')
    else:
        inference_frame = cv2.resize(img, tuple(face_input_shape[:-3:-1]), interpolation=cv2.INTER_LINEAR)
        inference_frame = np.transpose(inference_frame, [2, 0, 1]).reshape(face_input_shape)
        outputs = face_drv.predict({face_input_name: inference_frame})
        output = outputs[face_output_name]
        output = output.reshape(-1, 7)
        bboxes_raw = output[output[:, 2] > 0.7]
        # Extract 5 values
        boxes = bboxes_raw[:, 3:7]
        confidence = np.expand_dims(bboxes_raw[:, 2], axis=0).transpose()
        boxes = np.concatenate((boxes, confidence), axis=1)
        np.save(face_bboxes_file, boxes)
    xmin = boxes[:, 0] * img.shape[1]
    xmax = boxes[:, 2] * img.shape[1]
    ymin = boxes[:, 1] * img.shape[0]
    ymax = boxes[:, 3] * img.shape[0]
    xmin[xmin < 0] = 0
    xmax[xmax > img.shape[1]] = img.shape[1]
    ymin[ymin < 0] = 0
    ymax[ymax > img.shape[0]] = img.shape[0]
    boxes = []
    for i in range(len(xmin)):
        boxes.append(np.array([int(xmin[i]), int(ymin[i]), int(xmax[i]), int(ymax[i])], np.int32))
    for b in boxes:
        h = b[3] - b[1]
        w = b[2] - b[0]
        xmin = max(b[0] - w, 0)
        xmax = min(b[2] + w, mask.shape[1])
        ymin = max(b[1] - h, 0)
        ymax = min(b[3] + h, mask.shape[0])
        fm = mask[b[1]:b[3], b[0]:b[2]]
        fm[fm > 0] = 1
        a = np.sum(fm) / h / w
        if a < 0.1:
            continue
        fm = mask[ymin:ymax, xmin:xmax]
        fi = img[ymin:ymax, xmin:xmax, :]
        si = cv2.resize(fi, (320, 320))
        fm = cv2.resize(fm, (320, 320))
        fm[fm > 0] = 255
        input_trimap = generate_trimap(fm)
        input_trimap = np.expand_dims(input_trimap.astype(np.float32), 2)
        si = si.astype(np.float32)
        si = si[:, :, ::-1] - g_mean
        outputs = mat_drv.predict(
            {'input': np.expand_dims(si, axis=0), 'trimap': np.expand_dims(input_trimap, axis=0)})
        rm = outputs['output'][0] * 255
        rm = np.reshape(rm, (320, 320))
        rm = np.clip(rm, 0, 255)
        rm = rm.astype(np.uint8)
        rm = cv2.resize(rm, (fi.shape[1], fi.shape[0]))
        ni = files_counter.get(out_file_name, 0)
        ni += 1
        files_counter[out_file_name] = ni
        fname = f'{out_file_name}-{ni}.png'
        cv2.imwrite(os.path.join(save_dir, 'images', os.path.basename(fname)), fi)
        cv2.imwrite(os.path.join(save_dir, 'masks', os.path.basename(fname)), rm)

        rm = cv2.GaussianBlur(rm, (21, 21), 11)
        rm = np.expand_dims(rm, axis=2)
        result = np.concatenate([fi, rm], axis=2)
        cv2.imwrite(os.path.join(save_dir, 'preview', fname), result)


if __name__ == '__main__':
    logging.getLogger().setLevel('INFO')
    parser = argparse.ArgumentParser(
        add_help=False
    )
    parser.add_argument('--src_dir', type=str, default='./coco')
    parser.add_argument('--out_dir', type=str, default='./data_tmp')
    parser.add_argument('--limit_pic', type=int, default=10)
    parser.add_argument('--model_dir', type=str, default='./models')
    args = parser.parse_args()
    start_coco(args)
