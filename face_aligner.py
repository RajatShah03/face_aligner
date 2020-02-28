'''
    Script For Face Alignment
    using dlib and openCV
'''

import cv2
import numpy as np
from PIL import Image
import dlib
import matplotlib.pyplot as plt
import matplotlib.patches as patches

def vis_face_bbox(img, bbox):
    x, y, w, h = bbox
    fig, ax = plt.subplots(1)
    ax.imshow(img)
    r = patches.Rectangle((x, y), w, h, linewidth=2, edgecolor='red', facecolor='none')
    ax.add_patch(r)
    plt.show()
    input('ee')

def shape_to_normal(shape):
    shape_normal = []
    for i in range(5):
        shape_normal.append((i, (shape.part(i).x, shape.part(i).y)))
    return shape_normal

def get_eyes_nose_dlib(shape):
    nose = shape[4][1]
    left_eye_x = int(shape[3][1][0] + shape[2][1][0]) // 2
    left_eye_y = int(shape[3][1][1] + shape[2][1][1]) // 2
    right_eye_x = int(shape[1][1][0] + shape[0][1][0]) // 2
    right_eye_y = int(shape[1][1][1] + shape[0][1][1]) // 2
    return nose, (left_eye_x, left_eye_y), (right_eye_x, right_eye_y)

def distance(a, b):
    return np.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)

def rotate_point(origin, point, angle):
    ox, oy = origin
    px, py = point

    qx = ox + np.cos(angle)*(px - ox) - np.sin(angle)*(py - oy)
    qy = oy + np.sin(angle)*(px - ox) + np.cos(angle)*(py - oy)
    return qx, qy

def is_between(p1, p2, p3, q):
    c1 = (p2[0] - p1[0])*(q[1] - p1[1]) - (p2[1] - p1[1])*(q[0] - p1[0])
    c2 = (p3[0] - p2[0])*(q[1] - p2[1]) - (p3[1] - p2[1])*(q[0] - p2[0])
    c3 = (p1[0] - p3[0])*(q[1] - p3[1]) - (p1[1] - p3[1])*(q[0] - p3[0])

    if (c1 < 0 and c2 < 0 and c3 < 0) or (c1 > 0 and c2 > 0 and c3 > 0):
        return True
    else:
        return False

def cosine_formula(l1, l2, l3):
    a, b, c = l3, l2, l1
    cos_A = (b**2 + c**2 - a**2) / (2*b*c)
    return cos_A

def crop_face(img, bbox):
    x, y, w, h = bbox
    return img[y:h, x:w]

def face_aligner(img):
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor('shape_predictor_5_face_landmarks.dat')
   
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    pred_shape = None
    rects = detector(gray, 0)
    if len(rects) > 0:
        for rect in rects:
            x = rect.left()
            y = rect.top()
            w = rect.right()
            h = rect.bottom()
            pred_shape = predictor(gray, rect)

    shape = shape_to_normal(pred_shape)
    nose, left_eye, right_eye = get_eyes_nose_dlib(shape)

    center_forehead = ((left_eye[0] + right_eye[0]) // 2, (left_eye[1] + right_eye[1]) // 2)

    center_facetop = (int((x + w) / 2), int((y + y) / 2))

    L1 = distance(center_forehead, nose)
    L2 = distance(center_facetop, nose)
    L3 = distance(center_facetop, center_forehead)

    cos_a = cosine_formula(L1, L2, L3)
    angle = np.arccos(cos_a)

    rotated_point = rotate_point(nose, center_forehead, angle)

    rotated_point = (int(rotated_point[0]), int(rotated_point[1]))

    if is_between(nose, center_forehead, center_facetop, rotated_point):
        final_angle = np.degrees(-angle)
    else:
        final_angle = np.degrees(angle)

    img = Image.fromarray(img)
    final_img = np.array(img.rotate(final_angle))
    cropped_img = crop_face(final_img, (x, y, w, h))
    return final_img, cropped_img