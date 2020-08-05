import cv2 as cv
import numpy as np


def find_subimage(large_image_path, sub_img_path, threshold=0.99, debug=False):
    """
    :param large_image_path: путь к большому изображению, где будем искать
    :param sub_img_path: путь к малому изображению, что будем искать
    :param threshold: вероятность совпадения в процентах, где 1==100%
    :param debug: Открывает opencv окно для отображения результата поиска с обводкой
    :return tuple: Кортеж (x,y), где координаты указываются относительно левого верхнего угла sub_img_path
    """
    img_rgb = cv.imread(large_image_path)
    img_gray = cv.cvtColor(img_rgb, cv.COLOR_BGR2GRAY)

    template = cv.imread(sub_img_path, cv.IMREAD_GRAYSCALE)

    res = cv.matchTemplate(img_gray, template, cv.TM_CCOEFF_NORMED)
    loc = np.where(res >= threshold)
    if debug:
        w, h = template.shape[::-1]
        for pt in zip(*loc):
            pt = (pt[1], pt[0])
            cv.rectangle(img_rgb, pt, (pt[0] + w, pt[1] + h), (0, 0, 255), 4)
        cv.imshow("result", img_rgb)
        cv.waitKey(0)
        cv.destroyAllWindows()
    return (loc[1][0], loc[0][0])
