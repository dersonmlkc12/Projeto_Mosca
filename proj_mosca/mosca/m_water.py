import cv2
import numpy as np
from scipy import ndimage
from skimage.feature import peak_local_max
from skimage.morphology import watershed

def waterb(contor):
    ret, thresh = cv2.threshold(contor, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    d = ndimage.distance_transform_edt(thresh)

    localMax = peak_local_max(d,indices=False, min_distance=2,labels=thresh)

    markers = ndimage.label(localMax, structure=np.ones((3,3)))[0]
    labels = watershed(-d,markers,mask=thresh)




    return labels