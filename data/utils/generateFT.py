import cv2
import numpy as np
def generate_FT(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    f = np.fft.fft2(image)
    fshift = np.fft.fftshift(f)
    fimg = np.log(np.abs(fshift)+1)
    maxx = -1
    minn = 100000
    for i in range(len(fimg)):
        if maxx < max(fimg[i]):
            maxx = max(fimg[i])
        if minn > min(fimg[i]):
            minn = min(fimg[i])
    fimg = (fimg - minn+1) / (maxx - minn+1)
    return fimg

if __name__ == '__main__':
    img = cv2.imread("/home/aimenext/luantt/face_liveness_detecion/mixup.png")
    fimg = generate_FT(img)
    print(fimg)
    print(fimg.shape)