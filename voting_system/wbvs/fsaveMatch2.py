from selenium import webdriver
import time
import cv2
import sys
import numpy
import matplotlib.pyplot as plt
from enhance import *
from skimage.morphology import skeletonize, thin
import time
import os
import requests
import random
from PIL import Image
import io
import base64

# from bytearray import byte_data
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options


def removedot(invertThin):
    temp0 = numpy.array(invertThin[:])
    temp0 = numpy.array(temp0)
    temp1 = temp0 / 255
    temp2 = numpy.array(temp1)
    temp3 = numpy.array(temp2)

    enhanced_img = numpy.array(temp0)
    filter0 = numpy.zeros((10, 10))
    W, H = temp0.shape[:2]
    filtersize = 6

    for i in range(W - filtersize):
        for j in range(H - filtersize):
            filter0 = temp1[i : i + filtersize, j : j + filtersize]

            flag = 0
            if sum(filter0[:, 0]) == 0:
                flag += 1
            if sum(filter0[:, filtersize - 1]) == 0:
                flag += 1
            if sum(filter0[0, :]) == 0:
                flag += 1
            if sum(filter0[filtersize - 1, :]) == 0:
                flag += 1
            if flag > 3:
                temp2[i : i + filtersize, j : j + filtersize] = numpy.zeros(
                    (filtersize, filtersize)
                )

    return temp2


def get_descriptors(img):
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    img = clahe.apply(img)
    # img = image_enhance.image_enhance(img)
    img = numpy.array(img, dtype=numpy.uint8)
    # Threshold
    ret, img = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    # Normalize to 0 and 1 range
    img[img == 255] = 1

    # Thinning
    skeleton = skeletonize(img)
    skeleton = numpy.array(skeleton, dtype=numpy.uint8)
    skeleton = removedot(skeleton)
    # Harris corners
    harris_corners = cv2.cornerHarris(img, 3, 3, 0.04)
    harris_normalized = cv2.normalize(
        harris_corners, 0, 255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32FC1
    )
    threshold_harris = 125
    # Extract keypoints
    keypoints = []
    for x in range(0, harris_normalized.shape[0]):
        for y in range(0, harris_normalized.shape[1]):
            if harris_normalized[x][y] > threshold_harris:
                keypoints.append(cv2.KeyPoint(y, x, 1))
    # Define descriptor
    orb = cv2.ORB_create()
    # Compute descriptors
    _, des = orb.compute(img, keypoints)
    return (keypoints, des)


def matchmain(name, folder,noi):
    result =0 
    # image_name = sys.argv[1]
    img1 = cv2.imread("E:/OpenSource/Web-Based-Voting-System/voting_system/vg/"+ name, cv2.IMREAD_GRAYSCALE)
    kp1, des1 = get_descriptors(img1)

    # image_name = sys.argv[2]
    noi = noi+".bmp"
    filename = ""
    for file in [file for file in os.listdir(folder)]:
        print(name,noi,file)
        
        if noi != "all.bmp" and file !=noi : continue
        if noi== "all.bmp" and file == name : continue
        print(file)
        # fingerprint_image = cv2.imread("compare/"+file)
        img2 = cv2.imread(folder + "/" + file, cv2.IMREAD_GRAYSCALE)
        kp2, des2 = get_descriptors(img2)

        # Matching between descriptors
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        matches = sorted(bf.match(des1, des2), key=lambda match: match.distance)
        # Plot keypoints
        img4 = cv2.drawKeypoints(img1, kp1, outImage=None)
        img5 = cv2.drawKeypoints(img2, kp2, outImage=None)

        # Calculate score
        score = 0
        for match in matches:
            score += match.distance
        score_threshold = 42
        if score / len(matches) < score_threshold:
            print("Fingerprint matches.")
            f, axarr = plt.subplots(1, 2)
            axarr[0].imshow(img4)
            axarr[1].imshow(img5)
            #plt.show()
            # Plot matches
            img3 = cv2.drawMatches(img1, kp1, img2, kp2, matches, flags=2, outImg=None)
            #plt.imshow(img3)
            #plt.show()
            result =1
            break

        else:
            result = 0
            print("Fingerprint does not match.")
    return result

def fsm(noi) :

    PATH = "C:\Program Files (x86)\chromedriver.exe"
    WINDOW_SIZE = "1920,1080"

    chrome_options = Options()
    chrome_options.add_argument("--headless")
    chrome_options.add_argument("--window-size=%s" % WINDOW_SIZE)
    chrome_options.binary_location = PATH
    driver = webdriver.Chrome(PATH)
    driver.get("https://webapi.secugen.com/Demo1")

    print("info------------------------------------->>>>>>>>>>>>")
    print(driver.title)
    # time.sleep(5)
    button = driver.find_elements(By.TAG_NAME, "input")
    print("button -----------------")
    button[4].click()
    print(button[4])
    time.sleep(6)
    images = driver.find_element(By.ID, "FPImage1")
    print("image ---------", images)
    src = images.get_attribute("src")
    src = src.replace("data:image/bmp;base64,", "")

    #print("src-----------", src)

    folder = "E:/OpenSource/Web-Based-Voting-System/voting_system/vg"
    os.chdir(folder)
    # if not os.path.isdir(folder): os.makedirs(folder)
    b = base64.b64decode(src)
    img = Image.open(io.BytesIO(b))
    name = str(random.randint(0, 1000)) + ".bmp"
    img.save(name)
    driver.quit()
    res = 0
    try:
        res = matchmain(name, folder,"all")
        print("executed for vg")
        #if res == 0 :   
        #    folder = "E:/OpenSource/Web-Based-Voting-System/voting_system/fp-pics"
        #    res = matchmain(name, folder,noi)
       #     print("executed for fpp")
        #    if res == 1 : res =0
        #    else : res = 1
    except:
        raise
    return res

# img.show()

# src.save(str(random.randint(0,1000)),format ="BMP")
# with open(str(random.randint(0,1000))+'.BMP','wb') as f :
#    im = requests.get(src)
#    f.write(im.content)

# for ele in button :
#    try:
#        ele.click()
#        print(button.index(ele))
#    except :
#        continue
# driver.quit()
