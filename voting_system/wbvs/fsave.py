from selenium import webdriver 
import time
import os 
import requests
import random
from PIL import Image 
import io 
import base64
#from bytearray import byte_data
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options

def fsave() :
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
    #time.sleep(5)
    button = driver.find_elements(By.TAG_NAME, "input")
    print("button -----------------")
    button[4].click()
    print(button[4])
    time.sleep(6)
    images = driver.find_element(By.ID,"FPImage1")
    print("image ---------",images)
    src=images.get_attribute("src")
    src =src.replace("data:image/bmp;base64,","")

    print("src-----------",src)

    folder = "E:/OpenSource/Web-Based-Voting-System/voting_system/fp-pics"
    os.chdir(folder)
    #if not os.path.isdir(folder): os.makedirs(folder)
    b = base64.b64decode(src)
    img = Image.open(io.BytesIO(b))
    img.show()
    nameofim = str(random.randint(0,1000))
    img.save(nameofim+".bmp")

    return nameofim
#src.save(str(random.randint(0,1000)),format ="BMP")
#with open(str(random.randint(0,1000))+'.BMP','wb') as f :
#    im = requests.get(src)
#    f.write(im.content)

#for ele in button :
#    try:
#        ele.click()
#        print(button.index(ele))
#    except :
#        continue
#driver.quit()