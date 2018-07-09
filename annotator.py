import cv2
import numpy as np
from PIL import Image,ImageFont,ImageDraw

font = ImageFont.truetype('C:/Windows/Fonts/simhei.ttf',size = 30,encoding= "utf-8")

def annotator(img, class_box_scores ,upsize = 30,upsize_font = 27,lenscale = 20):
    # 每一个类，多个box，多个scores
    len_classes = len(class_box_scores)

    textinfos = []
    
    colors = [(0,0,255),(255,0,0),(0,255,0),(255,255,0),(255,0,255),(0,255,255)]
    i = 0
    for classname in class_box_scores:
        box_scores = class_box_scores[classname]
        color = colors[i % len(colors)]
        i += 1
        for box_score in box_scores:
            cv2.rectangle(img,(box_score[1],box_score[2]),(box_score[3],box_score[4]),color,3)
            text = classname + " : " + str("%.2f"%(box_score[0]))
            len_text = len(text)
            # background 
            
            cv2.rectangle(img,(box_score[1],box_score[2]-upsize),(box_score[1] + len_text * lenscale ,box_score[2]),color,-1) # thickness = -1 
            textinfo = [(box_score[1],box_score[2]),text]
            textinfos.append(textinfo)

        # PIL Image <- img
    img = Image.fromarray(img[:,:,(2,1,0)])
    texter = ImageDraw.Draw(img)
    for textinfo in textinfos:
        x,y = textinfo[0]
        text = textinfo[1]
        texter.text((x,y-upsize_font),text,fill=(255,255,255),font = font)
    return np.array(img)[:,:,::-1]


if __name__ == "__main__":
    img = cv2.imread("test.jpg")
    class_box_scores = {"枪支":[[0.89,300,300,500,500],[0.72,210,342,644,489]],"迷药":[[0.56,200,300,600,600]]}
    img = annotator(img,class_box_scores)
    cv2.imshow("look",img)
    cv2.waitKey(0)
