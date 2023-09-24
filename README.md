"# PaddleOCR_20230801" 


!pip install -q paddlepaddle

#libssl 설치
!wget http://archive.ubuntu.com/ubuntu/pool/main/o/openssl/libssl1.1_1.1.0g-2ubuntu4_amd64.deb
!sudo dpkg -i libssl1.1_1.1.0g-2ubuntu4_amd64.deb

!pip install paddleocr

from PIL import Image

img = Image.open('ocrrr.png').convert('RGB')
img

from paddleocr import PaddleOCR, draw_ocr

ocr = PaddleOCR(lang='korean')
img_path = 'ocrrr.png'
result = ocr.ocr(img_path , cls=False)
result
     
[2023/09/24 04:50:48] ppocr DEBUG: dt_boxes num : 2, elapsed : 0.5333883762359619
[2023/09/24 04:50:49] ppocr DEBUG: rec_res num  : 2, elapsed : 0.7195372581481934
[[[[[164.0, 112.0], [334.0, 115.0], [334.0, 164.0], [163.0, 162.0]],
   ("이그리트'", 0.9998728632926941)],
  [[[317.0, 112.0], [388.0, 112.0], [388.0, 165.0], [317.0, 165.0]],
   ("'다", 0.9999555349349976)]]]

!wget -q https://github.com/kairess/toy-datasets/raw/master/NanumSquareNeo-Variable.ttf
     
boxes = [temp[0] for temp in result[0]]
texts = [temp[1][0] for temp in result[0]]
scores = [temp[1][1] for temp in result[0]]
result_np = draw_ocr(img, boxes, texts, scores, font_path='NanumSquareNeo-Variable.ttf')
result_np = Image.fromarray(result_np)

result_np
     
result[0][0][1]

("이그리트'", 0.9998728632926941)

boxes = []

for i,r in enumerate(result[0]):
    x1,y1 = r[0][0]
    x2,y2 = r[0][2]

    w = x2-x1
    h = y2 - y1

    text, conf = r[1]

    boxes.append([int(x1) , int(y1), int(w), int(h), text, conf, i])

boxes

[[164, 112, 170, 52, "이그리트'", 0.9998728632926941, 0],
 [317, 112, 71, 53, "'다", 0.9999555349349976, 1]]

import numpy as np
from sklearn.cluster import DBSCAN

#박스의 중심점 찾기
def calculate_center(box):
    center_x = box[0] + box[2]/2
    center_y = box[1] + box[3]/2
    return np.array([center_x,center_y])

def cluster_boxes(boxes, eps):
    centers = np.array([calculate_center(box) for box in boxes])

    clustering = DBSCAN(eps = eps, min_samples=1).fit(centers)
    labels = clustering.labels_
    print(labels)

    clusters = {}
    for i, label in enumerate(labels):
        if label in clusters:
            clusters[label].append(i)
        else :
            clusters[label] = [i]

    return list(clusters.values())

clusters = cluster_boxes(boxes, 100)
     
[0 1]

clusters
     
[[0], [1]]

ocr_result = []

for c in clusters:
    sub_result = []

    for i, box in enumerate(boxes):
        if i in c:
            sub_result.append(box)

    ocr_result.append(sub_result)
ocr_result
     
[[[164, 112, 170, 52, "이그리트'", 0.9998728632926941, 0]],
 [[317, 112, 71, 53, "'다", 0.9999555349349976, 1]]]

ocr_result[0]
     
[[164, 112, 170, 52, "이그리트'", 0.9998728632926941, 0]]

final_result = []

for sub_result in ocr_result:
    x1 = sub_result[0][0]
    y1 = sub_result[0][1]
    x2 = sub_result[-1][0] + sub_result[-1][2]
    y2 = sub_result[-1][1] + sub_result[-1][3]

    w = x2 - x1
    h = y2-y1

    text = ''

    for r in sub_result:
        text += r[4] + ' '

    text = text.strip()

    final_result.append([x1,y1,w,h,text])

final_result
     
[[164, 112, 170, 52, "이그리트'"], [317, 112, 71, 53, "'다"]]


!pip install translate


from translate import Translator

translator = Translator(from_lang='ko', to_lang='en')

for i,r in enumerate(final_result):
    text_en = translator.translate(r[4])

    final_result[i].append(text_en)

final_result
     
[[164, 112, 170, 52, "이그리트'", "Igrit '"], [317, 112, 71, 53, "'다", 'C']]

from PIL import Image,ImageDraw

result_img = img.copy()
draw = ImageDraw.Draw(result_img)

for box in boxes:
    x1,y1,w,h,_,_,_ = box
    x2 = x1+w
    y2 = y1+h

    draw.rectangle([(x1,y1),(x2,y2)],outline='white',fill='white')

result_img
     


from PIL import Image,ImageDraw, ImageFont
import textwrap
     

result_img2 = result_img.copy()
draw = ImageDraw.Draw(result_img2)

for r in final_result:
    x1,y1,w,h,text_ko,text_en = r

    text_position = (x1,y1)

    font = ImageFont.truetype('NanumSquareNeo-Variable.ttf',12)
    wrapped_text = textwrap.wrap(text_en,width=w/8)

    line_height = 12 * 1.2

    for line in wrapped_text:
        draw.text(text_position, line, fill='black',stroke_width=1,stroke_fill="black")
        text_position = (text_position[0],text_position[1]+line_height)

result_img2

     

