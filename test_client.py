import requests
import json
import cv2

addr = 'http://localhost:5000'
test_url = addr + '/images/clip'

# prepare headers for http request
content_type = 'image/jpeg'
headers = {'content-type': content_type}

img = cv2.imread('butterfly.jpeg')
# encode image as jpeg
_, img_encoded = cv2.imencode('.jpg', img)
# send http request with image and receive response
response = requests.post(test_url, data=img_encoded.tobytes(), headers=headers, params={'prompt': "a photo of a butterfly"})
# decode response
print(json.loads(response.text))

# expected output: {u'message': u'image received. size=124x124'}
