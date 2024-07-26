import requests
import json
import cv2

addr = 'http://localhost:5000'
test_url = addr + '/images/aesthetic2'

# prepare headers for http request
content_type = 'image/jpeg'
headers = {'content-type': content_type}

img = cv2.imread('mermaid.jpeg')
# encode image as jpeg
_, img_encoded = cv2.imencode('.jpg', img)
# send http request with image and receive response
response = requests.post(test_url, data=img_encoded.tobytes(), headers=headers, params={'prompt': "a blue butterfly with black wings and a white background with a white border with the words, the butterfly is a blue butterfly, amano, p, a hologram, rayonism"})
# decode response
print(json.loads(response.text))

# expected output: {u'message': u'image received. size=124x124'}
