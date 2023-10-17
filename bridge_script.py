import sys

import requests
import json
import cv2

addr = 'http://localhost:5000'
test_url = addr + '/images/clip'

assert len(sys.argv) == 3

# Get image path and prompt
image_path = sys.argv[1]
prompt = sys.argv[2]

# prepare headers for http request
content_type = 'image/jpeg'
headers = {'content-type': content_type}

img = cv2.imread(image_path)
# encode image as jpeg
_, img_encoded = cv2.imencode('.jpg', img)
# send http request with image and receive response
response = requests.post(test_url, data=img_encoded.tobytes(), headers=headers, params={'prompt': prompt})
# decode response
response = json.loads(response.text)

print(response["clip"])
