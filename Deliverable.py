print('Loading Program...................')
from keras.models import load_model
from cv2 import imread,resize
import numpy as np
from time import sleep
model = load_model('Cancer_CNN_weights.keras')
print("+------------------------------------+")
print('Welcome to the Lung Cancer Detection System')
print("+------------------------------------+")

cancer = False
while True:
    print('Input file path of cancer image you would like to check for lung cancer: ')
    image_path = input()
    print('Checking for signs of Lung Cancer............')
    for i in range(0, 3):
        sleep(1)
        print('.', end='', flush=True)
    print()
    img = imread(image_path)
    img = resize(img, (256,256))
    img = np.expand_dims(img, axis=0)
    yhat = model.predict(img, verbose=0)
    max_idx = np.argmax(yhat)
    
    if max_idx in [2,4]:
        cancer = True
        print(f'!!!!!!Possible signs of Lung Cancer detected!!!!!!')
        print('WARNING: You should probably get that checked by a doctor as soon as possible')
    else:
        print('No signs of Cancer detected need to consult a doctor for further diagnosis')
    print('Would you like to check another image? (y/n)')
    choice = input()
    if choice.lower() == 'n' or choice.lower() == 'no':
        break
    print('------------------------------------')
print('Thank you for using the Lung Cancer Detection System')
if cancer:
    print('Like seriously, go see a doctor')
else:
    print('Good Job and Stay Healthy!')