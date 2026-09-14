from pyfingerprint.pyfingerprint import PyFingerprint
import os

try:
    # Initialize the fingerprint sensor
    sensor = PyFingerprint('COM5', 57600, 0xFFFFFFFF, 0x00000000)
    if not sensor.verifyPassword():
        raise Exception('The given fingerprint sensor password is wrong!')

    print('Waiting for finger...')
    while not sensor.readImage():
        pass

    print('Finger detected! Downloading image...')
    # Save the scanned image to a file
    image_destination = 'fingerprint.jpeg'
    sensor.downloadImage(image_destination)
    print(f'Fingerprint image saved as {image_destination}')

except Exception as e:
    print(f'Error: {e}')