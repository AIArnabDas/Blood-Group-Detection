import serial
import numpy as np
import cv2

# Open serial connection (Update COM port accordingly)
ser = serial.Serial('COM3', 57600, timeout=2)  # Use '/dev/ttyUSB0' for Linux

# Command to capture fingerprint image (Modify based on scanner's datasheet)
capture_cmd = b'\xEF\x01\xFF\xFF\xFF\xFF\x01\x00\x03\x01\x00\x05'  
ser.write(capture_cmd)

# Read response
response = ser.read(12)  # Read header response

if response:
    print("Fingerprint Captured! Reading Image...")

    # Read image data (Size depends on scanner)
    image_data = ser.read(36864)  # Adjust based on fingerprint scanner specs

    # Save raw image data
    with open("fingerprint.raw", "wb") as raw_file:
        raw_file.write(image_data)

    print("Raw image saved as fingerprint.raw")

    # Convert raw image to a NumPy array
    fingerprint_array = np.frombuffer(image_data, dtype=np.uint8)

    # Reshape based on fingerprint image dimensions (Check scanner specs)
    fingerprint_image = fingerprint_array.reshape((288, 256))  # Adjust for your scanner

    # Save as JPEG
    cv2.imwrite("fingerprint.jpeg", fingerprint_image)
    print("Fingerprint image saved as fingerprint.jpeg")

else:
    print("Failed to capture fingerprint.")

# Close serial connection
ser.close()
