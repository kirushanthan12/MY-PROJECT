import time
from PIL import Image
from naoqi import ALProxy
import numpy as np
import cv2
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization

# CIFAR-10 class names
cifar10_class_names = [
    'airplane', 'automobile', 'bird', 'cat', 'deer',
    'dog', 'frog', 'horse', 'ship', 'truck'
]

# Path to saved model weights
model_weights_path = 'my_model_.weights.h5'

# Build the CNN model (same architecture as training)
model = Sequential()

# Block 1
model.add(Conv2D(32, (3, 3), padding='same', activation='relu', input_shape=(32, 32, 3)))
model.add(BatchNormalization())
model.add(Conv2D(32, (3, 3), activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

# Block 2
model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
model.add(BatchNormalization())
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

# Block 3
model.add(Conv2D(128, (3, 3), padding='same', activation='relu'))
model.add(BatchNormalization())
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

# Fully connected layers
model.add(Flatten())
model.add(Dense(512, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.5))
model.add(Dense(10, activation='softmax'))

# Load weights
model.load_weights(model_weights_path)
print "Successfully loaded model weights for Nao integration."

# Nao robot connection info
nao_ip = "172.18.16.45"
nao_port = 9559

try:
    # Connect to Naoqi services
    video_service = ALProxy("ALVideoDevice", nao_ip, nao_port)
    tts_service = ALProxy("ALTextToSpeech", nao_ip, nao_port)
    print "Connected to Naoqi services."

    # Set volume
    initial_volume = tts_service.getVolume()
    tts_service.setVolume(1.0)  # Max volume

    # Nao introduction
    intro_text = ("Hello, I'm Nao. This is an image classification demonstration "
                  "using a custom CNN model trained on the CIFAR-10 dataset.")
    print "Nao is about to say: {}".format(intro_text)
    tts_service.say(intro_text)
    tts_service.say("Presented by Kirushanthan.")
    tts_service.say("The classes I can recognize are airplane, automobile, bird, cat, deer, dog, frog, horse, ship, and truck.")
    print "Nao finished introduction."

    # Subscribe to camera
    camera_client_name = "cifar10_classifier_client"
    resolution = 2  # VGA
    color_space = 11  # RGB
    fps = 5
    camera_index = 0  # Top camera

    print "Subscribing to Nao camera..."
    video_client = video_service.subscribeCamera(camera_client_name, camera_index, resolution, color_space, fps)
    print "Camera subscribed."

    print "\nStarting continuous image capture and classification. Press 'q' to exit."

    while True:
        nao_image = video_service.getImageRemote(video_client)

        if nao_image is None:
            print "Warning: Failed to capture image from Nao. Retrying..."
            time.sleep(0.1)
            continue

        image_width = nao_image[0]
        image_height = nao_image[1]
        image_array = nao_image[6]  # Raw bytes
        image_string = str(bytearray(image_array))

        # Convert to PIL image
        pil_image = Image.frombytes("RGB", (image_width, image_height), image_string)

        # Show image with OpenCV (convert RGB to BGR)
        cv_image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
        cv2.imshow("Nao Camera Feed", cv_image)

        # Resize to 32x32 for CNN
        try:
            resized_image = pil_image.resize((32, 32), Image.LANCZOS)
        except AttributeError:
            resized_image = pil_image.resize((32, 32), Image.ANTIALIAS)

        img_array = np.array(resized_image).astype('float32') / 255.0
        img_array = np.expand_dims(img_array, axis=0)  # Shape (1, 32, 32, 3)

        # Predict class
        predictions = model.predict(img_array)
        predicted_class_index = np.argmax(predictions[0])
        predicted_class_name = cifar10_class_names[predicted_class_index]
        print "Predicted class name: {}".format(predicted_class_name)

        # Nao says prediction
        tts_service.say("I see a {}.".format(predicted_class_name))

        # Exit on 'q' press
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            print "Exiting continuous capture loop."
            break

        time.sleep(1.5)

    # Cleanup
    video_service.unsubscribe(video_client)
    tts_service.setVolume(initial_volume)
    cv2.destroyAllWindows()

except RuntimeError as e:
    print "Could not connect to Naoqi. Check IP, port, or if Naoqi is running. Error: {}".format(e)
except Exception as e:
    print "An error occurred during Nao integration: {}".format(e)
