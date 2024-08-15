import cv2
import numpy as np
import os


def distance(v1, v2):
    return np.sqrt(((v1 - v2) ** 2).sum())  # Euclidean distance between two vectors

def knn(train, test, k=5):
    dist = []

    for i in range(train.shape[0]):
        ix = train[i, :-1]
        iy = train[i, -1]

        d = distance(test, ix)
        dist.append([d, iy])
    
    # Sort by distance and select the top k
    dk = sorted(dist, key=lambda x: x[0])[:k]
    labels = np.array(dk)[:, -1]

    # Find the most frequent label
    output = np.unique(labels, return_counts=True)
    index = np.argmax(output[1])
    
    return output[0][index]  # Return the label with the highest count


cap = cv2.VideoCapture(0)  # Start the webcam

face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_alt.xml")  # Load the face detection model

dataset_path = './data/'
face_data = []
labels = []

class_id = 0
names = {}


# Load the training data and map class IDs to names
for fx in os.listdir(dataset_path):
    if fx.endswith('.npy'):
        name = fx[:-4]  # Extract name from the filename
        names[class_id] = name 
        data_item = np.load(dataset_path + fx)
        face_data.append(data_item)

        target = class_id * np.ones((data_item.shape[0]),)
        class_id += 1
        labels.append(target)

face_dataset = np.concatenate(face_data, axis=0)  # Combine all face data into one array
face_labels = np.concatenate(labels, axis=0).reshape((-1, 1))  # Combine all labels

print(face_dataset.shape)
print(face_labels.shape)

trainset = np.concatenate((face_dataset, face_labels), axis=1)  # Create the training set
print(trainset.shape)


while True:
    ret, frame = cap.read()
    if ret == False:
        continue

    faces = face_cascade.detectMultiScale(frame, 1.3, 5)  # Detect faces in the frame

    for face in faces:
        x, y, w, h = face

        offset = 10
        face_section = frame[y - offset:y + h + offset, x - offset:x + w + offset]
        face_section = cv2.resize(face_section, (100, 100))

        out = knn(trainset, face_section.flatten())  # Predict the label using KNN

        pred_name = names[int(out)]  # Get the predicted name
        cv2.putText(frame, pred_name, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 255), 2)  # Draw a rectangle around the face

    cv2.imshow("Faces", frame)  # Display the frame

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
