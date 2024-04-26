import cv2
import face_recognition
import os

# Function to load training data
def load_training_data(train_dir):
    images = []
    labels = []
    for name in os.listdir(train_dir):
        person_dir = os.path.join(train_dir, name)
        for image_name in os.listdir(person_dir):
            image_path = os.path.join(person_dir, image_name)
            image = face_recognition.load_image_file(image_path)
            encoding = face_recognition.face_encodings(image)[0]  # Assuming only one face per image
            images.append(encoding)
            labels.append(name)
    return images, labels

# Function to train the model
def train_model(images, labels):
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.train(images, labels)
    return recognizer

# Function to capture and save images for training
def capture_images(name, num_images=10):
    capture = cv2.VideoCapture(0)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    image_dir = os.path.join('training_data', name)
    os.makedirs(image_dir, exist_ok=True)
    count = 0
    while count < num_images:
        ret, frame = capture.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)
        
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

        cv2.imshow('Capture', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        elif cv2.waitKey(1) & 0xFF == ord('c'):
            count += 1
            image_path = os.path.join(image_dir, f'image_{count}.jpg')
            cv2.imwrite(image_path, frame)
            print(f"Image {count} captured!")
            if count >= num_images:
                break  # Stop capturing after the specified number of images
    capture.release()
    cv2.destroyAllWindows()

# Function to recognize faces
def recognize_faces(test_image_path, recognizer, labels):
    test_image = cv2.imread(test_image_path)
    rgb_test_image = cv2.cvtColor(test_image, cv2.COLOR_BGR2RGB)
    face_locations = face_recognition.face_locations(rgb_test_image)

    for top, right, bottom, left in face_locations:
        cv2.rectangle(test_image, (left, top), (right, bottom), (0, 255, 0), 2)
        face_image = test_image[top:bottom, left:right]
        gray_face = cv2.cvtColor(face_image, cv2.COLOR_BGR2GRAY)
        label, confidence = recognizer.predict(gray_face)
        if confidence < 100:  # You may need to adjust this threshold
            name = labels[label]
        else:
            name = "Unknown"
        cv2.putText(test_image, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)

    cv2.imshow('Face Recognition', test_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Main function
def main():
    while True:
        choice = input("Choose an option:\n1. Train\n2. Detect\n3. Exit\n")

        if choice == '1':
            name = input("Enter your name: ")
            capture_images(name)
            print("Training...")
            images, labels = load_training_data('training_data')
            recognizer = train_model(images, labels)
            print("Training complete!")

        elif choice == '2':
            test_image_path = input("Enter the path to the test image: ")
            images, labels = load_training_data('training_data')
            recognizer = train_model(images, labels)
            recognize_faces(test_image_path, recognizer, labels)

        elif choice == '3':
            break

        else:
            print("Invalid choice.")

if __name__ == "__main__":
    main()
