import face_recognition
from sklearn import svm
import os
import joblib

# Training function
def train_classifier(train_dir, model_save_path, clf_type='svm'):
    # Lists to store face encodings and labels
    encodings = []
    labels = []

    # Loop through each person in the training directory
    for person in os.listdir(train_dir):
        person_dir = os.path.join(train_dir, person)

        # Loop through each image of the current person
        for image_file in os.listdir(person_dir):
            image_path = os.path.join(person_dir, image_file)

            # Load the image
            image = face_recognition.load_image_file(image_path)

            # Find all faces in the image
            face_locations = face_recognition.face_locations(image)
            
            # Ensure there's exactly one face in the image
            if len(face_locations) == 1:
                # Get face encodings for the face in the image
                face_encodings = face_recognition.face_encodings(image, known_face_locations=face_locations)

                # Add the encoding and label to the lists
                encodings.append(face_encodings[0])
                labels.append(person)

    # Choose the classifier type
    if clf_type == 'svm':
        clf = svm.SVC(gamma='scale')
    # Add other classifier types if desired (e.g., KNN)
    
    # Train the classifier
    clf.fit(encodings, labels)

    # Save the trained classifier
    joblib.dump(clf, model_save_path)

    print(f"Classifier saved at {model_save_path}")

if __name__  ==  "__main__":
    # Set training directory and model save path
    train_directory = "train_dir"
    model_save_path = "face_recognition_model.joblib"

    # Train the classifier
    train_classifier(train_directory, model_save_path)