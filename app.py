import os
import cv2
import face_recognition
from sklearn import svm
import joblib
import tkinter as tk
from tkinter import messagebox

def capture_images(train_dir, name, num_images=15):
    # Create a directory for the person within the train directory
    person_dir = os.path.join(train_dir, name)  # This should create the correct path as train_dir/person_name
    
    # Check if the directory exists; if not, create it
    if not os.path.exists(person_dir):
        os.makedirs(person_dir)
    
    # Start capturing images from the camera
    cap = cv2.VideoCapture(0)
    print(f"Capturing {num_images} images for {name} in {person_dir}...")
    
    # Initialize frame counter
    count = 0
    while count < num_images:
        # Read a frame from the camera
        ret, frame = cap.read()
        if not ret:
            print("Failed to capture frame.")
            break
        
        # Convert the frame to RGB for face recognition
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Find all the face locations in the frame
        face_locations = face_recognition.face_locations(rgb_frame)

        # Draw bounding boxes around detected faces
        for (top, right, bottom, left) in face_locations:
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)

        # Display the frame with bounding boxes
        cv2.imshow('Camera', frame)

        # Save the frame with faces if there are face detections
        if face_locations:
            # Create a filename for the image
            filename = os.path.join(person_dir, f"{name}_{count}.jpg")
            
            # Save the frame as an image file
            cv2.imwrite(filename, frame)
            print(f"Image {count + 1} saved as {filename}.")
            
            # Increment the counter
            count += 1

        # Break the loop if 'q' key is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the camera and close OpenCV windows
    cap.release()
    cv2.destroyAllWindows()

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
    
    # Train the classifier
    clf.fit(encodings, labels)

    # Save the trained classifier
    joblib.dump(clf, model_save_path)

    messagebox.showinfo("Success", f"Classifier saved at {model_save_path}")

# Function to recognize faces from live camera feed
def recognize_faces_from_camera(model_path):
    # Load the trained classifier
    clf = joblib.load(model_path)
    
    # Start capturing video from the camera
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        messagebox.showerror("Error", "Failed to open camera.")
        return
    
    print("Recognizing faces. Press 'q' to quit.")
    
    while True:
        # Read a frame from the camera
        ret, frame = cap.read()
        if not ret:
            messagebox.showerror("Error", "Failed to capture frame.")
            break
        
        # Convert the frame from BGR to RGB format
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Detect all faces in the frame
        face_locations = face_recognition.face_locations(rgb_frame)
        
        # Extract face encodings from the frame
        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)
        
        # Recognize faces and draw bounding boxes with labels
        for i, face_encoding in enumerate(face_encodings):
            # Predict the name for each face
            name = clf.predict([face_encoding])[0]
            
            # Get the location of the face
            top, right, bottom, left = face_locations[i]
            
            # Draw a bounding box around the face using OpenCV
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
            
            # Add the name label above the bounding box using OpenCV
            cv2.putText(frame, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # Display the frame with bounding boxes and labels
        cv2.imshow("Recognized Faces", frame)
        
        # Break the loop if 'q' key is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Release the camera and close OpenCV windows
    cap.release()
    cv2.destroyAllWindows()


def main():
    # Create a Tkinter window
    root = tk.Tk()
    root.title("Face Recognition App")

    # Define paths and directories
    train_dir = "train_dir"
    model_save_path = "face_recognition_model.joblib"

    # Set the background color of the root window to light gray
    root.configure(bg="light gray")

    # Create entry for person name
    # Set background and text color of the label to white
    name_label = tk.Label(root, text="Person Name:", bg="light gray", fg="black")
    name_label.grid(row=0, column=0, padx=10, pady=10)
    
    # Set background and text color of the entry field to white
    # Add relief for better visibility
    name_entry = tk.Entry(root, bg="white", fg="black", relief=tk.SUNKEN)
    name_entry.grid(row=0, column=1, padx=10, pady=10)
    
    # Function to call capture_images with the provided name and training directory
    def capture_images_with_name():
        # Retrieve the person's name from the entry field
        name = name_entry.get()
        # Call capture_images with the provided name and training directory
        capture_images(train_dir, name)
    
    # Create buttons for each functionality
    capture_button = tk.Button(root, text="Capture Images", command=capture_images_with_name)
    capture_button.grid(row=1, column=0, padx=10, pady=10)
    
    train_button = tk.Button(root, text="Train Model", command=lambda: train_classifier(train_dir, model_save_path))
    train_button.grid(row=1, column=1, padx=10, pady=10)
    
    recognize_button = tk.Button(root, text="Recognize Faces", command=lambda: recognize_faces_from_camera(model_save_path))
    recognize_button.grid(row=1, column=2, padx=10, pady=10)
    
    # Start the Tkinter event loop
    root.mainloop()

if __name__ == "__main__":
    main()