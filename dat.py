import cv2
import os
import face_recognition

# Function to capture images and save them
def capture_images(output_dir, name, num_images=15):
    # Ensure the output directory exists
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    # Start capturing images from the camera
    cap = cv2.VideoCapture(0)
    print(f"Capturing {num_images} images...")
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

        # Save the frame with faces
        if face_locations:
            # Save the frame as an image file
            filename = os.path.join(output_dir, f"{name}_{count}.jpg")
            cv2.imwrite(filename, frame)
            print(f"Image {count + 1} saved.")
            count += 1

        # Break the loop if 'q' key is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the camera and close OpenCV windows
    cap.release()
    cv2.destroyAllWindows()

if __name__  ==  "__main__":
    # Set the directory to save captured images
    train_dir = "train_dir"

    # Ask the user for the name of the person
    person_name = input("Enter the name of the person: ")

    # Create the directory for the person if it doesn't exist
    person_dir = os.path.join(train_dir, person_name)
    if not os.path.exists(person_dir):
        os.makedirs(person_dir)

    # Capture and save images
    capture_images(person_dir, person_name)
