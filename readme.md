Real-time Liveness, Emotion, and Face Recognition System
This is a web-based application that uses computer vision and AI to perform a secure, three-stage user verification:

Active Liveness Detection: Verifies the user is a real, live person by issuing a random challenge (e.g., "Smile", "Blink").

Emotion Analysis: After liveness is confirmed, it analyzes and displays the user's emotion in real-time.

Face Recognition: (Future goal) To verify the identity of registered users and grant access.

Technology Stack
Backend: Python, Flask

AI/ML: dlib, OpenCV, DeepFace (TensorFlow backend)

Frontend: HTML, JavaScript, Tailwind CSS

Setup Instructions (For a New Computer)
This guide contains all the necessary steps to set up the development environment for this project from scratch on a new Windows machine.

Step 1: Install Prerequisites
These are the essential system-wide tools that must be installed first.

Python:

Download and install a recent version of Python (e.g., 3.10+) from python.org.

Crucially, during installation, check the box that says "Add Python to PATH".

Git:

Download and install Git from git-scm.com.

CMake:

Download and install the "Windows x64 Installer" from cmake.org.

Crucially, during installation, select the option "Add CMake to the system PATH for all users".

Visual Studio Build Tools (C++ Compiler):

Download the "Build Tools for Visual Studio" from the Visual Studio downloads page (under "Tools for Visual Studio").

Run the installer. In the "Workloads" tab, you must select "Desktop development with C++". This is required to compile dlib.

After installation is complete, restart your computer.

Step 2: Set Up the Project
Clone the Repository:

Open a Command Prompt and navigate to the drive where you want to store the project (e.g., D:).

Run the following command:

git clone [https://github.com/YourUsername/YourRepoName.git](https://github.com/YourUsername/YourRepoName.git)
cd YourRepoName

Install Python Packages:

All required packages are listed in requirements.txt. Install them all with this one command:

pip install -r requirements.txt

(Note: If you are using Conda, you can run conda install -c conda-forge dlib first, and then run pip install -r requirements.txt for the rest.)

Download the Dlib Model:

This project requires a pre-trained facial landmark model.

Download it from this URL: http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2

Unzip the file to get shape_predictor_68_face_landmarks.dat.

Place this .dat file in the main project folder, alongside app.py.

Step 3: Run the Application
Start the Server:

In your command prompt (inside the project folder), run:

python app.py

Open in Browser:

Open your web browser and go to the address:
http://127.0.0.1:5000

The application should now be running.