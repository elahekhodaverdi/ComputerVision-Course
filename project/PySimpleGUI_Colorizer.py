# Import necessary libraries
import cv2
import numpy as np
from tkinter import filedialog
import customtkinter as ctk
from PIL import Image, ImageTk
import threading
import time

# Set the appearance mode and color theme for the GUI
ctk.set_appearance_mode("Dark")
ctk.set_default_color_theme("blue")

# Load the pre-trained model files
prototxt = r'model/colorization_deploy_v2.prototxt'  # architecture file
model = r'model/colorization_release_v2.caffemodel'  # model weights
points = r'model/pts_in_hull.npy'  # Cluster center points for colorization

# Load the model using OpenCV's Deep Neural Network (DNN) module
net = cv2.dnn.readNetFromCaffe(prototxt, model)  # Load the model
pts = np.load(points)  # Load the cluster center points

# Add cluster centers as 1x1 convolutions to the model
class8 = net.getLayerId("class8_ab")  # Get the ID of the layer "class8_ab"
conv8 = net.getLayerId("conv8_313_rh")  # Get the ID of the layer "conv8_313_rh"
pts = pts.transpose().reshape(2, 313, 1, 1)  # Reshape the cluster centers
net.getLayer(class8).blobs = [pts.astype("float32")]  # Set the weights for class8_ab
net.getLayer(conv8).blobs = [np.full([1, 313], 2.606, dtype="float32")]  # Set the weights for conv8_313_rh

# Function to colorize an image
def colorize_image(image):
    """
    Colorizes a grayscale or color image using the pre-trained model.
    Args:
        image (numpy.ndarray): Input image in BGR format.
    Returns:
        numpy.ndarray: Colorized image in BGR format.
    """
    # Normalize the image to the range [0, 1]
    scaled = image.astype("float32") / 255.0
    # Convert the image from BGR to LAB color space
    lab = cv2.cvtColor(scaled, cv2.COLOR_BGR2LAB)
    # Resize the image to 224x224 (required by the model)
    resized = cv2.resize(lab, (224, 224))
    # Extract the L channel (lightness) from the LAB image
    L = cv2.split(resized)[0]
    L -= 50  # Subtract 50 for normalization
    # Pass the L channel through the model to predict the AB channels
    net.setInput(cv2.dnn.blobFromImage(L))
    ab = net.forward()[0, :, :, :].transpose((1, 2, 0))
    # Resize the predicted AB channels to the original image size
    ab = cv2.resize(ab, (image.shape[1], image.shape[0]))
    # Extract the L channel from the original LAB image
    L = cv2.split(lab)[0]
    # Combine the L channel with the predicted AB channels
    colorized = np.concatenate((L[:, :, np.newaxis], ab), axis=2)
    # Convert the LAB image back to BGR color space
    colorized = cv2.cvtColor(colorized, cv2.COLOR_LAB2BGR)
    # Clip the pixel values to the range [0, 1] and scale to [0, 255]
    colorized = np.clip(colorized, 0, 1)
    colorized = (255 * colorized).astype("uint8")
    return colorized

# Main application class
class ColorizerApp(ctk.CTk):
    def __init__(self):
        """
        Initialize the Colorizer application.
        """
        super().__init__()
        self.title("Image and Video Colorizer")  # Set the window title
        self.geometry("1000x700")  # Set the window size
        self.resizable(False, False)  # Disable window resizing

        # Variables for managing video processing
        self.running = False  # Flag to control video processing threads
        self.current_thread = None  # Current thread for video processing
        self.video_capture = None  # Video capture object for webcam or video files

        # Create a tabbed interface
        self.tabview = ctk.CTkTabview(self)
        self.tabview.pack(pady=20, padx=20, fill="both", expand=True)

        # Add tabs to the interface
        self.tab1 = self.tabview.add("Colorizer")  # Tab for colorizing images/videos
        self.tab2 = self.tabview.add("Save Media")  # Tab for saving processed media

        # Set up the UI for both tabs
        self.setup_colorizer_tab()
        self.setup_save_media_tab()

    def setup_colorizer_tab(self):
        """
        Set up the UI for the Colorizer tab.
        """
        # Main frame for the Colorizer tab
        self.main_frame = ctk.CTkFrame(self.tab1)
        self.main_frame.pack(pady=20, padx=20, fill="both", expand=True)

        # Top frame for buttons
        self.top_frame = ctk.CTkFrame(self.main_frame)
        self.top_frame.pack(pady=10, padx=10, fill="x")

        # Button to upload an image
        self.upload_image_button = ctk.CTkButton(self.top_frame, text="Upload Image", command=self.upload_image)
        self.upload_image_button.pack(side="left", padx=10)

        # Button to upload a video
        self.upload_video_button = ctk.CTkButton(self.top_frame, text="Upload Video", command=self.upload_video)
        self.upload_video_button.pack(side="left", padx=10)

        # Frame for displaying input and output images
        self.image_frame = ctk.CTkFrame(self.main_frame)
        self.image_frame.pack(pady=10, padx=10, fill="both", expand=True)

        # Input panel for the original image
        self.input_panel = ctk.CTkFrame(self.image_frame, width=400, height=400)
        self.input_panel.pack(side="left", padx=10)
        self.input_panel.pack_propagate(False)

        # Label for the input panel
        self.input_label = ctk.CTkLabel(self.input_panel, text="Input")
        self.input_label.pack(pady=5)

        # Canvas to display the input image
        self.input_canvas = ctk.CTkCanvas(self.input_panel, bg="gray", width=400, height=400)
        self.input_canvas.pack()

        # Output panel for the colorized image
        self.output_panel = ctk.CTkFrame(self.image_frame, width=400, height=400)
        self.output_panel.pack(side="left", padx=10)
        self.output_panel.pack_propagate(False)

        # Label for the output panel
        self.output_label = ctk.CTkLabel(self.output_panel, text="Output")
        self.output_label.pack(pady=5)

        # Canvas to display the output image
        self.output_canvas = ctk.CTkCanvas(self.output_panel, bg="gray", width=400, height=400)
        self.output_canvas.pack()

    def setup_save_media_tab(self):
        """
        Set up the UI for the Save Media tab.
        """
        # Main frame for the Save Media tab
        self.save_frame = ctk.CTkFrame(self.tab2)
        self.save_frame.pack(pady=20, padx=20, fill="both", expand=True)

        # Button to upload media (image or video)
        self.upload_save_button = ctk.CTkButton(self.save_frame, text="Upload Media", command=self.upload_save_media)
        self.upload_save_button.pack(pady=10)

        # Button to save the processed media
        self.save_button = ctk.CTkButton(self.save_frame, text="Save Media", command=self.save_media, state="disabled")
        self.save_button.pack(pady=10)

        # Label to display the status of the saving process
        self.save_status_label = ctk.CTkLabel(self.save_frame, text="")
        self.save_status_label.pack(pady=10)

        # Variables to store the uploaded media
        self.save_media_path = None  # Path to the uploaded media
        self.save_media_type = None  # Type of media ('image' or 'video')
        self.save_media = None  # The actual media data (image or video path)

    def upload_image(self):
        """
        Upload an image and process it.
        """
        self._reset_processing()  # Stop any ongoing processing
        # Open a file dialog to select an image
        file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.png;*.jpg;*.jpeg")])
        if file_path:
            # Read the image using OpenCV
            self.current_image = cv2.imread(file_path)
            # Process the image (colorize it)
            self.process_frame(self.current_image)

    def process_frame(self, frame):
        """
        Process a single frame (colorize it) and display the input and output.
        Args:
            frame (numpy.ndarray): The input frame to process.
        """
        # Colorize the frame
        colorized_frame = colorize_image(frame)
        # Display the original and colorized frames on the canvases
        self.display_image_on_canvas(self.input_canvas, frame)
        self.display_image_on_canvas(self.output_canvas, colorized_frame)

    def upload_video(self):
        """
        Upload a video and process it frame by frame.
        """
        self._reset_processing()  # Stop any ongoing processing
        # Open a file dialog to select a video
        file_path = filedialog.askopenfilename(filetypes=[("Video files", "*.mp4;*.avi;*.mov")])
        if file_path:
            self.running = True  # Set the running flag to True
            # Start a new thread to process the video
            self.current_thread = threading.Thread(target=self.process_video, args=(file_path,))
            self.current_thread.start()

    def process_video(self, file_path):
        """
        Process a video frame by frame.
        Args:
            file_path (str): Path to the video file.
        """
        # Open the video file
        cap = cv2.VideoCapture(file_path)
        while self.running and cap.isOpened():
            # Read a frame from the video
            ret, frame = cap.read()
            if not ret:
                break  # Exit the loop if no more frames are available
            # Process the frame (colorize it)
            self.process_frame(frame)
            time.sleep(0.01)  # Add a small delay to control the processing speed
        cap.release()  # Release the video capture object

    def display_image_on_canvas(self, canvas, image):
        """
        Display an image on a canvas.
        Args:
            canvas (CTkCanvas): The canvas to display the image on.
            image (numpy.ndarray): The image to display.
        """
        # Convert the image from BGR to RGB format
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # Convert the image to a PIL image
        image = Image.fromarray(image)
        # Resize the image to fit the canvas while maintaining the aspect ratio
        image.thumbnail((400, 400), Image.Resampling.LANCZOS)
        # Convert the PIL image to a Tkinter-compatible image
        image_tk = ImageTk.PhotoImage(image)
        # Display the image on the canvas
        canvas.create_image(200, 200, image=image_tk)
        canvas.image = image_tk  # Keep a reference to avoid garbage collection

    def upload_save_media(self):
        """
        Upload media (image or video) for saving.
        """
        # Open a file dialog to select an image or video
        file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.png;*.jpg;*.jpeg"), ("Video files", "*.mp4;*.avi;*.mov")])
        if file_path:
            self.save_media_path = file_path  # Store the file path
            # Determine if the media is an image or video
            if file_path.lower().endswith(('.png', '.jpg', '.jpeg')):
                self.save_media_type = 'image'  # Set the media type to 'image'
                self.save_media = cv2.imread(file_path)  # Read the image
            else:
                self.save_media_type = 'video'  # Set the media type to 'video'
                self.save_media = file_path  # Store the video path
            # Enable the save button and update the status label
            self.save_button.configure(state="normal")
            self.save_status_label.configure(text="Media uploaded. Click 'Save Media' to process and save.")

    def save_media(self):
        """
        Save the processed media (image or video).
        """
        if self.save_media is not None:
            # Disable the buttons to prevent multiple clicks
            self.save_button.configure(state="disabled")
            self.upload_save_button.configure(state="disabled")
            # Update the status label to indicate processing
            self.save_status_label.configure(text="Processing... Please wait.")

            if self.save_media_type == 'image':
                # Colorize the image
                colorized_image = colorize_image(self.save_media)
                # Open a file dialog to choose the save location
                save_path = filedialog.asksaveasfilename(defaultextension=".jpg", filetypes=[("JPEG files", "*.jpg")])
                if save_path:
                    # Save the colorized image
                    cv2.imwrite(save_path, colorized_image)
                    self.save_status_label.configure(text=f"Image saved to {save_path}")
                else:
                    self.save_status_label.configure(text="Save cancelled.")
            elif self.save_media_type == 'video':
                # Open a file dialog to choose the save location
                save_path = filedialog.asksaveasfilename(defaultextension=".mp4", filetypes=[("MP4 files", "*.mp4")])
                if save_path:
                    # Process and save the video
                    self.process_and_save_video(self.save_media, save_path)
                else:
                    self.save_status_label.configure(text="Save cancelled.")

            # Re-enable the buttons after processing is complete
            self.save_button.configure(state="normal")
            self.upload_save_button.configure(state="normal")

    def process_and_save_video(self, input_path, output_path):
        """
        Process and save a video frame by frame.
        Args:
            input_path (str): Path to the input video.
            output_path (str): Path to save the output video.
        """
        # Open the input video
        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            self.save_status_label.configure(text="Error opening video file.")
            return

        # Get video properties (width, height, frames per second)
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        # Define the codec for the output video
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        # Create a VideoWriter object to save the output video
        out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

        while cap.isOpened():
            # Read a frame from the input video
            ret, frame = cap.read()
            if not ret:
                break  # Exit the loop if no more frames are available
            # Colorize the frame
            colorized_frame = colorize_image(frame)
            # Write the colorized frame to the output video
            out.write(colorized_frame)

        # Release the video capture and writer objects
        cap.release()
        out.release()
        # Update the status label to indicate success
        self.save_status_label.configure(text=f"Video saved to {output_path}")

    def _reset_processing(self):
        """
        Reset the processing state.
        """
        self.running = False  # Stop any ongoing processing

# Main entry point of the application
if __name__ == "__main__":
    app = ColorizerApp()  # Create an instance of the application
    app.mainloop()  # Start the main event loop