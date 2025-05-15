import tkinter as tk
import PIL.ImageGrab as ImageGrab
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Handwritten Digit Recognizer")
        self.canvas = tk.Canvas(self, width=280, height=280, bg='black')
        self.canvas.pack()
        self.button_predict = tk.Button(self, text="Predict", command=self.predict_digit)
        self.button_predict.pack()
        self.button_clear = tk.Button(self, text="Clear", command=self.clear_canvas)
        self.button_clear.pack()
        self.label_result = tk.Label(self, text="", font=("Helvetica", 20))
        self.label_result.pack()
        self.canvas.bind("<B1-Motion>", self.draw)
        self.model = tf.keras.models.load_model("model.h5")

    def draw(self, event):
        x, y = event.x, event.y
        r = 8
        self.canvas.create_oval(x - r, y - r, x + r, y + r, fill="white", outline="white")

    def clear_canvas(self):
        self.canvas.delete("all")
        self.label_result.config(text="")

    def preprocess(self, image):
        # Resize to 28x28 and convert to grayscale
        image = image.resize((28, 28)).convert("L")
        image = np.array(image)
    
        # Invert the colors (white digit on black)
        image = 255 - image
    
        # Thresholding to clean up the image
        image[image < 100] = 0
        image[image >= 100] = 255

        # Normalize
        image = image / 255.0

        # Ensure that the image has the correct shape for the model
        image = np.expand_dims(image, axis=-1)  # Add channel dimension
        image = np.expand_dims(image, axis=0)   # Add batch dimension

        # Show image for debugging
        plt.imshow(image[0, :, :, 0], cmap='gray')
        plt.title("Processed Image")
        plt.show()

        return image


    def predict_digit(self):
        x = self.winfo_rootx() + self.canvas.winfo_x()
        y = self.winfo_rooty() + self.canvas.winfo_y()
        x1 = x + self.canvas.winfo_width()
        y1 = y + self.canvas.winfo_height()
    
        # Capture the image and preprocess it
        image = ImageGrab.grab().crop((x, y, x1, y1))
        processed_image = self.preprocess(image)

        # Get model prediction
        prediction = self.model.predict(processed_image)

        # Log raw prediction output for debugging
        print("Raw model prediction:", prediction)

        digit = np.argmax(prediction)
        confidence = np.max(prediction)

        # Display results
        self.label_result.config(text=f"Predicted Digit: {digit}\nConfidence: {confidence*100:.2f}%")



if __name__ == "__main__":
    app = App()
    app.mainloop()