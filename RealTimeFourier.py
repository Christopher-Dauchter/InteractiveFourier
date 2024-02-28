import tkinter as tk
from tkinter import filedialog, Scale, messagebox
from PIL import Image, ImageTk
import cv2
import numpy as np

class FFTApp:
    def __init__(self, master):
        self.master = master
        self.master.title('FFT Image Masking and Difference Visualization')

        # Frame for the top row (original and FFT images)
        self.frame_top = tk.Frame(self.master)
        self.frame_top.pack()

        # Frame for the bottom row (resultant and difference images)
        self.frame_bottom = tk.Frame(self.master)
        self.frame_bottom.pack()

        # Frame for controls (below images)
        self.frame_controls = tk.Frame(self.master)
        self.frame_controls.pack()

        # Load image button
        self.btn_load = tk.Button(self.frame_controls, text='Load Image', command=self.load_image)
        self.btn_load.pack(side=tk.LEFT, padx=5, pady=5)

        # Slider for brush radius
        self.slider_radius = Scale(self.frame_controls, from_=1, to=50, orient=tk.HORIZONTAL, label='Brush Radius')
        self.slider_radius.pack(side=tk.LEFT, padx=5)

        # Save image button
        self.btn_save = tk.Button(self.frame_controls, text='Save Resultant Image', command=self.save_image)
        self.btn_save.pack(side=tk.LEFT, padx=5, pady=5)

        # Canvas for original image and FFT
        self.canvas_orig = tk.Canvas(self.frame_top, width=384, height=384)
        self.canvas_orig.pack(side=tk.LEFT, padx=5, pady=5)

        self.canvas_fft = tk.Canvas(self.frame_top, width=384, height=384, cursor="cross")
        self.canvas_fft.pack(side=tk.LEFT, padx=5, pady=5)
        self.canvas_fft.bind("<B1-Motion>", self.mask_fft)

        # Canvas for resultant image and difference
        self.canvas_result = tk.Canvas(self.frame_bottom, width=384, height=384)
        self.canvas_result.pack(side=tk.LEFT, padx=5, pady=5)

        self.canvas_difference = tk.Canvas(self.frame_bottom, width=384, height=384)
        self.canvas_difference.pack(side=tk.LEFT, padx=5, pady=5)

        # Variables for images and mask
        self.orig_img = None
        self.fft_img = None
        self.result_img = None
        self.fft_mask = None
        self.difference_img = None

    def load_image(self):
        file_path = filedialog.askopenfilename()
        if file_path:
            img = cv2.imread(file_path, 0)
            self.orig_img = cv2.resize(img, (384, 384))
            self.show_image(self.orig_img, self.canvas_orig)

            # FFT
            f = np.fft.fft2(self.orig_img)
            fshift = np.fft.fftshift(f)
            magnitude_spectrum = 20*np.log(np.abs(fshift) + 1)  # Added +1 to avoid log(0)
            self.fft_img = np.array(magnitude_spectrum, dtype=np.uint8)
            self.show_image(self.fft_img, self.canvas_fft)

            # Initialize the mask with ones
            self.fft_mask = np.ones_like(self.orig_img, dtype=np.uint8)

    def show_image(self, img, canvas):
        img_pil = Image.fromarray(img)
        img_tk = ImageTk.PhotoImage(image=img_pil)
        canvas.imgtk = img_tk  # Keep reference
        canvas.create_image(0, 0, anchor="nw", image=img_tk)

    def mask_fft(self, event):
        x, y = event.x, event.y
        radius = self.slider_radius.get()  # Get radius from slider
        cv2.circle(self.fft_mask, (x, y), radius, (0,), -1)  # Draw on the mask

        # Apply the mask visually on FFT
        masked_fft_display = self.fft_img.copy()
        masked_fft_display[self.fft_mask == 0] = 0  # Black out masked areas
        self.show_image(masked_fft_display, self.canvas_fft)  # Update FFT display with mask

        # Apply mask for inverse FFT and update the result and difference images
        self.update_result_and_difference()

    def update_result_and_difference(self):
        # Apply mask for inverse FFT
        f = np.fft.fft2(self.orig_img)
        fshift = np.fft.fftshift(f)
        fshift[self.fft_mask == 0] = 0  # Apply mask
        f_ishift = np.fft.ifftshift(fshift)
        img_back = np.fft.ifft2(f_ishift)
        img_back = np.abs(img_back)
        self.result_img = np.array(img_back, dtype=np.uint8)
        self.show_image(self.result_img, self.canvas_result)

        # Calculate and display difference
        difference = cv2.absdiff(self.orig_img, self.result_img)
        self.difference_img = np.array(difference, dtype=np.uint8)
        self.show_image(self.difference_img, self.canvas_difference)

    def save_image(self):
        if self.result_img is not None:
            file_path = filedialog.asksaveasfilename(defaultextension=".png", filetypes=[("PNG files", "*.png"), ("JPEG files", "*.jpg"), ("All files", "*.*")])
            if file_path:
                cv2.imwrite(file_path, self.result_img)
                messagebox.showinfo("Save Image", "Resultant image saved successfully!")
        else:
            messagebox.showwarning("Save Image", "No resultant image to save.")

if __name__ == "__main__":
    root = tk.Tk()
    app = FFTApp(root)
    root.mainloop()
