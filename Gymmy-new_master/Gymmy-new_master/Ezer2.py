import tkinter as tk
from PIL import Image, ImageTk
import itertools


class CircularProgressBar:
    def __init__(self, root, size=200, max_count=10, gif_path="C:/Users/Administrator/Downloads/926f0a15541e503306ceac5e92dbfd36.gif"):
        self.root = root
        self.size = size  # Diameter of the circle
        self.max_count = max_count  # Maximum count
        self.count = 0  # Current count
        self.gif_path = gif_path  # Path to GIF

        self.canvas = tk.Canvas(root, width=size, height=size, bg="white", highlightthickness=0)
        self.canvas.pack(pady=20)

        # Draw the background circle
        self.canvas.create_oval(10, 10, size - 10, size - 10, outline="#ddd", width=10)

        # Arc that represents the progress
        self.arc = self.canvas.create_arc(10, 10, size - 10, size - 10, start=90, extent=0,
                                          outline="#00AAFF", width=10, style="arc")

        # Text in the center
        self.text = self.canvas.create_text(size // 2, size // 2, text="0", font=("Arial", 20, "bold"), fill="black")

        # Button to increase the count
        self.button = tk.Button(root, text="Increment", command=self.increment_count, font=("Arial", 14))
        self.button.pack(pady=10)

        # Load the GIF but keep it hidden
        self.load_gif()

    def load_gif(self):
        """Loads GIF frames"""
        self.gif = Image.open(self.gif_path)
        self.gif_frames = []

        try:
            while True:
                frame = self.gif.copy()
                frame.thumbnail((self.size // 2, self.size // 2))  # Resize for center display
                self.gif_frames.append(ImageTk.PhotoImage(frame))
                self.gif.seek(len(self.gif_frames))  # Move to next frame
        except EOFError:
            pass  # Stop when all frames are loaded

        self.current_frame = itertools.cycle(self.gif_frames)
        self.gif_display = None  # Placeholder for GIF display

    def increment_count(self):
        """Increases count and updates progress if below max"""
        if self.count < self.max_count:
            self.count += 1
            self.update_progress()
        if self.count == self.max_count:
            self.show_gif()  # Show GIF when complete

    def update_progress(self):
        """Updates the circular bar and count in the center"""
        progress = (self.count / self.max_count) * 360  # Convert count to degrees
        self.canvas.itemconfig(self.arc, extent=-progress)  # Update arc
        self.canvas.itemconfig(self.text, text=str(self.count))  # Update count

    def show_gif(self):
        """Shows the GIF in the center when the bar is complete"""
        self.canvas.itemconfig(self.text, state="hidden")  # Hide the number
        self.animate_gif()

    def animate_gif(self):
        """Animates the GIF in the center"""
        if self.gif_display:
            self.canvas.delete(self.gif_display)

        frame = next(self.current_frame)
        self.gif_display = self.canvas.create_image(self.size // 2, self.size // 2, image=frame)
        self.root.after(50, self.animate_gif)  # Loop animation


# Create the GUI
if __name__ == "__main__":
    root = tk.Tk()
    root.title("Circular Progress Bar with GIF")

    progress_bar = CircularProgressBar(root, size=200, max_count=10)

    root.mainloop()
