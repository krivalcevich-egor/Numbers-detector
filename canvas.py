import tkinter as tk
from PIL import Image, ImageOps, ImageGrab
import numpy as np
import cv2
from testing import classify_image  


screen_size = (512, 512)


def draw_line(event):
    global last_x, last_y
    # Draw a line following the cursor
    canvas.create_line(last_x, last_y, event.x, event.y, fill="white", width=15)
    last_x, last_y = event.x, event.y

def set_last_xy(event):
    global last_x, last_y
    # Set the initial coordinates for the line
    last_x, last_y = event.x, event.y

def process_single_digit(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    resized_digit = cv2.resize(gray, (28, 28))
    
    return resized_digit

def clear_canvas():
    canvas.delete("all")
    canvas.create_rectangle(0, 0, screen_size[0], screen_size[1], fill="black")
    
def save_image():
    # Save the canvas content to a file
    canvas.postscript(file="canvas.ps", colormode='color')
    img = Image.open("canvas.ps")
    img = img.convert('L')  
    img.save("drawing.png")
    
    image = cv2.imread("drawing.png")
    digit = process_single_digit(image)
    cv2.imwrite("digit_processed.png", digit)
    
    predicted_digit = classify_image(digit)
    print(f"Predicted Digit: {predicted_digit}")
    
    with open("result.txt", "w") as f:
        f.write(f"{predicted_digit}\n")


root = tk.Tk()
root.title("Drawing Numbers")

# Create a canvas 
canvas = tk.Canvas(root, width=screen_size[0], height=screen_size[1], bg="black")
canvas.pack()

# Bind mouse events 
canvas.bind("<Button-1>", set_last_xy)
canvas.bind("<B1-Motion>", draw_line)

# Create buttons 
btn_clear = tk.Button(root, text="Clear", command=clear_canvas)
btn_clear.pack(side=tk.LEFT, padx=10, pady=10)

btn_save = tk.Button(root, text="Save", command=save_image)
btn_save.pack(side=tk.RIGHT, padx=10, pady=10)

clear_canvas()
root.mainloop()
