import cv2
import numpy as np
import turtle
from .gen import generate

def detect_edges(image_path, out_path="edges.png", threshold1=100, threshold2=200):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    edges = cv2.Canny(img, threshold1, threshold2)
    cv2.imwrite(out_path, edges)
    plt.imshow(edges, cmap='gray')
    plt.title("Edge Detection")
    plt.axis("off")
    return edges 

def generate_pixel_vector(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        print(f"Error: Could not load {image_path}")
        return None
    pixel_vector = (image == 255)  # 255 is white in grayscale
    return pixel_vector

def generate_8bit_pixel_vector(image_path, k=256):
    # Load image in BGR format
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if image is None:
        print(f"Error: Could not load {image_path}")
        return None
    h, w, _ = image.shape
    pixels = image.reshape(-1, 3).astype(np.float32)
    _, labels, palette = cv2.kmeans(
        pixels, k, None,
        (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0),
        10, cv2.KMEANS_RANDOM_CENTERS
    )

    quantized_pixels = np.uint8(palette)[labels.flatten()]
    hex_pixels = [f"#{r:02x}{g:02x}{b:02x}" for b, g, r in quantized_pixels]  # Convert BGR to RGB

    pixel_vector = [hex_pixels[i * w: (i + 1) * w] for i in range(h)]
    
    return pixel_vector



def bw_turtle_render(pixel_vector, screen_width=300, screen_height=200):
    screen = turtle.Screen()
    turtle.color("white")
    screen.bgcolor("black")
    screen.setup(screen_width, screen_height)
    turtle.speed(0)  
    turtle.penup()
    turtle.shape("turtle")
    turtle.showturtle()
    scale_x = screen_width / len(pixel_vector[0])
    scale_y = screen_height / len(pixel_vector)
    
    # print(f"Scale X: {scale_x}, Scale Y: {scale_y}")

    current_row = None

    for y in range(len(pixel_vector)):
        for x in range(len(pixel_vector[y])):
            if pixel_vector[y, x]:  
                # print(f"Drawing at ({x}, {y})")
                turtle.goto(x * scale_x - screen_width // 2, screen_height // 2 - y * scale_y)
                turtle.pendown()
            else:
                turtle.penup() 
    turtle.done()

def config_turtle(screen_width=300, screen_height=200):
    screen = turtle.Screen()
    screen.bgcolor("black")
    screen.setup(screen_width, screen_height)
    turtle.speed(0)
    turtle.penup()
    turtle.shape("turtle")
    turtle.showturtle()

def turtle_render(pixel_vector, screen_width=300, screen_height=200):
    config_turtle()
    scale_x = screen_width / len(pixel_vector[0])
    scale_y = screen_height / len(pixel_vector)

    curr_row = None

    for i in range(len(pixel_vector)):
        for j in range(len(pixel_vector[i])):
                turtle.goto(j * scale_x - screen_width // 2, screen_height // 2 - i * scale_y)
                turtle.color(pixel_vector[i][j])
                turtle.pendown()
    turtle.done()


def draw_ai_image(is_bw=False):
    
    generate("assets/image.png")

    if is_bw:
       e = detect_edges("assets/image.png")
       pv = generate_pixel_vector(e)
       bw_turtle_render(pv)
    else:
        pv = generate_8bit_pixel_vector("assets/image.png")
        turtle_render(pv)
if __name__ == '__main__':

    # out_path = "sponge.png"

    # # Detect edges and save the output image
    # # edges_image = detect_edges("sponge.png", out_path)
    # pixel_vector = generate_pixel_vector(out_path)
    # if pixel_vector is not None:
    #     draw_edges_row_by_row(pixel_vector, 300, 200) 
    draw_ai_image()