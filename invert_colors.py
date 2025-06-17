from PIL import Image
import numpy as np

def invert_colors(input_path, output_path):
    """
    Invertiert Farben, PIL und Numphy (4k Bilder) 
    """
    
    # laden
    img = Image.open(input_path)
    
    # PIL: Zu RGBA konvertieren falls nötig (für Transparenz)
    #if img.mode != 'RGBA':
    #    img = img.convert('RGBA')
    
    # PIL: Farben invertieren
    #pixels = img.load()
    #width, height = img.size
    
    #for x in range(width):
    #    for y in range(height):
    #        r, g, b, a = pixels[x, y]
            # RGB invertieren, Alpha-Kanal beibehalten
    #        pixels[x, y] = (255 - r, 255 - g, 255 - b, a)     
    
    # Numpy   
    img_array = np.array(img)
    
    # Farben invertieren (RGB-Kanäle)
    if len(img_array.shape) == 3:  # Farbbilder
        img_array[:, :, :3] = 255 - img_array[:, :, :3]
    else:  # Graustufenbilder
        img_array = 255 - img_array
    
    # PIL Speichern
    #img.save(output_path, 'PNG')
    #print(f"gespeichert: {output_path}")

    # Speichern
    inverted_img = Image.fromarray(img_array)
    inverted_img.save(output_path, 'PNG')

invert_colors('input.png', 'output_inverted.png')