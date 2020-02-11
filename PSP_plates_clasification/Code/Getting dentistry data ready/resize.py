import os
from datetime import datetime
from PIL import Image
for image_file_name in os.listdir('./selected_blank/'):
    if image_file_name.endswith(".png"):


        im = Image.open('./selected_blank/'+image_file_name)
        new_width  = 1152
        new_height = 869
        im = im.resize((new_width, new_height), Image.ANTIALIAS)
        im.save('./selected_blank/'+image_file_name)
