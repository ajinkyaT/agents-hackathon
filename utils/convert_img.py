import base64
import io
from PIL import Image

def get_image_base64(file):
    base_width = 300
    img = Image.open(file)
    wpercent = (base_width / float(img.size[0]))
    hsize = int((float(img.size[1]) * float(wpercent)))
    img = img.resize((base_width, hsize), Image.Resampling.LANCZOS)
    img_byte_arr = io.BytesIO()
    img.save(img_byte_arr, format='jpeg')
    img_byte_arr = img_byte_arr.getvalue()
    image_data = base64.b64encode(img_byte_arr).decode("utf-8")
    return image_data