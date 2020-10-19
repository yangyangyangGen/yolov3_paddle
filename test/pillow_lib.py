import PIL
from PIL import Image

if __name__ == "__main__":
    fname = r"C:\Users\Administrator\Desktop\yang\bgImg\nice\v2-91e3d9c8949d1175afa0767a02bfa18b_r.jpg"

    im = Image.open(fname)
    print(im.size)
    box_xyxy = [500, 500, 800, 800]

    out = im.crop(box_xyxy).resize(im.size, Image.LANCZOS)
    out.show()

    # im.crop()
