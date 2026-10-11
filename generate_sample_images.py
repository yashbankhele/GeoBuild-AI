from pathlib import Path
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "data" / "sample_images"
OUT.mkdir(parents=True, exist_ok=True)

def make_site(after=False):
    im = Image.new("RGB", (640, 480), (198, 190, 166))
    d = ImageDraw.Draw(im)
    d.rectangle((0,0,145,130), fill=(104,143,91))
    d.rectangle((465,0,639,145), fill=(118,150,97))
    d.rectangle((0,355,140,479), fill=(110,146,89))
    d.ellipse((480,350,610,465), fill=(101,137,86))
    d.rectangle((0,210,639,257), fill=(122,123,119))
    d.rectangle((140,80,470,410), outline=(230,215,165), width=5)
    d.rectangle((175,105,270,180), fill=(151,147,140))
    d.rectangle((300,290,390,365), fill=(163,157,148))
    d.rectangle((185,285,255,350), fill=(181,160,129))
    if after:
        d.rectangle((300,105,410,183), fill=(143,140,135))
        d.rectangle((180,370,290,402), fill=(154,149,142))
        d.rectangle((400,260,430,345), fill=(126,126,121))
        d.rectangle((35,155,112,190), fill=(160,155,147))
    return im

if __name__ == "__main__":
    make_site(False).save(OUT / "before.png")
    make_site(True).save(OUT / "after.png")
    print(f"Generated sample images at {OUT}")
