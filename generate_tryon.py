from PIL import Image

def overlay_cloth(person_path, cloth_path, output_path="tryon.jpg"):
    person = Image.open(person_path).resize((256, 256))
    cloth = Image.open(cloth_path).resize((256, 256)).convert("RGBA")
    person.paste(cloth, (0, 0), cloth)
    person.save(output_path)
    print(f"Saved try-on image to {output_path}")

if __name__ == "__main__":
    overlay_cloth("input.jpg", "data/cloth.png")

