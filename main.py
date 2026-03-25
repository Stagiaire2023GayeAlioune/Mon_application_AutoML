from generate_images import generate_images
from generate_voice import generate_voice
from generate_video import create_video

def main():
    print("Generating images...")
    images = generate_images()

    print("Generating voice...")
    voice = generate_voice()

    print("Creating video...")
    create_video(images, voice)

    print("Video generated in output folder")

if __name__ == "__main__":
    main()
