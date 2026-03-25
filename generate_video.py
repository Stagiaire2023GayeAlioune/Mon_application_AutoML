
from config import LOGO_PATH, COMPANY_NAME
from moviepy import ImageClip, AudioFileClip, TextClip, CompositeVideoClip, concatenate_videoclips
def create_video(images, voice):
    clips = []

    for img in images:
        clip = ImageClip(img).set_duration(4)
        clips.append(clip)

    video = concatenate_videoclips(clips)

    logo = (
        ImageClip(LOGO_PATH)
        .set_duration(video.duration)
        .resize(height=80)
        .set_position(("right", "bottom"))
    )

    text = (
        TextClip(
            COMPANY_NAME,
            fontsize=50,
            color="white",
            size=(1280,720),
            method="caption"
        )
        .set_duration(video.duration)
    )

    final = CompositeVideoClip([video, logo, text])

    audio = AudioFileClip(voice)
    final = final.set_audio(audio)

    final.write_videofile("output/marketing_video.mp4", fps=24)
