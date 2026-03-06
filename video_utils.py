import cv2
import whisper
from moviepy.editor import VideoFileClip
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration

# Load models once
speech_model = whisper.load_model("base")
similarity_model = SentenceTransformer('all-MiniLM-L6-v2')

processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
vision_model = BlipForConditionalGeneration.from_pretrained(
    "Salesforce/blip-image-captioning-base"
)


def check_audio(video_path):

    video = VideoFileClip(video_path)

    return video.audio is not None


def extract_audio(video_path):

    audio_path = "audio.wav"

    video = VideoFileClip(video_path)

    video.audio.write_audiofile(audio_path)

    return audio_path


def speech_to_text(audio_path):

    result = speech_model.transcribe(audio_path)

    return result["text"]


def compare_text(reference_text, transcribed_text):

    ref_emb = similarity_model.encode([reference_text])
    trans_emb = similarity_model.encode([transcribed_text])

    similarity = cosine_similarity(ref_emb, trans_emb)

    return similarity[0][0] * 100


def extract_frames(video_path, interval=60):

    cap = cv2.VideoCapture(video_path)

    frames = []

    frame_id = 0

    while True:

        ret, frame = cap.read()

        if not ret:
            break

        if frame_id % interval == 0:
            frames.append(frame)

        frame_id += 1

    cap.release()

    return frames


def caption_frame(frame):

    image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    inputs = processor(image, return_tensors="pt")

    out = vision_model.generate(**inputs)

    caption = processor.decode(out[0], skip_special_tokens=True)

    return caption


def describe_video_visual(video_path):

    frames = extract_frames(video_path)

    captions = []

    for frame in frames:

        caption = caption_frame(frame)

        captions.append(caption)

    return " ".join(captions)


def process_video(video_path, reference_text):

    result = {}

    has_audio = check_audio(video_path)

    visual_description = describe_video_visual(video_path)

    result["visual_description"] = visual_description

    if not has_audio:

        result["audio"] = "No audio detected"
        return result

    audio_path = extract_audio(video_path)

    transcribed_text = speech_to_text(audio_path)

    similarity_score = compare_text(reference_text, transcribed_text)

    result["transcribed_text"] = transcribed_text
    result["similarity"] = similarity_score

    return result