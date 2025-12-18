from gtts import gTTS
from pydub import AudioSegment
import os
import random
from tqdm import tqdm
import csv

AUDIO_DIR = "milestone1/dataset/audio"
TRANS_DIR = "milestone1/dataset/transcripts"
LABELS_FILE = "milestone1/dataset/labels.csv"

os.makedirs(AUDIO_DIR, exist_ok=True)
os.makedirs(TRANS_DIR, exist_ok=True)

# Sample sales call templates
conversations = [
    [
        ("agent", "Hello! I’m calling from XYZ Insurance. How are you today?"),
        ("customer", "I’m doing good, thank you. What is this about?"),
        ("agent", "We’re offering a new health plan with flexible monthly pricing."),
        ("customer", "Sounds interesting. How much does it cost?")
    ],
    [
        ("agent", "Hi! I’m from ABC CRM. Do you currently use any CRM software?"),
        ("customer", "Yes, but it’s slow and expensive."),
        ("agent", "Our software is faster, affordable, and integrates with WhatsApp."),
        ("customer", "That’s great. Can you send me a demo link?")
    ],
    [
        ("agent", "Hello, this is Mike from Bright Broadband. Are you interested in a faster plan?"),
        ("customer", "Maybe, depends on the cost."),
        ("agent", "We have a 50% discount this week on all new connections."),
        ("customer", "That’s a good offer. How can I apply?")
    ]
]

def heuristic_sentiment(text):
    """Simple sentiment + emotion labeling"""
    positive_keywords = ["great", "good", "discount", "offer", "flexible", "interested", "affordable"]
    negative_keywords = ["expensive", "slow", "no", "not", "problem"]
    text_lower = text.lower()
    pos = sum(word in text_lower for word in positive_keywords)
    neg = sum(word in text_lower for word in negative_keywords)
    if pos > neg:
        return "POSITIVE", "Happy"
    elif neg > pos:
        return "NEGATIVE", "Frustrated"
    else:
        return "NEUTRAL", "Calm"

def generate_dataset(num_samples=100):
    """Generate dataset with 2 distinct voices (agent female, customer male)."""
    with open(LABELS_FILE, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["filename", "speaker", "text", "sentiment", "emotion"])

        for i in tqdm(range(num_samples), desc="Generating Calls"):
            convo = random.choice(conversations)
            transcript_lines = []
            combined_audio = AudioSegment.silent(duration=500)  # start with silence

            for speaker, line in convo:
                # Choose voice type
                if speaker == "agent":
                    tts = gTTS(text=line, lang="en", tld="co.uk")  # female (UK)
                else:
                    tts = gTTS(text=line, lang="en", tld="com")    # male (US)

                # Save temporary MP3 and convert to segment
                temp_file = f"temp_{speaker}_{i}.mp3"
                tts.save(temp_file)
                segment = AudioSegment.from_mp3(temp_file)
                combined_audio += segment + AudioSegment.silent(duration=400)
                os.remove(temp_file)

                # Write each dialogue line to CSV
                sentiment, emotion = heuristic_sentiment(line)
                writer.writerow([f"call_{i+1}.wav", speaker, line, sentiment, emotion])
                transcript_lines.append(f"{speaker}: {line}")

            # Save transcript
            transcript_path = os.path.join(TRANS_DIR, f"call_{i+1}.txt")
            with open(transcript_path, "w", encoding="utf-8") as f:
                f.write("\n".join(transcript_lines))

            # Export combined WAV file
            audio_path = os.path.join(AUDIO_DIR, f"call_{i+1}.wav")
            combined_audio.export(audio_path, format="wav")

    print(f"\n✅ Dataset successfully generated with 2 voices! Saved in 'milestone1/dataset/'")

if __name__ == "__main__":
    generate_dataset(100)
