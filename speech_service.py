import whisper
import tempfile
import os

class SpeechService:
    def __init__(self):
        self.model = whisper.load_model("base")  # If using 'openai/whisper' pip package

        # If you are using 'whisper' from 'whisper.cpp' or other implementation, use:
        # self.model = whisper.Whisper("base")

    def transcribe_audio(self, audio_file) -> str:
        try:
            if isinstance(audio_file, bytes):
                with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp_file:
                    tmp_file.write(audio_file)
                    audio_path = tmp_file.name
            else:
                audio_path = audio_file

            result = self.model.transcribe(audio_path)
            return result["text"]
        except Exception as e:
            return f"Transcription error: {str(e)}"
    
    def is_available(self) -> bool:
        # Check if essential keys or dependencies are configured
        return os.getenv('OPENAI_API_KEY') is not None  # Modify or add checks as needed