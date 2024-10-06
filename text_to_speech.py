import pyttsx3
import threading

class TextToSpeech:
    def __init__(self):
        self.engine = pyttsx3.init()
        self.engine.setProperty('rate', 150)
        self.engine.setProperty('volume', 1.0)
        self.pronunciation_map = {
            'A': 'A',
            'B': 'B',
            'C': 'C',
            'D': 'D',
            'E': 'E',
            'F': 'F',
            'G': 'G',
            'H': 'H',
            'I': 'I',
            'J': 'J',
            'K': 'K',
            'L': 'L',
            'M': 'M',
            'N': 'N',
            'O': 'O',
            'P': 'P',
            'Q': 'Q',
            'R': 'R',
            'S': 'S',
            'T': 'T',
            'U': 'U',
            'V': 'V',
            'W': 'double u',
            'X': 'X',
            'Y': 'Y',
            'Z': 'Z',
            'del': 'delete',
            'nothing': 'nothing',
            'space': 'space'
        }
        self.last_spoken_prediction = None
        self.lock = threading.Lock()
    def speak(self, prediction):
        with self.lock:
            if prediction != self.last_spoken_prediction:
                self.last_spoken_prediction = prediction
                threading.Thread(target=self._speak_prediction, args=(prediction,), daemon=True).start()
    def _speak_prediction(self, prediction):
        try:
            spoken_text = self.pronunciation_map.get(prediction, prediction)
            self.engine.say(spoken_text)
            self.engine.runAndWait()
        except Exception as e:
            print(f"Error in TTS: {e}")
    def reset(self):
        with self.lock:
            self.last_spoken_prediction = None
