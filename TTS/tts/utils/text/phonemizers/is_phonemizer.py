from TTS.tts.utils.text.phonemizers.base import BasePhonemizer
 
from ice_g2p.transcriber import Transcriber
from ice_g2p.transcriber import G2P_METHOD

AVAILABLE_DIALECTS = ['standard', 'north']


def process_string(input_str: str, dialect='standard', use_dict=False, syllab_symbol='', word_sep='',
                   stress_label=False, lang_detect=False) -> str:
    g2p = Transcriber(G2P_METHOD.FAIRSEQ, dialect=dialect, lang_detect=lang_detect, syllab_symbol=syllab_symbol, word_sep=word_sep,
                      stress_label=stress_label, use_dict=use_dict)
    return g2p.transcribe(input_str)


class Ice_G2P_Phonemizer(BasePhonemizer):

    def _init_language(self, language):
        if not self.is_supported_language(language):
            raise RuntimeError(f'language "{language}" is not supported by the ' f"{self.name()} backend")
        
        dialect = "standard"
        if language == "is-north":
            dialect = "north"

        self._g2p = Transcriber(G2P_METHOD.FAIRSEQ, dialect=dialect, lang_detect=False, syllab_symbol="", word_sep="",
                      stress_label=False, use_dict=False)

        return language

    @staticmethod
    def name():
        return "ice_g2p"

    @classmethod
    def is_available(cls):
        try:
            import ice_g2p
            return True
        except ImportError:
            return False

    @classmethod
    def version(cls):
        return "0.0.1"

    @staticmethod
    def supported_languages():
        return {
            "is-is": "Icelandic (Iceland)",
            "is-north": "Icelandic (North Iceland)"
        }

    def _phonemize(self, text, separator):
        return self._g2p.transcribe(text)

    def phonemize(self, text: str, separator="|") -> str:
        phonemized = self._phonemize(text, separator)
        return phonemized

    def print_logs(self, level: int = 0):
        indent = "\t" * level
        print(f"{indent}| > phoneme language: {self.language}")
        print(f"{indent}| > phoneme backend: {self.name()}")
