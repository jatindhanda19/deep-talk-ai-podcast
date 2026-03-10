""" This module converts a podcast script into a multi-voice audio file using MeloTTS
    It supports multi speaker, accents and different speech speeds

    Key Features:
    -Parser that script into speaker-text segments
    -Preloads TTS models for required languages
    -Generates audio segments concurrently for speed
    -Merges all segments into single MP3 file
    -Adds short silence pauses between segments
"""
# Library import
import os
import tempfile
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from melo.api import TTS
from pydub import AudioSegment

#Logger setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Constants 
DEVICE = "cpu"  # Device for TTS(CPU or GPU)
SILENCE_MS = 300 # Silence between segments in millisecods

#Configuration for different speakers
VOICE_MAP ={
     "Host":     {"language": "EN", "speaker": "EN-US",  "speed": 1.0},
    "Expert":   {"language": "EN", "speaker": "EN-BR",  "speed": 0.95},
    "Expert_A": {"language": "EN", "speaker": "EN-BR",  "speed": 0.95},
    "Expert_B": {"language": "EN", "speaker": "EN-AU",  "speed": 1.0}

}

#cache to store loaded TTS models for reuse
_model_cache: dict[str, TTS]= {}

#Helper Fuctions
def _get_model(language:str) -> TTS:
    """Load or retrieve the TTS models for a given language
       Uses caching to avoid reloading models multiple times
    """
    if language not in _model_cache:
        logger.info("Loading MeloTTS model:%s", language)
        _model_cache[language] = TTS(language=language, device=DEVICE)
    return _model_cache[language]

def _preload_models(segments: list[tuple[str,str]]) -> None:
    """Preloads all necessary TTS models based on script segments
    """
    languages_needed = {
        VOICE_MAP[spk]["language"]
        for spk, _ in segments
        if spk in VOICE_MAP
    }
    for language in languages_needed:
        _get_model(language)
        logger.info("Preloaded model for language:%s", language)

def split_script(script:str)-> list[tuple[str,str]]:
    """Convert a podcast script into a list of (speaker, text) tuples
      skips blank lines and lines that are not in "Speaker:text" format.
    """

    segments = []
    for line in script.strip().split("\n"):
        line = line.strip()
        if line.lower().startswith("title:"):
            continue
        if ":" in line:
            speaker,text = line.split(":",1)
            speaker = speaker.strip()
            text = text.strip()
            if speaker and text:
                segments.append((speaker,text))
    return segments

def _generate_segment(idx: int, speaker: str , text: str) -> tuple[int,str] | None:
    """Generate a single audio segments for a given speaker and text
       Returns the segment index and path to temporary WAV file
    """
    if speaker not in VOICE_MAP:
        logger.warning("Unknown speaker '%s' -- skipping",speaker)
        return None
    
    cfg = VOICE_MAP[speaker]
    language = cfg["language"]
    accent = cfg["speaker"]
    speed = cfg["speed"]

    model = _get_model(language)
    speaker_ids = model.hps.data.spk2id
    
    # IF accent not found in model , fallback to default
    if accent not in speaker_ids:
        logger.warning("Accent '%s' not found -- using EN-default.", accent)
        accent="EN-Default"
    
    #Temporary file for this segment
    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    tmp.close()

    # Generates TTS audio
    model.tts_to_file(
        text=text,
        speaker_id=speaker_ids[accent],
        output_path=tmp.name,
        speed=speed
    )

    return (idx, tmp.name)

#Main Function
def generate_multi_voice_audio(script:str ,output_file: str = "podcast.mp3", max_workers: int= 2) -> str:
    """Converts a podcast script to multi-voice  MP3 audio file
       steps:
       1. Parse script into speaker-text segments
       2.Preload all required TTS models
       3.Generate each segment concurrently
       4.Merge segments into a single audio file with silence in between
       5.Clean up  temporary WAV files
    """
    #Handle input in list-of-dict or list-of-string format
    if isinstance(script, list):
        if script and isinstance(script[0], dict):
                script = "\n".join([f"{item.get('speaker', 'Unknown')}: {item.get('text')}" for item in script])
        else:
            script = "\n".join(script)
        
    segments = split_script(script)
    if not segments:
        raise ValueError("No valid speaker segments found in script")
    
    _preload_models(segments)
    results: dict[int, str] = {}
    
    #Generate segments in parallel
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(_generate_segment, i, spk, txt): i
            for i, (spk,txt) in enumerate(segments)
        }

        for future in as_completed(futures):
            result = future.result()
            if result:
                idx, wav_path = result
                results[idx] = wav_path
 
    # Merge segments into final audio
    final_audio = AudioSegment.empty()
    for i in sorted(results):
        seg = AudioSegment.from_wav(results[i])
        final_audio += seg + AudioSegment.silent(duration=SILENCE_MS)
        os.remove(results[i])

    if len(final_audio) == 0:
        raise RuntimeError("No audio generated. Check speaker names and MeloTTS")
    
    # Export final audio
    final_audio.export(output_file, format="mp3", bitrate="128k")
    logger.info("podcast Saved %s", output_file)
    return output_file
    
if __name__ == "__main__":

    test_script = """
Host: Welcome to about AI podcast.
Expert: Today we will discuss langchain in deatil.
Host: That sounds interesting. Can you explain it simply?
Expert: Of course. RAG combines document  language models.
"""

    result = generate_multi_voice_audio(test_script)
    print("Podcast generated:", result)

