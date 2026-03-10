"""
This module adds bacground music to generate podcast audio
The function overlays a music track behind the voice recording
automatically adjusting the volume and looping it if the voice
track is longer

Main resposibility
-Load the voice podcast audio
-Load background music
-Reduce music volume  so it does not overpower the voices
-Loop the music if needed
-Export the final combined podcast
"""
import os
import logging
from pydub import AudioSegment

#Create a logger for debugging and warnings
logger = logging.getLogger(__name__)

def add_background(voice_file: str, music_file: str, output_file: str="final_podcast.mp3", music_volume_db: int = -25) ->str:
    """
    Voice_file : path to generate podcast voice audio
    music_file : path to the background music file
    output_file: Name of final exported podcast file
    music_volume_db: Volume reduction applied to music
    """
    #Ensure the main voice audio exists
    if not os.path.exists(voice_file):
        raise FileNotFoundError(f"Voice File not found:{voice_file}")
    
    #If background music is missing , return the voice file only
    if not os.path.exists(music_file):
        logger.warning(f"Music File not found:{music_file}")
        return voice_file
    
    #Load the voice track
    voice = AudioSegment.from_file(voice_file)

    #Load music track
    music = AudioSegment.from_file(music_file)

    # Lower the background music volume so it stay subtle behind the voice
    music = music + music_volume_db
    
    #If the music is shorter than voice , repeat it
    if len(music) < len(voice):
        loops = int(len(voice)/ len(music)) + 1
        music = music * loops

    # trim music to match voice length
    music = music[:len(voice)]

    # Overlay the voice and music
    combined = voice.overlay(music)
    print("Overlaid audio")  
    
    #Export the final podcast
    combined.export(output_file, format="mp3")

    return output_file


if __name__ == "__main__":
    # Requires podcast.mp3 and background.mp3 to exist
    if os.path.exists("podcast.mp3") and os.path.exists("background.mp3"):
        result = add_background("podcast.mp3", "background.mp3")
        print("Final audio:", result)
    else:
        print("Test skipped: podcast.mp3 or background.mp3 not found.")