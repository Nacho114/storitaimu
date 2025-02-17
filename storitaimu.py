from dotenv import load_dotenv
load_dotenv()
import asyncio
from llama_index.readers.whisper import WhisperReader
from llama_index.llms.openai import OpenAI
from typing import Literal, List
from pydantic import BaseModel, Field
import os
import shutil
import json
import uuid
from datetime import datetime

# Data Models for transcript and story analysis
class TranscriptPatternAnalysis(BaseModel):
    """
    Analysis of text patterns and filler words from the transcript.
    This analysis is based solely on the transcribed text, not the actual audio.
    It gives clear and concrete recommendations, rather than vague statements.
    """
    found_filler_words: List[str] = Field(
        ...,
        description="List of all filler words and phrases identified in the transcript, including hesitations (um, uh), hedging phrases (sort of, kind of), repetitive transitions (and then, so), and unnecessary qualifiers (I think, basically)"
    )
    speaking_suggestions: List[str] = Field(
        ...,
        description="Recommendations for clearer communication based on the transcript, such as using strategic silence instead of filler words, structuring thoughts before speaking, and employing more confident language patterns"
    )

class StoryStructureReview(BaseModel):
    """
    Analysis of the story's content, structure, and effectiveness.
    Evaluates how well the narrative is constructed and delivered.
    It gives clear and concrete recommendations, rather than vague statements.
    """
    summary: str = Field(
        ..., 
        description="A concise summary of the main message, key points, and overall narrative arc"
    )
    story_strength: Literal["weak", "average", "good", "strong", "excellent"] = Field(
        ..., 
        description="Overall assessment of how effectively the content is structured and presented"
    )
    story_length: Literal["too short", "just right", "too long"] = Field(
        ...,
        description="Evaluation of content length appropriateness"
    )
    narrative_suggestions: List[str] = Field(
        ...,
        description="Specific recommendations for improving story structure, impact, and audience engagement"
    )

def find_audio_file() -> str | None:
    """
    Find the first audio file in the current directory.
    Supports common audio formats: mp3, m4a, wav, aac, and mp4.
    """
    audio_extensions = ('.mp3', '.m4a', '.wav', '.aac', '.mp4')
    for file in os.listdir('.'):
        if file.lower().endswith(audio_extensions):
            return file
    return None

async def analyze_audio():
    """
    Process audio file and generate comprehensive analysis of transcribed content.
    Creates a unique folder for each analysis and saves all results.
    """
    try:
        # Look for an audio file in the current directory
        print("\n🔍 Looking for audio files...")
        audio_file = find_audio_file()
        
        if not audio_file:
            print("❌ No audio files found in current directory!")
            print("Please add an audio file (mp3, m4a, wav, aac, or mp4) to this directory.")
            return
        
        print(f"✅ Found audio file: {audio_file}")
        
        # Create a unique analysis folder
        analysis_id = str(uuid.uuid4())[:8]
        folder_name = f"{analysis_id}-{os.path.splitext(audio_file)[0]}"
        folder_path = os.path.join('data', folder_name)
        os.makedirs(folder_path, exist_ok=True)
        
        # Move the audio file to the analysis folder
        print("\n🔄 Moving audio file...")
        new_audio_path = os.path.join(folder_path, audio_file)
        shutil.move(audio_file, new_audio_path)
        
        # Generate transcript using Whisper
        print("\n🎤 Transcribing audio...")
        reader = WhisperReader(model="whisper-1")
        documents = await reader.aload_data(new_audio_path)
        transcript = documents[0].text
        
        # Initialize OpenAI LLM for analysis
        llm = OpenAI(model="gpt-4o")
        
        # Analyze transcript patterns (filler words and language patterns)
        print("\n🔍 Analyzing transcript patterns...")
        sllm = llm.as_structured_llm(TranscriptPatternAnalysis)
        pattern_completion = sllm.complete(transcript)
        
        # Analyze story structure and content
        print("\n🤖 Analyzing story structure...")
        sllm = llm.as_structured_llm(StoryStructureReview)
        story_completion = sllm.complete(transcript)
       
        # Save all analysis results
        print("\n💾 Saving results...")

        # Save transcript pattern analysis
        pattern_analysis_path = os.path.join(folder_path, 'pattern_analysis.json')
        with open(pattern_analysis_path, 'w') as f:
            pattern_data = json.loads(pattern_completion.text)
            json.dump(pattern_data, f, indent=4)
            
        # Save story structure analysis
        story_review_path = os.path.join(folder_path, 'story_review.json')
        with open(story_review_path, 'w') as f:
            story_data = json.loads(story_completion.text)
            json.dump(story_data, f, indent=4)
            
        # Save the raw transcript
        transcript_path = os.path.join(folder_path, 'transcript.txt')
        with open(transcript_path, 'w') as f:
            f.write(transcript)
            
        print(f"\nAnalysis completed successfully!")
        print(f"📁 All files saved in: {folder_path}")
        print(f"📊 Pattern analysis: {pattern_analysis_path}")
        print(f"📝 Story review: {story_review_path}")
        print(f"📄 Transcript: {transcript_path}")

    except Exception as e:
        print(f"\n❌ Error: {str(e)}")
        raise

# Script entry point
if __name__ == "__main__":
    # Ensure data directory exists
    os.makedirs('data', exist_ok=True)
    # Run the analysis
    asyncio.run(analyze_audio())
