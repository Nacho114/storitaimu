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

class FillerWordAnalysis(BaseModel):
    """
    Comprehensive analysis of speaking patterns, filler words, and vocal delivery.
    The LLM will analyze both technical patterns and their impact on the speech.
    """
    found_filler_words: List[str] = Field(
        ...,
        description="List of all filler words and phrases identified in the transcript, including hesitations (um, uh), hedging phrases (sort of, kind of), repetitive transitions (and then, so), and unnecessary qualifiers (I think, basically)"
    )
    pacing_observations: List[str] = Field(
        ...,
        description="Observations about speaking rhythm, including rushed sections, pauses, breathing patterns, and natural flow versus mechanical delivery"
    )
    vocal_delivery: List[str] = Field(
        ...,
        description="Analysis of voice modulation, including changes in pitch, volume, emphasis patterns, and emotional expression"
    )
    speaking_recommendations: List[str] = Field(
        ...,
        description="Specific suggestions for improving vocal delivery, including strategic use of silence, pacing adjustments, and techniques for reducing filler words"
    )

class ContentReview(BaseModel):
    """
    Detailed analysis of speech content, structure, and storytelling effectiveness.
    Focuses on both the technical and artistic elements of the presentation.
    """
    summary: str = Field(
        ..., 
        description="A concise summary of the main message, key points, and overall narrative arc"
    )
    story_structure: List[str] = Field(
        ...,
        description="Analysis of speech components including opening hook, transitions, main points organization, and conclusion effectiveness"
    )
    storytelling_elements: List[str] = Field(
        ...,
        description="Evaluation of narrative techniques including scene-setting, character development, emotional resonance, and memorable moments"
    )
    audience_engagement: List[str] = Field(
        ...,
        description="Analysis of engagement techniques like rhetorical questions, inclusive language, relatable examples, and audience connection moments"
    )
    story_strength: Literal["weak", "average", "good", "strong", "excellent"] = Field(
        ..., 
        description="Overall assessment of how effectively the content is presented and resonates with the audience"
    )
    story_length: Literal["too short", "just right", "too long"] = Field(
        ...,
        description="Evaluation of content length and pacing appropriateness"
    )
    improvement_suggestions: List[str] = Field(
        ...,
        description="Specific recommendations for improving content structure, storytelling impact, and audience engagement"
    )

# Helper Functions

def find_audio_file() -> str | None:
    """Find the first audio file in the current directory."""
    audio_extensions = ('.mp3', '.m4a', '.wav', '.aac', '.mp4')
    for file in os.listdir('.'):
        if file.lower().endswith(audio_extensions):
            return file
    return None

# Main Analysis Function
async def analyze_audio():
    """Process audio file and generate comprehensive analysis."""
    try:
        # Look for an audio file
        print("\n🔍 Looking for audio files...")
        audio_file = find_audio_file()
        
        if not audio_file:
            print("❌ No audio files found in current directory!")
            print("Please add an audio file (mp3, m4a, wav, aac, or mp4) to this directory.")
            return
        
        print(f"✅ Found audio file: {audio_file}")
        
        # Set up analysis folder with unique ID
        analysis_id = str(uuid.uuid4())[:8]
        folder_name = f"{analysis_id}-{os.path.splitext(audio_file)[0]}"
        folder_path = os.path.join('data', folder_name)
        os.makedirs(folder_path, exist_ok=True)
        
        # Move the audio file to analysis folder
        print("\n🔄 Moving audio file...")
        new_audio_path = os.path.join(folder_path, audio_file)
        shutil.move(audio_file, new_audio_path)
        
        # Transcribe the audio using Whisper
        print("\n🎤 Transcribing audio...")
        reader = WhisperReader(model="whisper-1")
        documents = await reader.aload_data(new_audio_path)
        transcript = documents[0].text
        
        # Initialize OpenAI LLM for analysis
        llm = OpenAI(model="gpt-4o-mini")
        
        # Analyze filler words
        print("\n🔍 Analyzing filler words...")
        sllm = llm.as_structured_llm(FillerWordAnalysis)
        filler_completion = sllm.complete(transcript)
        
        # Generate content review
        print("\n🤖 Generating content review...")
        sllm = llm.as_structured_llm(ContentReview)
        review_completion = sllm.complete(transcript)
       
        # Save results to files
        print("\n💾 Saving results...")

        # Save both analyses to JSON files
        filler_analysis_path = os.path.join(folder_path, 'filler_analysis.json')
        with open(filler_analysis_path, 'w') as f:
            # Parse the JSON string from the text field and save directly
            filler_data = json.loads(filler_completion.text)
            json.dump(filler_data, f, indent=4)
            
        content_review_path = os.path.join(folder_path, 'content_review.json')
        with open(content_review_path, 'w') as f:
            # Parse the JSON string from the text field and save directly
            content_data = json.loads(review_completion.text)
            json.dump(content_data, f, indent=4)
            
        # Save the raw transcript
        transcript_path = os.path.join(folder_path, 'transcript.txt')
        with open(transcript_path, 'w') as f:
            f.write(transcript)

    except Exception as e:
        print(f"\n❌ Error: {str(e)}")
        raise

# Script Entry Point
if __name__ == "__main__":
    # Ensure data directory exists
    os.makedirs('data', exist_ok=True)
    # Run the analysis
    asyncio.run(analyze_audio())
