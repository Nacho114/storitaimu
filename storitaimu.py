from dotenv import load_dotenv
load_dotenv()
import asyncio
from llama_index.readers.whisper import WhisperReader
from llama_index.llms.openai import OpenAI
from typing import Literal, List, Dict, Optional
from pydantic import BaseModel, Field
import os
import shutil
import json
from datetime import datetime
import uuid
import re

class TranscriptMetrics(BaseModel):
    """Basic metrics about the transcript content."""
    word_count: int = Field(..., description="Total number of words in the transcript")
    filler_words: Dict[str, int] = Field(default_factory=dict, description="Dictionary of filler words and their counts")
    total_filler_words: int = Field(default=0, description="Total count of all filler words used")

class ContentReview(BaseModel):
    """LLM's analysis of the content and story."""
    summary: str = Field(..., description="A concise summary of the main points and content")
    story_strength: Literal["weak", "average", "good", "strong", "excellent"] = Field(
        ..., description="Assessment of how effectively the content is presented")
    story_length: Literal["too short", "just right", "too long"] = Field(
        ..., description="Evaluation of content length appropriateness")
    suggestions: List[str] = Field(..., description="Specific recommendations for improvement")

class AudioAnalysis(BaseModel):
    """Complete analysis results."""
    analysis_id: str = Field(..., description="Unique identifier for this analysis")
    filename: str = Field(..., description="Name of the analyzed audio file")
    timestamp: str = Field(..., description="When the analysis was performed")
    metrics: TranscriptMetrics
    review: ContentReview
    
    def save_json(self, filepath: str):
        """Save the analysis to a JSON file."""
        with open(filepath, 'w') as f:
            json.dump(self.model_dump(), f, indent=4)

def find_audio_file() -> Optional[str]:
    """
    Look for audio files in the current directory.
    Returns the first audio file found or None if no audio files are present.
    """
    # Common audio extensions we support
    audio_extensions = ('.mp3', '.m4a', '.wav', '.aac', '.mp4')
    
    # List all files in current directory
    files = os.listdir('.')
    
    # Find the first audio file
    for file in files:
        if file.lower().endswith(audio_extensions):
            return file
            
    return None

def analyze_filler_words(transcript: str) -> Dict[str, int]:
    """Find and count filler words in the transcript."""
    common_fillers = {
        'um': r'\bum\b',
        'uh': r'\buh\b',
        'like': r'\blike\b',
        'you know': r'\byou know\b',
        'sort of': r'\bsort of\b',
        'kind of': r'\bkind of\b'
    }
    
    filler_counts = {}
    for filler, pattern in common_fillers.items():
        count = len(re.findall(pattern, transcript.lower()))
        if count > 0:
            filler_counts[filler] = count
            
    return filler_counts

async def analyze_audio():
    """Process audio file and generate analysis."""
    try:
        # Look for an audio file
        print("\n🔍 Looking for audio files...")
        audio_file = find_audio_file()
        
        if not audio_file:
            print("❌ No audio files found in current directory!")
            print("Please add an audio file (mp3, m4a, wav, aac, or mp4) to this directory.")
            return
        
        print(f"✅ Found audio file: {audio_file}")
        
        # Generate analysis ID and create folder name
        analysis_id = str(uuid.uuid4())[:8]  # Using first 8 characters for shorter name
        folder_name = f"{analysis_id}-{os.path.splitext(audio_file)[0]}"
        folder_path = os.path.join('data', folder_name)
        
        print(f"\n📂 Creating analysis folder: {folder_path}")
        os.makedirs(folder_path, exist_ok=True)
        
        # Move the audio file
        print("\n🔄 Moving audio file...")
        new_audio_path = os.path.join(folder_path, audio_file)
        shutil.move(audio_file, new_audio_path)
        
        # Transcribe the audio
        print("\n🎤 Transcribing audio...")
        reader = WhisperReader(model="whisper-1")
        documents = await reader.aload_data(new_audio_path)
        transcript = documents[0].text
        
        # Analyze the transcript
        print("\n📊 Analyzing transcript...")
        words = transcript.split()
        word_count = len(words)
        
        # Find filler words
        filler_counts = analyze_filler_words(transcript)
        total_fillers = sum(filler_counts.values())
        
        # Create metrics object
        metrics = TranscriptMetrics(
            word_count=word_count,
            filler_words=filler_counts,
            total_filler_words=total_fillers
        )
        
        # Get the content review from LLM
        print("\n🤖 Getting content review...")
        llm = OpenAI(model="gpt-4")
        prompt = f"""
        Review this transcript, taking note that it contains {total_fillers} filler words.
        
        Transcript: {transcript}
        
        Provide a review with:
        1. A brief summary of the content
        2. The story strength (choose one: weak, average, good, strong, excellent)
        3. The story length (choose one: too short, just right, too long)
        4. 1-2 suggestions for improvement
        
        Format your response as a JSON object with these exact keys: summary, story_strength, story_length, suggestions
        """
        
        response = llm.complete(prompt)
        review_data = json.loads(response.text)
        
        # Create the review object
        review = ContentReview(
            summary=review_data['summary'],
            story_strength=review_data['story_strength'],
            story_length=review_data['story_length'],
            suggestions=review_data['suggestions']
        )
        
        # Create the complete analysis
        analysis = AudioAnalysis(
            analysis_id=analysis_id,
            filename=audio_file,
            timestamp=datetime.now().isoformat(),
            metrics=metrics,
            review=review
        )
        
        # Save the results
        print("\n💾 Saving results...")
        
        # Save the transcript
        with open(os.path.join(folder_path, 'transcript.txt'), 'w') as f:
            f.write(transcript)
            
        # Save the analysis
        analysis.save_json(os.path.join(folder_path, 'analysis.json'))
        
        # Print a summary
        print("\n✨ Analysis complete!")
        print(f"📁 Results saved in: {folder_path}")
        print(f"\n📊 Quick Stats:")
        print(f"- Analysis ID: {analysis_id}")
        print(f"- Total words: {metrics.word_count}")
        print(f"- Filler words: {metrics.total_filler_words}")
        if metrics.filler_words:
            print("- Common fillers used:", ", ".join(f"{word} ({count}x)" 
                  for word, count in metrics.filler_words.items()))
        
    except Exception as e:
        print(f"\n❌ Error: {str(e)}")
        raise

if __name__ == "__main__":
    os.makedirs('data', exist_ok=True)
    asyncio.run(analyze_audio())
