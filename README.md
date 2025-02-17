# Audio Analysis Tool

This tool automatically transcribes audio files and provides a detailed content analysis using GPT-4. It generates insights about speaking patterns, identifies filler words, and offers suggestions for improvement.

## Features

- Automatic audio file detection and processing
- Speech-to-text transcription using Whisper
- Content analysis using GPT-4o-mini
- Filler word detection and counting
- Organized output with unique IDs for each analysis

## Setup

1. First, clone this repository:
   ```bash
   git clone [your-repo-url]
   cd [your-repo-name]
   ```

2. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install the required packages:
   ```bash
   pip install python-dotenv llama-index openai mutagen
   ```

4. Create a `.env` file in the project root and add your OpenAI API key:
   ```plaintext
   OPENAI_API_KEY=your_api_key_here
   ```

## Usage

1. Place any audio file (supported formats: mp3, m4a, wav, aac, mp4) in the project's root directory.

2. Run the analysis script:
   ```bash
   python review_engine.py
   ```

3. The script will automatically:
   - Find the first audio file in the directory
   - Create a uniquely identified folder in `data/`
   - Generate a transcript and analysis
   - Save all results in the created folder

## Output

The tool creates a folder for each analysis with the following structure:
```
data/
  [unique-id]-[original-filename]/
    - original-audio-file
    - transcript.txt
    - analysis.json
```

The analysis.json file contains:
- A unique analysis ID
- Word counts and metrics
- Filler word analysis
- Content review with:
  - Summary
  - Story strength assessment
  - Length evaluation
  - Improvement suggestions

## Example Output

```json
{
    "analysis_id": "a1b2c3d4",
    "filename": "my_audio.mp3",
    "timestamp": "2024-02-17T14:30:00",
    "metrics": {
        "word_count": 150,
        "filler_words": {
            "um": 3,
            "like": 2
        },
        "total_filler_words": 5
    },
    "review": {
        "summary": "A discussion about...",
        "story_strength": "good",
        "story_length": "just right",
        "suggestions": [
            "Consider adding more specific examples",
            "Try to reduce the use of filler words"
        ]
    }
}
```

## Requirements

- Python 3.8 or higher
- OpenAI API key
- Internet connection for transcription and analysis

## Notes

- Audio files are moved from the root directory to their analysis folder during processing
- Each analysis gets a unique ID for easy reference
- The tool will process the first audio file it finds in the root directory
- Keep your `.env` file secure and never commit it to version control

## Troubleshooting

If you encounter issues:

1. Make sure your `.env` file contains a valid OpenAI API key
2. Check that your audio file is in a supported format
3. Ensure you have an active internet connection
4. Verify that the audio file is in the project's root directory

## Contributing

Feel free to submit issues and enhancement requests!
