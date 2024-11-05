from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from llama_cpp import Llama
import yt_dlp
from typing import Optional, List, Tuple
import uvicorn
import logging
import json
import random
import requests
import re
from video_processor import VideoProcessor
from fastapi.responses import FileResponse
import os

import whisper
import numpy as np
import tempfile
from pathlib import Path
import subprocess
import shutil
from typing import List

# Initialize video processor
video_processor = VideoProcessor()

class TransformTextRequest(BaseModel):
    text: str

class VideoRequest(BaseModel):
    video_id: str
    timestamp: Optional[int] = None

class TweetRequest(BaseModel):
    transformed_text: str
    video_title: str

class VideoTitleRequest(BaseModel):
    video_id: str

class ReplyRequest(BaseModel):
    tweet_text: str
    test_mode: Optional[bool] = False

class ReplyResponse(BaseModel):
    response_text: str
    video_id: str
    timestamp: int
    similarity_score: float

PLAYLIST_ID = "PLTgvaF3a9YMipq-kfDW1Tkrl-sG7eTQqV"
MAX_RETRIES = 3

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize Whisper model globally after your other initializations
logger.info("Initializing Whisper model...")
whisper_model = whisper.load_model("tiny")
logger.info("Whisper model initialized successfully")

app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize LLaMa model
logger.info("Initializing LLaMa model...")
llm = Llama(
    model_path="models/llama-2-7b-chat.Q4_K_M.gguf",
    n_gpu_layers=32,
    n_ctx=4096,
    n_threads=4
)
logger.info("LLaMa model initialized successfully")


async def get_video_title(video_id: str) -> Optional[str]:
    """Get the title of a YouTube video"""
    ydl_opts = {
        'quiet': True,
        'no_warnings': True,
        'extract_flat': True
    }
    
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(f"https://www.youtube.com/watch?v={video_id}", download=False)
            return info.get('title')
    except Exception as e:
        logger.error(f"Error fetching video title: {e}")
        return None


def transcribe_video_file(video_path: str) -> Optional[str]:
    """Transcribe a video file using Whisper with improved file handling"""
    temp_audio = None
    try:
        # Ensure ffmpeg is available
        if not shutil.which('ffmpeg'):
            raise Exception("ffmpeg not found in system PATH")

        # Create temp file with explicit permissions
        temp_audio = tempfile.NamedTemporaryFile(
            suffix='.wav',
            delete=False,
            mode='wb',
            prefix='whisper_audio_'
        )
        temp_audio_path = temp_audio.name
        temp_audio.close()  # Close file handle immediately

        # Use subprocess.run instead of os.system
        try:
            process = subprocess.run([
                'ffmpeg',
                '-i', video_path,
                '-ac', '1',
                '-ar', '16000',
                '-y',
                temp_audio_path
            ], capture_output=True, text=True)

            if process.returncode != 0:
                logger.error(f"FFmpeg error: {process.stderr}")
                raise Exception(f"FFmpeg failed: {process.stderr}")

            # Verify the audio file exists and is readable
            if not os.path.exists(temp_audio_path):
                raise Exception(f"Audio file not created: {temp_audio_path}")

            # Load audio and transcribe
            audio = whisper.load_audio(temp_audio_path)
            result = whisper_model.transcribe(
                audio,
                language="en",
                fp16=False,
                max_initial_timestamp=6.0
            )
            
            text = result["text"].strip()
            text = text.replace("[Music]", "").replace("[Applause]", "").strip()
            
            logger.info(f"Transcription result: {text}")
            return text

        except subprocess.SubprocessError as e:
            logger.error(f"FFmpeg subprocess error: {str(e)}")
            raise
            
    except Exception as e:
        logger.error(f"Error in transcribe_video_file: {e}")
        return None
        
    finally:
        # Clean up temp files
        try:
            if temp_audio is not None and os.path.exists(temp_audio.name):
                os.chmod(temp_audio.name, 0o666)  # Ensure we have permission to delete
                os.unlink(temp_audio.name)
        except Exception as e:
            logger.warning(f"Failed to clean up temp audio file: {e}")

def convert_timestamp_to_seconds(timestamp: str) -> int:
    """Convert HH:MM:SS or MM:SS timestamp to seconds"""
    try:
        # Remove any trailing periods or spaces and brackets
        timestamp = timestamp.strip(' .[]')
        
        # Try different timestamp formats
        if ':' in timestamp:
            parts = timestamp.split(':')
            if len(parts) == 2:  # MM:SS
                minutes, seconds = parts
                return int(minutes) * 60 + int(seconds)
            elif len(parts) == 3:  # HH:MM:SS
                hours, minutes, seconds = parts
                return int(hours) * 3600 + int(minutes) * 60 + int(seconds)
        
        return int(timestamp)
    except (ValueError, AttributeError):
        logger.warning(f"Could not parse timestamp: {timestamp}")
        return 0

def get_chunks(transcript: str, chunk_size: int = 3) -> List[Tuple[str, str]]:
    """Split transcript into small chunks, each with its timestamp"""
    segments = transcript.split('\n')
    chunks = []
    current_chunk = []
    chunk_start_time = None
    
    for segment in segments:
        # Look for timestamp at start of line
        timestamp_match = re.match(r'\[(.*?)\]', segment)
        if timestamp_match:
            if not chunk_start_time:
                chunk_start_time = timestamp_match.group(1)
            
            current_chunk.append(segment)
            
            # When we hit our chunk size, save the chunk and reset
            if len(current_chunk) >= chunk_size:
                chunks.append((chunk_start_time, '\n'.join(current_chunk)))
                current_chunk = []
                chunk_start_time = None
    
    # Add any remaining lines
    if current_chunk:
        chunks.append((chunk_start_time, '\n'.join(current_chunk)))
    
    return chunks

def get_vtt_chunks(transcript: str, chunk_size: int = 3) -> List[Tuple[str, str]]:
    """Split transcript into chunks by timestamp"""
    segments = transcript.split('\n')
    logger.info(f"Splitting transcript into chunks, got {len(segments)} segments")
    
    chunks = []
    current_text = []
    current_time = None
    
    for line in segments:
        # Look for timestamp in square brackets at start of line
        timestamp_match = re.match(r'\[([\d:\.]+)\]', line)
        if timestamp_match:
            # If we have accumulated text from previous timestamp
            if current_time and current_text:
                # Clean up the text by removing HTML-like tags and extra whitespace
                clean_text = re.sub(r'<[^>]+>', '', ' '.join(current_text)).strip()
                chunks.append((current_time, clean_text))
                current_text = []
            
            # Get the new timestamp
            current_time = timestamp_match.group(1)
            # Get the text after the timestamp
            text = re.sub(r'\[[\d:\.]+\]\s*', '', line)
            if text.strip():
                current_text.append(text)
        elif line.strip():
            # Clean the line of HTML-like tags before adding
            clean_line = re.sub(r'<[^>]+>', '', line).strip()
            if clean_line:
                current_text.append(clean_line)
    
    # Add the last chunk if exists
    if current_time and current_text:
        clean_text = re.sub(r'<[^>]+>', '', ' '.join(current_text)).strip()
        chunks.append((current_time, clean_text))
    
    logger.info(f"Created {len(chunks)} chunks")
    if chunks:
        logger.info("First few chunks:")
        for i, (time, text) in enumerate(chunks[:3]):
            logger.info(f"Chunk {i}: Time={time}, Text={text}")
    
    return chunks

def convert_vtt_timestamp_to_seconds(timestamp: str) -> int:
    """Convert timestamp (HH:MM:SS.mmm) to seconds"""
    logger.debug(f"Converting timestamp: '{timestamp}'")
    try:
        # Clean up the timestamp
        clean_timestamp = timestamp.strip('[]')
        
        if '.' in clean_timestamp:
            # Split off milliseconds
            clean_timestamp = clean_timestamp.split('.')[0]
        
        # Split hours, minutes, seconds
        parts = clean_timestamp.split(':')
        if len(parts) == 3:  # HH:MM:SS
            hours, minutes, seconds = parts
            total_seconds = (
                int(hours) * 3600 +    # hours to seconds
                int(minutes) * 60 +     # minutes to seconds
                int(seconds)            # seconds
            )
        elif len(parts) == 2:  # MM:SS
            minutes, seconds = parts
            total_seconds = int(minutes) * 60 + int(seconds)
        else:
            logger.warning(f"Unexpected timestamp format: {timestamp}")
            return 0
        
        logger.debug(f"Converted {timestamp} to {total_seconds} seconds")
        return total_seconds
            
    except Exception as e:
        logger.warning(f"Could not parse timestamp {timestamp}: {str(e)}")
        return 0



async def get_playlist_videos() -> List[str]:
    """Get list of video IDs from playlist"""
    ydl_opts = {
        'extract_flat': True,
        'quiet': True,
    }
    
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            playlist_url = f"https://www.youtube.com/playlist?list={PLAYLIST_ID}"
            playlist_info = ydl.extract_info(playlist_url, download=False)
            videos = [entry['id'] for entry in playlist_info['entries'] if entry['id']]
            logger.info(f"Found {len(videos)} videos in playlist")
            return videos
    except Exception as e:
        logger.error(f"Error fetching playlist: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error fetching playlist: {str(e)}")

async def get_video_transcript(video_id: str) -> Optional[str]:
    """Get video transcript, returns None if not found"""
    logger.info(f"Fetching transcript for video {video_id}")
    ydl_opts = {
        'writesubtitles': True,
        'writeautomaticsub': True,
        'subtitlesformat': 'vtt',
        'subtitleslangs': ['en'],
        'skip_download': True,
    }
    
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(f"https://www.youtube.com/watch?v={video_id}", download=False)
            
            if 'automatic_captions' in info and 'en' in info['automatic_captions']:
                captions = info['automatic_captions']['en']
                for caption in captions:
                    if caption['ext'] == 'vtt':
                        response = requests.get(caption['url'])
                        if response.status_code == 200:
                            vtt_content = response.text
                            
                            # Parse VTT to text with timestamps
                            lines = vtt_content.split('\n')
                            transcript = []
                            current_time = None
                            current_text = []
                            
                            for line in lines:
                                if '-->' in line:
                                    if current_time and current_text:
                                        transcript.append(f"[{current_time}] {' '.join(current_text)}")
                                    current_time = line.split('-->')[0].strip()
                                    current_text = []
                                elif line.strip() and not line.startswith('WEBVTT'):
                                    current_text.append(line.strip())
                            
                            if current_time and current_text:
                                transcript.append(f"[{current_time}] {' '.join(current_text)}")
                            
                            return '\n'.join(transcript)
            
            return None
            
    except Exception as e:
        logger.error(f"Error fetching transcript: {str(e)}")
        return None
    
def get_transcript_chunks(transcript: str) -> List[Tuple[str, str]]:
    """Split transcript into chunks by timestamp with clean text"""
    segments = transcript.split('\n')
    chunks = []
    current_chunk = []
    current_time = None
    
    for segment in segments:
        timestamp_match = re.match(r'\[([\d:\.]+)\]', segment)
        if timestamp_match:
            # If we have a previous chunk, process it
            if current_chunk:
                timestamp, text = process_transcript_chunk('\n'.join(current_chunk))
                if timestamp and text:
                    chunks.append((timestamp, text))
                current_chunk = []
            
            current_chunk.append(segment)
        elif segment.strip():
            if current_chunk:  # Only add text if we have a current chunk
                current_chunk.append(segment)
    
    # Process the last chunk
    if current_chunk:
        timestamp, text = process_transcript_chunk('\n'.join(current_chunk))
        if timestamp and text:
            chunks.append((timestamp, text))
    
    logger.info(f"Created {len(chunks)} transcript chunks")
    for i, (time, text) in enumerate(chunks[:3]):
        logger.info(f"Chunk {i}: Time={time}, Text={text}")
    
    return chunks

def process_transcript_chunk(chunk: str) -> Tuple[str, str]:
    """Extract timestamp and clean text from a transcript chunk"""
    # Get the timestamp from the start of the chunk
    timestamp_match = re.match(r'\[([\d:\.]+)\]', chunk)
    if not timestamp_match:
        return None, None
        
    timestamp = timestamp_match.group(1)
    
    # Clean up the text
    text = re.sub(r'\[[\d:\.]+\]', '', chunk)  # Remove timestamp
    text = re.sub(r'<[^>]+>', '', text)        # Remove HTML-like tags
    text = text.strip()                         # Clean whitespace
    
    return timestamp, text

async def find_marc_timestamp(video_id: str) -> Optional[int]:
    """Find a timestamp where Marc begins speaking within first 2 seconds"""
    transcript = await get_video_transcript(video_id)
    if not transcript:
        return None
    
    chunks = get_transcript_chunks(transcript)
    random.shuffle(chunks)
    
    for i, (start_time, chunk) in enumerate(chunks):
        try:
            # Convert timestamp to seconds for this chunk
            chunk_start = convert_vtt_timestamp_to_seconds(start_time)
            
            # Get next chunk's timestamp if available
            next_chunk_start = None
            if i + 1 < len(chunks):
                next_time, _ = chunks[i + 1]
                next_chunk_start = convert_vtt_timestamp_to_seconds(next_time)
            
            # Calculate chunk duration
            chunk_duration = (next_chunk_start - chunk_start) if next_chunk_start else 6
            
            # Split the analysis into smaller segments to ensure Marc starts early
            words = chunk.split()
            estimated_words_per_second = len(words) / chunk_duration
            
            # Only analyze first 2 seconds worth of text to ensure early start
            words_in_two_seconds = int(estimated_words_per_second * 2)
            first_segment = ' '.join(words[:words_in_two_seconds])
            
            # First check if Marc starts in the first 2 seconds
            early_prompt = f"""Does Marc Andreessen start speaking in the first few words of this text? Answer only 'yes' or 'no'.

Text:
{first_segment}

Answer:"""

            early_response = llm(
                early_prompt,
                max_tokens=5,
                temperature=0.1,
                stop=["\n", "\r"],
                echo=False
            )
            
            if early_response['choices'][0]['text'].strip().lower() == 'yes':
                # Double check with a verification prompt
                verify_prompt = f"""Verify if this text BEGINS with Marc Andreessen speaking (not just contains his speech). Must start with his voice. Answer only 'yes' or 'no'.

Text:
{first_segment}

Answer:"""

                verify_response = llm(
                    verify_prompt,
                    max_tokens=5,
                    temperature=0.1,
                    stop=["\n", "\r"],
                    echo=False
                )
                
                if verify_response['choices'][0]['text'].strip().lower() == 'yes':
                    logger.info(f"Found Marc speaking at start of chunk: {start_time}")
                    
                    # Optional: Analyze specific words to further verify
                    common_marc_starts = ['so', 'well', 'yeah', 'right', 'exactly', 'okay']
                    first_word = words[0].lower().strip(',.!?')
                    if first_word in common_marc_starts:
                        logger.info(f"Found likely Marc speech pattern starting with: {first_word}")
                    
                    return chunk_start
                    
        except Exception as e:
            logger.error(f"Error processing chunk {i+1}: {str(e)}")
            continue
    
    return None

async def verify_marc_speaking(text: str) -> bool:
    """Verify if text matches Marc's speech patterns"""
    # First, a simpler prompt that forces a single digit response
    verify_prompt = """Rate if this sounds like Marc Andreessen speaking.
Give ONLY a single digit 0-9, nothing else.

Example response: 8

Text: "%s"

Single digit rating:""" % text

    try:
        verify_response = llm(
            verify_prompt,
            max_tokens=1,      # Force single token
            temperature=0.1,   # Low temperature for consistency
            stop=["\n", "\r", " ", ".", ",", ":", ";", "(", ")", "[", "]"],  # More stop tokens
            echo=False
        )
        
        # Log the raw response for debugging
        logger.info(f"Raw LLama response object: {verify_response}")
        
        # Get the response text
        response_text = verify_response['choices'][0]['text'].strip()
        logger.info(f"Raw text response: '{response_text}'")
        
        # Check if we got a response
        if not response_text:
            logger.warning("Empty response received")
            return False
            
        # Verify it's a single digit
        if not response_text.isdigit() or len(response_text) != 1:
            logger.warning(f"Response is not a single digit: '{response_text}'")
            return False
            
        # Convert to integer
        confidence = int(response_text)
        logger.info(f"Confidence score: {confidence}/9")
        
        # Consider it Marc if confidence is 6 or higher
        return confidence >= 6

    except Exception as e:
        logger.error(f"Error in speech verification: {e}")
        logger.error(f"Full response data: {verify_response if 'verify_response' in locals() else 'No response'}")
        return False

@app.get("/health")
async def health_check():
    return {"status": "ok", "message": "LLM service is running"}

@app.get("/random")
async def get_random_video():
    try:
        videos = await get_playlist_videos()
        random_video = random.choice(videos)
        return {"video_id": random_video}
    except Exception as e:
        logger.error(f"Error getting random video: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/analyze")
async def analyze_video(request: VideoRequest):
    """Find a timestamp where Marc is speaking, with improved verification"""
    videos = await get_playlist_videos()
    tried_videos = set()
    
    # Try the requested video first
    tried_videos.add(request.video_id)
    timestamp = await find_marc_timestamp(request.video_id)
    
    # If timestamp found, verify the clip
    if timestamp is not None:
        try:
            # Get and transcribe a short clip to verify
            success, clip_path = video_processor.get_video_clip(request.video_id, timestamp)
            if success:
                verification_text = transcribe_video_file(clip_path)
                if verification_text:
                    verify_prompt = f"""Does this transcription start with Marc Andreessen speaking? Answer only 'yes' or 'no'.

Text:
{verification_text}

Answer:"""
                    verify_response = llm(
                        verify_prompt,
                        max_tokens=5,
                        temperature=0.1,
                        stop=["\n", "\r"],
                        echo=False
                    )
                    
                    if verify_response['choices'][0]['text'].strip().lower() != 'yes':
                        timestamp = None  # Reset if verification fails
                        logger.warning("Verification failed - Marc not detected in clip")
        except Exception as e:
            logger.error(f"Error during clip verification: {e}")
    
    # If that didn't work, try other random videos
    attempts = 0
    while timestamp is None and attempts < MAX_RETRIES:
        available_videos = [v for v in videos if v not in tried_videos]
        if not available_videos:
            break
            
        video_id = random.choice(available_videos)
        tried_videos.add(video_id)
        
        logger.info(f"Trying alternate video: {video_id}")
        timestamp = await find_marc_timestamp(video_id)
        attempts += 1
    
    if timestamp is not None:
        logger.info(f"Found and verified timestamp: {timestamp}")
        return {"timestamp": timestamp}
    
    raise HTTPException(
        status_code=422,
        detail="Could not find a verified timestamp of Marc speaking"
    )

@app.post("/get_frame")
async def get_video_frame(request: VideoRequest):
    """Get a still frame from the video at the specified timestamp"""
    if request.timestamp is None:
        raise HTTPException(
            status_code=400,
            detail="Timestamp is required"
        )
    
    logger.info(f"Getting frame for video {request.video_id} at timestamp {request.timestamp}")
    
    try:
        success, frame_path = video_processor.get_video_frame(request.video_id, request.timestamp)
        
        if not success or frame_path is None:
            raise HTTPException(
                status_code=500,
                detail="Failed to extract frame from video"
            )
        
        if not os.path.exists(frame_path):
            raise HTTPException(
                status_code=404,
                detail=f"Frame file not found at {frame_path}"
            )
        
        logger.info(f"Returning frame from {frame_path}")
        return FileResponse(
            frame_path,
            media_type="image/jpeg",
            headers={"Content-Disposition": f"inline; filename={os.path.basename(frame_path)}"}
        )
    except Exception as e:
        logger.error(f"Error in get_frame endpoint: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Error processing video frame: {str(e)}"
        )
    
@app.post("/get_clip")
async def get_video_clip(request: VideoRequest):
    """Get a 6-second clip from the video at the specified timestamp"""
    if request.timestamp is None:
        raise HTTPException(
            status_code=400,
            detail="Timestamp is required"
        )
    
    logger.info(f"Getting clip for video {request.video_id} at timestamp {request.timestamp}")
    
    try:
        success, clip_path = video_processor.get_video_clip(request.video_id, request.timestamp)
        
        if not success or clip_path is None:
            raise HTTPException(
                status_code=500,
                detail="Failed to extract clip from video"
            )
        
        if not os.path.exists(clip_path):
            raise HTTPException(
                status_code=404,
                detail=f"Clip file not found at {clip_path}"
            )
        
        logger.info(f"Returning clip from {clip_path}")
        return FileResponse(
            clip_path,
            media_type="video/mp4",
            headers={"Content-Disposition": f"inline; filename={os.path.basename(clip_path)}"}
        )
    except Exception as e:
        logger.error(f"Error in get_clip endpoint: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Error processing video clip: {str(e)}"
        )

@app.post("/get_transcript")
async def get_transcript_chunk(request: VideoRequest):
    if request.timestamp is None:
        raise HTTPException(status_code=400, detail="Timestamp is required")
    
    logger.info(f"Looking for transcript at timestamp: {request.timestamp} seconds")
    transcript = await get_video_transcript(request.video_id)
    if not transcript:
        raise HTTPException(status_code=404, detail="Transcript not found")
    
    chunks = get_transcript_chunks(transcript)
    target_time = request.timestamp
    
    # Find the chunk containing our timestamp
    for start_time, chunk_text in chunks:
        chunk_time = convert_vtt_timestamp_to_seconds(start_time)
        logger.info(f"\nComparing timestamps:")
        logger.info(f"Target time: {target_time} seconds")
        logger.info(f"Chunk time: {chunk_time} seconds")
        logger.info(f"Timestamp string: {start_time}")
        logger.info(f"Text: {chunk_text}")
        
        if abs(chunk_time - target_time) <= 5:  # Within 5 seconds
            logger.info(f"Found matching chunk!")
            logger.info(f"Timestamp: {start_time} ({chunk_time} seconds)")
            logger.info(f"Text: {chunk_text}")
            return {"text": chunk_text}

@app.post("/transcribe_clip")
async def transcribe_clip(request: VideoRequest):
    """Transcribe a 6-second clip from the video at the specified timestamp"""
    if request.timestamp is None:
        raise HTTPException(
            status_code=400,
            detail="Timestamp is required"
        )
    
    logger.info(f"Transcribing clip for video {request.video_id} at timestamp {request.timestamp}")
    
    try:
        # Use existing video processor to get the clip
        success, clip_path = video_processor.get_video_clip(request.video_id, request.timestamp)
        
        if not success or clip_path is None:
            raise HTTPException(
                status_code=500,
                detail="Failed to extract clip from video"
            )
        
        # Transcribe the clip
        text = transcribe_video_file(clip_path)
        
        if text is None:
            raise HTTPException(
                status_code=500,
                detail="Failed to transcribe video clip"
            )
        
        return {
            "success": True,
            "text": text
        }
            
    except Exception as e:
        logger.error(f"Error in transcribe_clip endpoint: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Error transcribing video clip: {str(e)}"
        )

@app.post("/transcribe_buffer")
async def transcribe_buffer(file: UploadFile):
    """Transcribe a video buffer directly"""
    temp_video = None
    try:
        # Create temp file with a unique name
        temp_video = tempfile.NamedTemporaryFile(
            suffix='.mp4',
            delete=False,
            prefix='whisper_video_'
        )
        temp_video_path = temp_video.name
        
        # Close the file handle before writing
        temp_video.close()
        
        # Write uploaded file to temp location
        content = await file.read()
        with open(temp_video_path, 'wb') as f:
            f.write(content)
        
        # Transcribe the video
        text = transcribe_video_file(temp_video_path)
        
        if text is None:
            raise HTTPException(
                status_code=500,
                detail="Failed to transcribe video"
            )
        
        return {
            "success": True,
            "text": text
        }
            
    except Exception as e:
        logger.error(f"Error transcribing buffer: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to transcribe video buffer: {str(e)}"
        )
    finally:
        # Clean up temp video file
        try:
            if temp_video is not None and os.path.exists(temp_video.name):
                os.unlink(temp_video.name)
        except Exception as e:
            logger.warning(f"Failed to clean up temp video file: {e}")

# Add this after your existing imports and before the endpoints
async def transform_text_with_llm(text: str) -> str:
    """Transform transcribed text into casual, twitter-style speech using Llama"""
    prompt = f"""Transform this speech into casual twitter slang:
    internet
Original: "basically you is the user kind of, could give it direction of what path you wanted to go down. Right? And so,"
Internet: yo basically u kinda tell it where to go fr

Original: "{text}"
Internet:"""

    try:
        response = llm(
            prompt,
            max_tokens=30,        # Reduced since we want short tweets
            temperature=0.7,
            stop=["\n", "\r"],
            echo=False
        )
        
        transformed_text = response['choices'][0]['text'].strip()
        
        # Additional cleanup to ensure style consistency
        transformed_text = transformed_text.lower()
        transformed_text = transformed_text.replace('"', '')
        transformed_text = transformed_text.strip('.')
        
        logger.info(f"Original text: {text}")
        logger.info(f"Transformed text: {transformed_text}")
        
        return transformed_text
    except Exception as e:
        logger.error(f"Error transforming text with LLM: {e}")
        return text  # Return original text if transformation fails

# Add this new endpoint
@app.post("/transform_text")
async def transform_text(request: TransformTextRequest):
    """Transform transcribed text into casual twitter style"""
    try:
        transformed = await transform_text_with_llm(request.text)
        return {
            "success": True,
            "original": request.text,
            "transformed": transformed
        }
    except Exception as e:
        logger.error(f"Error in transform_text endpoint: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Error transforming text: {str(e)}"
        )
    
@app.post("/generate_tweet")
async def generate_tweet(request: TweetRequest):
    """Generate a tweet that mostly preserves the already-good transformed text"""
    try:
        # More robust name and common word filtering
        skip_words = {
            'joe', 'rogan', 'marc', 'andreessen', 'says', 'said', 'and', 'the', 'with', 'on',
            'explains', 'discusses', 'talks', 'about', 'podcast', 'interview', 'chief',
            'implications', 'regarding', 'analysis', 'perspective'
        }
        
        # Split on spaces and special characters
        title_parts = re.split(r'[\s\'"]+', request.video_title.lower())
        
        # More thorough name filtering - remove any part that contains our skip words
        topic_words = []
        for part in title_parts:
            if len(part) > 2 and not any(skip in part.lower() for skip in skip_words):
                topic_words.append(part.strip('.,!?#@'))
        
        topic_context = ' '.join(topic_words[:3])
        
        logger.info(f"Using transformed text: {request.transformed_text}")
        logger.info(f"With topic words: {topic_context}")
        
        prompt = f"""Your input is already a great casual tweet. Just enhance it slightly with [{topic_context}] if it adds value - but if it doesn't fit naturally, keep the original. Use that twitter talk in the first person bro.

Original casual tweet: {request.transformed_text}

Slightly enhanced tweet (keep at least 90% the same):"""

        response = llm(
            prompt,
            max_tokens=30,  # Allow more room since we want to preserve length
            temperature=0.5,  # Lower temperature for more faithful preservation
            stop=["\n", "\r", "#"],
            echo=False
        )
        
        tweet_text = response['choices'][0]['text'].strip()
        
        # Clean up but preserve existing slang/style
        tweet_text = tweet_text.lower()
        tweet_text = tweet_text.replace('"', '')
        tweet_text = tweet_text.replace('#', '')
        tweet_text = tweet_text.strip('.,!? ')
        
        # Only add an ending if the text is very short or has no expression
        if len(tweet_text) < 10 or not any(char in tweet_text for char in ['🔥', '💯', '🚀', '💪', '🤔', '😤', '💀', '😳', 'fr', 'wtf', 'ngl']):
            tweet_text += ' fr fr 🔥'
        
        # If the result is too short, return the original transformed text with an emoji
        if len(tweet_text) < 20:
            tweet_text = request.transformed_text + ' 🔥'
        
        logger.info(f"Generated tweet: {tweet_text}")
            
        return {
            "success": True,
            "tweet": tweet_text
        }
        
    except Exception as e:
        logger.error(f"Error generating tweet: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Error generating tweet: {str(e)}"
        )
    
@app.post("/video_title")
async def get_video_title_endpoint(request: VideoTitleRequest):
    """Get the title of a YouTube video"""
    try:
        title = await get_video_title(request.video_id)
        if not title:
            raise HTTPException(
                status_code=404,
                detail="Could not find video title"
            )
            
        return {
            "success": True,
            "title": title
        }
        
    except Exception as e:
        logger.error(f"Error fetching video title: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Error fetching video title: {str(e)}"
        )
    
@app.post("/generate_reply")
async def generate_reply(request: ReplyRequest):
    """Generate a snarky reply and find a matching Marc clip"""
    try:
        # Step 1: Generate snarky response (this part stays the same)
        prompt = f"""Generate a short, snarky response to this tweet, speaking as an AI that makes music beats. 
        Use casual twitter slang, emojis, and sound effect words. Keep it playful and fun, not mean.
        
Tweet: "{request.tweet_text}"

Response (short, fun, casual):"""

        response = llm(
            prompt,
            max_tokens=30,
            temperature=0.8,
            stop=["\n", "\r"],
            echo=False
        )
        
        response_text = response['choices'][0]['text'].strip()
        response_text = response_text.lower().replace('"', '').strip('.,!? ')
        
        if not any(char in response_text for char in ['🎵', '🔥', '🎶', '💯', '🚀', '✨']):
            response_text += ' 🎵🔥'

        logger.info(f"Generated response: {response_text}")

        # For testing mode, return hardcoded values
        if request.test_mode:
            return {
                "response_text": response_text,
                "video_id": "a16z-PTsCCY",
                "timestamp": 180,
                "similarity_score": 0.8
            }

        # Step 2: Extract topic words
        topic_prompt = f"""Extract 3-5 key topic words from this tweet, focusing on actionable or descriptive words.
        Ignore usernames, common words, and @mentions. Return only the words, separated by spaces.

Tweet: "{request.tweet_text}"

Key topic words:"""

        topic_response = llm(
            topic_prompt,
            max_tokens=20,
            temperature=0.2,
            stop=["\n", "\r"],
            echo=False
        )
        
        topic_words = topic_response['choices'][0]['text'].strip().lower().split()
        logger.info(f"Extracted topic words: {topic_words}")

        # Step 3: Search through videos until we find a good match
        best_match = {
            "video_id": None,
            "timestamp": None,
            "score": 0
        }

        max_attempts = 10
        attempts = 0

        while attempts < max_attempts and best_match["score"] < 0.7:
            attempts += 1
            
            try:
                # Get a random video
                video_id = (await get_random_video())["video_id"]
                logger.info(f"\nChecking video {video_id} (attempt {attempts}/{max_attempts})")
                
                # First find timestamps where Marc is speaking
                marc_timestamp = await find_marc_timestamp(video_id)
                if not marc_timestamp:
                    logger.info(f"No clear Marc speech found in video {video_id}")
                    continue

                # Verify the clip with transcription
                success, clip_path = video_processor.get_video_clip(video_id, marc_timestamp)
                if not success:
                    continue

                verification_text = transcribe_video_file(clip_path)
                if not verification_text:
                    continue

                is_marc = await verify_marc_speaking(verification_text)
                if not is_marc:
                    logger.info(f"Speech pattern validation failed - confidence too low")
                    continue

                logger.info(f"Found Marc speaking at {marc_timestamp}")
                
                # Get transcript around this timestamp
                transcript = await get_video_transcript(video_id)
                if not transcript:
                    continue

                chunks = get_transcript_chunks(transcript)
                
                # Look for topic matches in chunks near Marc's speech
                for timestamp, chunk_text in chunks:
                    chunk_time = convert_vtt_timestamp_to_seconds(timestamp)
                    
                    # Only check chunks within 30 seconds of where Marc starts speaking
                    if abs(chunk_time - marc_timestamp) > 30:
                        continue

                    # Count matching topic words
                    chunk_lower = chunk_text.lower()
                    matching_words = sum(1 for word in topic_words if word in chunk_lower)
                    
                    if matching_words >= 1:  # Even one match might be interesting
                        base_score = matching_words / len(topic_words)
                        
                        logger.info(f"\nAnalyzing potential match:")
                        logger.info(f"Timestamp: {timestamp}")
                        logger.info(f"Text: {chunk_text}")
                        logger.info(f"Matching words: {matching_words}")
                        
                        relevance_prompt = f"""Rate how relevant this clip is to these topics on a scale of 0-10.
                        Consider context and meaning, not just word matching.

Topics: {', '.join(topic_words)}

Clip: {chunk_text}

Rating (just the number 0-10):"""

                        relevance_response = llm(
                            relevance_prompt,
                            max_tokens=2,
                            temperature=0.2,
                            stop=["\n", "\r", " "],
                            echo=False
                        )

                        try:
                            relevance_score = float(relevance_response['choices'][0]['text'].strip()) / 10
                            final_score = (base_score + relevance_score) / 2
                            
                            logger.info(f"Score: {final_score}")
                            
                            if final_score > best_match["score"]:
                                best_match = {
                                    "video_id": video_id,
                                    "timestamp": chunk_time,
                                    "score": final_score
                                }
                                
                                if final_score > 0.7:
                                    break

                        except ValueError:
                            continue

            except Exception as e:
                logger.error(f"Error processing video {video_id}: {e}")
                continue

        if best_match["video_id"] is None:
            raise HTTPException(
                status_code=404,
                detail="Could not find a matching clip"
            )

        return {
            "response_text": response_text,
            "video_id": best_match["video_id"],
            "timestamp": best_match["timestamp"],
            "similarity_score": best_match["score"]
        }

    except Exception as e:
        logger.error(f"Error generating reply: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Error generating reply: {str(e)}"
        )

if __name__ == "__main__":
    logger.info("Starting FastAPI server...")
    uvicorn.run(app, host="0.0.0.0", port=3000)