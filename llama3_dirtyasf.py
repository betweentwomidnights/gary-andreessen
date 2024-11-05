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
import cv2
import torch
import psutil

print("CUDA available:", torch.cuda.is_available())
print("CUDA device count:", torch.cuda.device_count())
if torch.cuda.is_available():
    print("CUDA device name:", torch.cuda.get_device_name(0))

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

def check_gpu_usage():
    """Check if GPU is being used by Llama"""
    if torch.cuda.is_available():
        gpu_memory = torch.cuda.memory_allocated() / 1024**2  # Convert to MB
        return f"GPU Memory Used: {gpu_memory:.2f} MB"
    return "GPU not available"

def log_system_resources():
    """Log current system resource usage"""
    cpu_percent = psutil.cpu_percent()
    memory = psutil.virtual_memory()
    return f"CPU Usage: {cpu_percent}%, RAM Usage: {memory.percent}%"

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
logger.info("Initializing LLaMa 3 model...")

def create_system_prompt(task_description: str) -> str:
    """Create a formatted system prompt for Llama 3"""
    return f"""<|im_start|>system
You are a helpful AI assistant specialized in {task_description}. 
You aim to provide clear, accurate, and relevant responses.
Respond in a direct and natural way without mentioning that you are an AI.
<|im_end|>"""

def format_prompt(system_prompt: str, user_input: str) -> str:
    """Format a prompt for Llama 3 with proper tokens"""
    return f"""{system_prompt}
<|im_start|>user
{user_input}<|im_end|>
<|im_start|>assistant
"""

# Initialize the model with updated parameters
try:
    logger.info("Initializing Llama model with GPU support...")
    llm = Llama(
        model_path="models/Llama-3.1-8B-Lexi-Uncensored_V2_F16.gguf",
        n_gpu_layers=35,
        n_ctx=128000,
        n_threads=4,
        verbose=False  # Enable verbose mode temporarily
    )
    logger.info(f"Llama initialization complete. {check_gpu_usage()}")
    logger.info(f"System resources: {log_system_resources()}")
    
    # Test GPU usage with a small inference
    test_result = llm("Test prompt", max_tokens=1)
    logger.info(f"Test inference complete. {check_gpu_usage()}")
except Exception as e:
    logger.error(f"Error initializing Llama: {e}")
    raise

# Example helper function for making calls to the model
async def llm_generate(
    prompt: str,
    system_context: str,
    max_tokens: int = 100,
    temperature: float = 0.7,
    stop: List[str] = None
) -> str:
    """
    Generate text using LLaMa 3 with proper formatting
    
    Args:
        prompt: The user's input prompt
        system_context: Description of the current task for system prompt
        max_tokens: Maximum tokens to generate
        temperature: Sampling temperature
        stop: List of stop sequences
        
    Returns:
        Generated text response
    """
    try:
        system_prompt = create_system_prompt(system_context)
        formatted_prompt = format_prompt(system_prompt, prompt)
        
        response = llm(
            formatted_prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            stop=stop or ["<|im_end|>", "<|im_start|>", "\n\n"],
            echo=False
        )
        
        # Extract just the assistant's response
        return response['choices'][0]['text'].strip()
        
    except Exception as e:
        logger.error(f"Error in llm_generate: {e}")
        raise


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

def get_chunks(transcript: str, chunk_size: int = 10) -> List[Tuple[str, str]]:
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

def get_vtt_chunks(transcript: str, chunk_size: int = 10) -> List[Tuple[str, str]]:
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

def convert_vtt_timestamp_to_seconds(timestamp: str) -> float:  # Changed return type to float
    """Convert timestamp (HH:MM:SS.mmm) to seconds with millisecond precision"""
    logger.debug(f"Converting timestamp: '{timestamp}'")
    try:
        # Clean up the timestamp - remove brackets and whitespace
        clean_timestamp = timestamp.strip('[] \n')
        
        # Handle milliseconds
        seconds = 0.0
        if '.' in clean_timestamp:
            time_part, ms_part = clean_timestamp.split('.')
            seconds = float(f"0.{ms_part}")
            clean_timestamp = time_part
        
        # Split hours, minutes, seconds
        parts = clean_timestamp.split(':')
        if len(parts) == 3:  # HH:MM:SS
            hours, minutes, secs = map(float, parts)
            total_seconds = (
                hours * 3600 +     # hours to seconds
                minutes * 60 +     # minutes to seconds
                secs +            # seconds
                seconds          # add milliseconds part
            )
        elif len(parts) == 2:  # MM:SS
            minutes, secs = map(float, parts)
            total_seconds = minutes * 60 + secs + seconds
        else:
            logger.warning(f"Unexpected timestamp format: {timestamp}")
            return 0.0
        
        logger.debug(f"Converted {timestamp} to {total_seconds} seconds")
        return round(total_seconds, 3)  # Round to milliseconds
            
    except Exception as e:
        logger.warning(f"Could not parse timestamp {timestamp}: {str(e)}")
        return 0.0



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
    
def clean_transcript_text(text: str) -> str:
    """Clean HTML tags and deduplicate repeated phrases"""
    try:
        if not text:
            return ""
            
        # Remove HTML tags
        text = re.sub(r'<[^>]+>', ' ', text)
        # Remove timestamp markers
        text = re.sub(r'\[\d+:\d+:\d+\.\d+\]', ' ', text)
        # Clean up multiple spaces
        text = re.sub(r'\s+', ' ', text)
        
        # Simple deduplication of repeated phrases
        words = text.split()
        if not words:
            return ""
            
        # Build final text without immediate repetitions
        final_words = []
        prev_phrase = ""
        for i in range(len(words)):
            current_phrase = words[i]
            # Don't add if it's an immediate repetition
            if current_phrase != prev_phrase:
                final_words.append(current_phrase)
                prev_phrase = current_phrase
        
        return ' '.join(final_words).strip()
        
    except Exception as e:
        logger.error(f"Error in clean_transcript_text: {e}")
        logger.error(f"Input text was: {text}")
        return text  # Return original text if cleaning fails
    
def get_transcript_chunks(transcript: str) -> List[Tuple[str, str]]:
    """Split transcript into meaningful conversation chunks with proper timestamp handling"""
    logger.info("Starting transcript chunking...")
    segments = transcript.split('\n')
    chunks = []
    conversation_buffer = []
    current_time = None
    
    CONTEXT_WINDOW = 1  # Number of segments to combine
    
    for segment in segments:
        # Log every few segments for debugging
        if len(chunks) % 10 == 0:
            logger.debug(f"Processing segment: {segment}")
            
        timestamp_match = re.match(r'\[([\d:\.]+)\]', segment)
        if timestamp_match:
            # Get and convert timestamp
            timestamp_str = timestamp_match.group(1)
            timestamp_seconds = convert_vtt_timestamp_to_seconds(timestamp_str)
            
            # Only process if timestamp is valid (non-zero)
            if timestamp_seconds > 0:
                # Clean up the text before adding to buffer
                cleaned_text = clean_transcript_text(
                    re.sub(r'\[[\d:\.]+\]\s*', '', segment)
                )
                
                if not current_time:
                    current_time = timestamp_str
                    logger.debug(f"Setting initial timestamp: {current_time}")
                
                if cleaned_text and len(cleaned_text.split()) >= 3:  # Ensure minimum meaningful content
                    conversation_buffer.append(cleaned_text)
                    logger.debug(f"Added to buffer: {cleaned_text}")
                
                if len(conversation_buffer) >= CONTEXT_WINDOW:
                    full_text = ' '.join(conversation_buffer)
                    if len(full_text.split()) >= 10:  # Minimum words for analysis
                        chunks.append((current_time, full_text))
                        logger.debug(f"Created chunk at {current_time}: {full_text[:100]}...")
                    conversation_buffer = conversation_buffer[1:]  # Sliding window
                    current_time = timestamp_str  # Update timestamp for next chunk
    
    # Add remaining conversation if exists
    if conversation_buffer and current_time:
        full_text = ' '.join(conversation_buffer)
        if len(full_text.split()) >= 10:
            chunks.append((current_time, full_text))
            logger.debug(f"Added final chunk at {current_time}")
    
    logger.info(f"Created {len(chunks)} chunks")
    if chunks:
        logger.info("First few chunks with timestamps:")
        for i, (time, text) in enumerate(chunks[:3]):
            seconds = convert_vtt_timestamp_to_seconds(time)
            logger.info(f"Chunk {i}: Time={time} ({seconds} seconds)")
            logger.info(f"Text: {text[:100]}...")
    
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
    """Find a timestamp where Marc begins speaking"""
    try:
        transcript = await get_video_transcript(video_id)
        if not transcript:
            logger.info(f"No transcript found for video {video_id}")
            return None
        
        chunks = get_transcript_chunks(transcript)
        if not chunks:
            logger.info(f"No valid chunks found for video {video_id}")
            return None
            
        random.shuffle(chunks)
        
        system_context = """You analyze podcast transcripts to identify when Marc Andreessen is speaking.
        Marc has two distinct speaking styles:
        
        Formal/Technical:
        - Often starts with "So", "Well", "Yeah", "Right"
        - Uses technical or economic terms
        - Gives detailed, analytical answers
        - Builds points with "And so", "And then"
        
        Casual/Conversational:
        - Uses colloquialisms like "grinding", "stuff", "things"
        - Can be more repetitive when emphasizing points
        - Sometimes uses casual filler words like "um", "you know"
        - Often gets excited about technical topics
        - Frequently relates topics to software or technology"""
        
        for i, (start_time, chunk) in enumerate(chunks):
            try:
                # Validate chunk before processing
                if not chunk or len(chunk.split()) < 10:
                    logger.debug(f"Skipping short chunk {i}: {chunk}")
                    continue
                
                logger.info(f"\nProcessing chunk {i} at {start_time}:")
                logger.info(f"Chunk text: {chunk}")

                analysis_prompt = f"""Analyze this podcast segment, considering both Marc's formal and casual speaking styles.

Transcript: "{chunk}"

Provide your analysis in exactly this format:
Is Marc Speaking: [yes/no]
Confidence Rating: [number 1-10]
Reasoning: [your explanation]"""

                response = await llm_generate(
                    prompt=analysis_prompt,
                    system_context=system_context,
                    max_tokens=150,
                    temperature=0.1,
                    stop=["Transcript:", "<|im_end|>"]
                )
                
                # Validate response
                if not response:
                    logger.warning(f"Empty response for chunk {i}")
                    continue
                    
                # Clean up response
                cleaned_response = response.replace('<|im_start|>', '').replace('<|im_end|>', '').strip()
                logger.info(f"Raw response: {cleaned_response}")
                
                # Parse the structured response with fallbacks
                try:
                    # Look for Marc speaking indication
                    is_marc_match = re.search(r'Is Marc Speaking:\s*(yes|no)', cleaned_response, re.IGNORECASE)
                    is_marc_speaking = bool(is_marc_match and 'yes' in is_marc_match.group(1).lower())
                    
                    # Look for confidence rating
                    confidence_match = re.search(r'Confidence Rating:\s*(\d+)', cleaned_response)
                    confidence = int(confidence_match.group(1)) if confidence_match else 0
                    
                    logger.info(f"Parsed results - Is Marc: {is_marc_speaking}, Confidence: {confidence}/10")
                    
                    # Accept if we're confident it's Marc
                    if is_marc_speaking and confidence >= 7:
                        chunk_start = convert_vtt_timestamp_to_seconds(start_time)
                        logger.info(f"Found high-confidence Marc segment at {start_time}!")
                        return chunk_start
                        
                except (AttributeError, ValueError) as e:
                    logger.error(f"Error parsing response for chunk {i}: {e}")
                    logger.error(f"Response was: {cleaned_response}")
                    continue
                        
            except Exception as e:
                logger.error(f"Error processing chunk {i}: {e}")
                logger.error(f"Chunk was: {chunk if 'chunk' in locals() else 'No chunk'}")
                logger.error(f"Response was: {response if 'response' in locals() else 'No response'}")
                continue
        
        logger.info(f"No suitable Marc segments found in video {video_id} after checking {len(chunks)} chunks")
        return None
        
    except Exception as e:
        logger.error(f"Error processing video {video_id}: {e}")
        logger.error(f"Stack trace:", exc_info=True)
        return None

async def verify_marc_speaking(text: str) -> bool:
    """Verify if text matches Marc's speech patterns"""
    try:
        system_context = """You are an expert at identifying Marc Andreessen's speech patterns 
        and mannerisms. You know his common phrases, speaking style, and conversation patterns 
        in great detail."""
        
        verify_prompt = f"""Analyze this text and rate how confident you are that it's Marc Andreessen speaking.
Consider these aspects:
- His common phrases and word choices
- His speaking rhythm and style
- His tendency to start sentences with "So", "Well", "Right", "Yeah"
- His way of explaining complex ideas

Text: "{text}"

Rate your confidence from 0-9 (ONLY respond with a single digit, nothing else):"""

        response = await llm_generate(
            prompt=verify_prompt,
            system_context=system_context,
            max_tokens=1,
            temperature=0.1,
            stop=["\n", " ", ".", ","]  # Strict stops to ensure single digit
        )
        
        # Log the response for debugging
        logger.info(f"Speech verification response: '{response}'")
        
        # Verify we got a valid digit
        if not response or not response.isdigit() or len(response) != 1:
            logger.warning(f"Invalid verification response: '{response}'")
            return False
            
        # Convert to integer and check confidence
        confidence = int(response)
        logger.info(f"Marc speech confidence score: {confidence}/9")
        
        # Higher threshold for Llama 3 since it might be more generous
        return confidence >= 7

    except Exception as e:
        logger.error(f"Error in speech verification: {e}")
        return False

# Alternative approach using find_marc_timestamp
async def verify_marc_speaking_alternative(video_id: str, timestamp: int) -> bool:
    """Verify if Marc is speaking using the timestamp finding logic"""
    try:
        # Get transcript around this timestamp
        transcript = await get_video_transcript(video_id)
        if not transcript:
            return False
            
        chunks = get_transcript_chunks(transcript)
        
        # Look for Marc speaking within 5 seconds of our timestamp
        for chunk_timestamp, chunk_text in chunks:
            chunk_time = convert_vtt_timestamp_to_seconds(chunk_timestamp)
            if abs(chunk_time - timestamp) <= 5:
                # Use the more comprehensive analysis from find_marc_timestamp
                system_context = """You are specialized in analyzing speech patterns and speaker identification, 
                particularly for Marc Andreessen. You can recognize his speech patterns, common phrases, 
                and speaking style with high accuracy."""
                
                analysis_prompt = f"""Analyze this text segment for Marc Andreessen's speech patterns.
Consider:
1. Does it begin with Marc speaking (not just containing his speech)?
2. Does it match his common speech patterns (e.g., starting with "So", "Well", "Yeah", "Right", "Exactly")?
3. Is this definitely the start of his statement (not mid-sentence)?

Text segment:
{chunk_text}

Rate each aspect:
1. Starts with Marc (yes/no):
2. Matches patterns (yes/no):
3. Statement start (yes/no):"""

                response = await llm_generate(
                    prompt=analysis_prompt,
                    system_context=system_context,
                    max_tokens=50,
                    temperature=0.1,
                    stop=["Text segment:", "\n\n"]
                )
                
                # Parse the response for all three criteria
                starts_with_marc = 'yes' in response.lower().split('\n')[0]
                matches_patterns = 'yes' in response.lower().split('\n')[1]
                statement_start = 'yes' in response.lower().split('\n')[2]
                
                return starts_with_marc and (matches_patterns or statement_start)
                
        return False

    except Exception as e:
        logger.error(f"Error in alternative speech verification: {e}")
        return False
    
async def generate_combined_reply(snarky_response: str, transformed_text: str) -> str:
    """Combine snarky response with transformed text themes"""
    system_context = """You are a Gen-Z social media expert who excels at combining different 
    ideas into witty, sarcastic responses. You use internet slang and emojis naturally."""
    
    combine_prompt = f"""yo we got these two pieces:

1. our initial reaction: {snarky_response}
2. i said (transformed): {transformed_text}

Give more weight to the initial reaction, but blend these into ONE short tweet that captures both the sarcastic vibe AND references our point a little bit.
keep it super casual and fun! use some of the same emojis/slang vibes.

just give the final combined tweet (30 tokens max):"""

    try:
        combined_response = await llm_generate(
            prompt=combine_prompt,
            system_context=system_context,
            max_tokens=30,
            temperature=0.9
        )
        
        # Clean up and ensure we have emojis
        response_text = combined_response.lower().strip('"`.,!? \n')
        internet_vibes = ['💀', '🤣', '😭', '💯', '🚀', '💪', '🤔', '😤', '😳', '🤯', '⚡️']
        
        if not any(vibe in response_text for vibe in internet_vibes):
            response_text += f' {random.choice(internet_vibes)}'
            
        logger.info(f"Combined response: {response_text}")
        return response_text
        
    except Exception as e:
        logger.error(f"Error combining replies: {e}")
        return snarky_response  # Fallback to original response

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
    logger.info(f"Starting analyze request for video: {request.video_id}")
    logger.info(f"Initial resource check: {log_system_resources()}")
    
    videos = await get_playlist_videos()
    tried_videos = set()
    
    # Try the requested video first
    tried_videos.add(request.video_id)
    
    try:
        logger.info("Calling find_marc_timestamp...")
        timestamp = await find_marc_timestamp(request.video_id)
        logger.info(f"find_marc_timestamp returned: {timestamp}")
        
        if timestamp is not None:
            logger.info("Timestamp found, starting verification...")
            
            # Basic validation first
            if not isinstance(timestamp, (int, float)) or timestamp < 0:
                logger.error(f"Invalid timestamp value: {timestamp}")
                raise ValueError("Invalid timestamp")
            
            try:
                # Get video clip
                logger.info("Getting video clip...")
                success, clip_path = video_processor.get_video_clip(request.video_id, timestamp)
                
                if not success or not clip_path or not os.path.exists(clip_path):
                    logger.error(f"Failed to get video clip. Success: {success}, Path exists: {os.path.exists(clip_path) if clip_path else False}")
                    raise Exception("Failed to get video clip")
                
                logger.info(f"Clip generated successfully at: {clip_path}")
                
                # Transcribe the clip
                logger.info("Transcribing clip...")
                verification_text = transcribe_video_file(clip_path)
                
                if verification_text:
                    logger.info(f"Transcription successful: {verification_text}")
                    
                    # Return successful result
                    result = {
                        "timestamp": timestamp,
                        "video_id": request.video_id
                    }
                    logger.info(f"Returning successful result: {result}")
                    return result
                else:
                    logger.error("Transcription failed or returned empty")
                    timestamp = None
                    
            except Exception as e:
                logger.error(f"Error during clip verification: {str(e)}", exc_info=True)
                timestamp = None
    
    except Exception as e:
        logger.error(f"Error in analyze endpoint: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Error analyzing video: {str(e)}"
        )
    
    logger.error("Could not find or verify any timestamp")
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
        
        if not success:
            logger.error("Frame extraction failed")
            raise HTTPException(
                status_code=500,
                detail="Failed to extract frame from video"
            )
        
        if frame_path is None or not os.path.exists(frame_path):
            logger.error(f"Frame file not found at expected path: {frame_path}")
            raise HTTPException(
                status_code=404,
                detail="Frame file not found"
            )
        
        # Verify the frame file is a valid image
        try:
            img = cv2.imread(frame_path)
            if img is None:
                logger.error("Frame file exists but cannot be read as image")
                raise HTTPException(
                    status_code=500,
                    detail="Invalid frame image file"
                )
        except Exception as e:
            logger.error(f"Error verifying frame file: {e}")
            raise HTTPException(
                status_code=500,
                detail="Error verifying frame file"
            )
        
        logger.info(f"Successfully extracted and verified frame at {frame_path}")
        return FileResponse(
            frame_path,
            media_type="image/jpeg",
            headers={
                "Content-Disposition": f"inline; filename={os.path.basename(frame_path)}",
                "Cache-Control": "no-cache"  # Prevent caching issues
            }
        )
    except HTTPException:
        raise
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
    """Generate a tweet that's casual and internet-speak styled"""
    try:
        # Filter out boring words from title
        skip_words = {
            'joe', 'rogan', 'marc', 'andreessen', 'says', 'said', 'and', 'the', 'with', 'on',
            'explains', 'discusses', 'talks', 'about', 'podcast', 'interview', 'chief',
            'implications', 'regarding', 'analysis', 'perspective'
        }
        
        # Get interesting words from title
        title_parts = re.split(r'[\s\'"]+', request.video_title.lower())
        topic_words = [
            part.strip('.,!?#@') for part in title_parts 
            if len(part) > 2 and not any(skip in part.lower() for skip in skip_words)
        ]
        topic_context = ' '.join(topic_words[:3])
        
        logger.info(f"Using transformed text: {request.transformed_text}")
        logger.info(f"With topic words: {topic_context}")

        system_context = """You create ultra-casual Generation Z style tweets. You use internet slang, 
        emojis, and very informal language. You make things sound silly and fun, like chatting with 
        friends on Discord. You never use formal language or complex terms."""
        
        prompt = f"""yo make this tweet super internet vibes! use these topic words if they fit: {topic_context}
never use brackets or parentheses fr fr

peep these examples of the vibe we want:
"machine learning is computationally intensive" -> "this ai stuff be eatin up all the cpu power fr fr 💀"
"economic policy affects market stability" -> "economy actin real sus these days no cap 😭"
"quantum entanglement demonstrates non-locality" -> "quantum particles be texting each other from across the universe frfr 🤯"
"blockchain technology enables decentralization" -> "crypto do be hitting different with that decentralized life tho 🔥"

take this mid tweet and make it way more fun:
{request.transformed_text}

give me just the transformed tweet and nothing else:"""

        response = await llm_generate(
            prompt=prompt,
            system_context=system_context,
            max_tokens=30,
            temperature=0.8,  # Higher temperature for more creative slang
            stop=["\n", "original:", "take this", "<|im_end|>"]  # Added token stop
        )
        
        # Clean up the response
        tweet_text = response.strip().replace('<|im_start|>', '').replace('<|im_end|>', '')
        
        # Basic cleanup
        tweet_text = tweet_text.lower()
        tweet_text = tweet_text.replace('"', '')
        tweet_text = tweet_text.strip('.,!? ')
        
        # Make sure we have some internet energy
        internet_vibes = [
            '💀', '🔥', '😭', '💯', '🚀', '💪', '🤔', '😤', '😳', '🤯', '⚡️',
            'fr', 'ngl', 'fr fr', 'frfr', 'ong', 'no cap', 'deadass', 'bussin',
            'lowkey', 'highkey', 'finna', 'sus', 'vibes'
        ]
        
        if not any(vibe in tweet_text for vibe in internet_vibes):
            # Add random internet energy
            vibes = [' fr fr 💀', ' ngl 🔥', ' ong 😤', ' no cap 💯', ' frfr 🤯']
            tweet_text += random.choice(vibes)
        
        # Ensure we're not duplicating common endings
        common_endings = ['fr fr', 'frfr', 'ong', 'no cap']
        for ending in common_endings:
            if tweet_text.count(ending) > 1:
                tweet_text = tweet_text.rsplit(ending, 1)[0] + ending
        
        # If the result is too short or seems incomplete
        if len(tweet_text) < 20:
            tweet_text = request.transformed_text + random.choice(vibes)
        
        logger.info(f"Original: {request.transformed_text}")
        logger.info(f"Generated tweet: {tweet_text}")
            
        return {
            "success": True,
            "tweet": tweet_text
        }
        
    except Exception as e:
        logger.error(f"Error generating tweet: {e}")
        logger.error(f"Response was: {response if 'response' in locals() else 'No response'}")
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
    """Generate a snarky reply and prepare video assets for processing"""
    try:
        # Step 1: Generate snarky response
        system_context = """You create ultra-casual Generation Z style responses. 
        You're snarky, sarcastic, and use lots of internet slang and emojis."""
        
        response_prompt = f"""yo give me a snarky internet response to this! be creative and use emojis!
make it sound like a zoomer on twitter fr fr

tweet: "{request.tweet_text}"

response:"""

        response = await llm_generate(
            prompt=response_prompt,
            system_context=system_context,
            max_tokens=30,
            temperature=0.9
        )
        
        response_text = response.lower().replace('"', '').strip('.,!? ')
        if not any(char in response_text for char in ['💀', '🔥', '😭', '💯', '🚀', '💪', '🤔', '😤', '😳', '🤯', '⚡️']):
            response_text += f' {random.choice(["💀", "🔥", "😭", "💯"])}'

        logger.info(f"Generated response: {response_text}")

        if request.test_mode:
            return {
                "response_text": response_text,
                "video_id": "a16z-PTsCCY",
                "timestamp": 180,
                "similarity_score": 0.8,
                "clip_url": "/api/clip/test",
                "frame_url": "/api/frame/test"
            }

         # Step 2: Find relevant Marc clip
        max_attempts = 5
        attempts = 0
        best_match = None

        while attempts < max_attempts and not best_match:
            attempts += 1
            try:
                video_id = (await get_random_video())["video_id"]
                logger.info(f"\nChecking video {video_id} (attempt {attempts}/{max_attempts})")
                
                marc_timestamp = await find_marc_timestamp(video_id)
                if marc_timestamp is None or marc_timestamp < 0:
                    continue

                # Ensure timestamp is valid
                marc_timestamp = max(0, float(marc_timestamp))
                logger.info(f"Found Marc speaking at timestamp: {marc_timestamp}")

                # Get clip immediately to verify it works
                success, clip_path = video_processor.get_video_clip(
                    video_id=video_id,
                    timestamp=int(marc_timestamp),  # VideoProcessor expects int
                    duration=6
                )

                if not success or not clip_path:
                    logger.error(f"Failed to get clip for timestamp {marc_timestamp}")
                    continue

                # If we can get the clip, proceed with transcript matching
                transcript = await get_video_transcript(video_id)
                if not transcript:
                    continue

                chunks = get_transcript_chunks(transcript)
                
                for timestamp, chunk_text in chunks:
                    chunk_time = convert_vtt_timestamp_to_seconds(timestamp)
                    
                    if abs(chunk_time - marc_timestamp) > 30:
                        continue

                    context_prompt = f"""Consider this exchange:

Comment: "{request.tweet_text}"

Marc's clip: "{chunk_text}"

Could Marc's clip work as a response or relate to the comment? Consider:
1. Thematic connection
2. Topical relevance
3. Natural flow of conversation
4. Similar concepts or ideas

Rate 0-10 how well it works (just respond with the number):"""

                    relevance_response = await llm_generate(
                        prompt=context_prompt,
                        system_context="You evaluate how well speech segments could work as responses in a conversation.",
                        max_tokens=2,
                        temperature=0.3
                    )

                    try:
                        relevance_score = float(relevance_response.strip()) / 10
                        logger.info(f"\nEvaluating clip relevance:")
                        logger.info(f"Original comment: {request.tweet_text}")
                        logger.info(f"Marc's clip: {chunk_text}")
                        logger.info(f"Relevance score: {relevance_score}")
                        
                        if relevance_score >= 0.7:
                            # Get the frame now to verify everything works
                            frame_success, frame_path = video_processor.get_video_frame(
                                video_id=video_id,
                                timestamp=int(marc_timestamp + 5)  # Last frame
                            )

                            if not frame_success:
                                logger.error("Failed to get frame, trying next match")
                                continue

                            best_match = {
                                "video_id": video_id,
                                "timestamp": int(marc_timestamp),  # Ensure integer
                                "score": relevance_score,
                                "transcript": chunk_text,
                                "clip_path": clip_path,
                                "frame_path": frame_path
                            }
                            break

                    except ValueError:
                        continue

                if best_match:
                    break

            except Exception as e:
                logger.error(f"Error processing video {video_id}: {e}")
                continue

        if not best_match:
            raise HTTPException(
                status_code=404,
                detail="Could not find a suitable clip"
            )

        youtube_url = f"https://youtube.com/watch?v={best_match['video_id']}&t={best_match['timestamp']}"
        
        # Step 3: Get transcript and transform text
        transformed_text = None
        try:
            # Use our existing transcribe_video_file function directly
            transcript_text = transcribe_video_file(best_match["clip_path"])
            if transcript_text:
                logger.info(f"Transcribed clip: {transcript_text}")
                
                # Use our transform_text_with_llm function directly
                transformed_text = await transform_text_with_llm(transcript_text)
                logger.info(f"Transformed text: {transformed_text}")
        except Exception as e:
            logger.error(f"Error in transcription/transformation: {e}")
            # Continue without transformed text if it fails

        # Step 4: Generate final combined response if we have transformed text
        final_response = response_text  # Default to original response
        if transformed_text:
            try:
                final_response = await generate_combined_reply(response_text, transformed_text)
                logger.info(f"Combined response: {final_response}")
            except Exception as e:
                logger.error(f"Error combining responses: {e}")
                # Fall back to original response if combination fails

        youtube_url = f"https://youtube.com/watch?v={best_match['video_id']}&t={best_match['timestamp']}"

        return {
            "response_text": final_response,
            "original_response": response_text,
            "video_id": best_match["video_id"],
            "timestamp": best_match["timestamp"],
            "similarity_score": best_match["score"],
            "transcript": best_match["transcript"],
            "transformed_text": transformed_text,
            "clip_path": best_match["clip_path"],
            "frame_path": best_match["frame_path"],
            "youtube_url": youtube_url
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