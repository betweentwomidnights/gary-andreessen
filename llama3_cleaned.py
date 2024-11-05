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
from timestamp_utils import TimestampUtils

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
    from_username: Optional[str] = None
    test_mode: Optional[bool] = False

class ReplyResponse(BaseModel):
    response_text: str
    video_id: str
    timestamp: int
    similarity_score: float
    
class PMarcaReplyRequest(BaseModel):
    tweet_text: str
    quoted_text: Optional[str] = None
    test_mode: bool = False

# Configuration
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

timestamp_utils = TimestampUtils(
    playlist_id=PLAYLIST_ID,
    llm=llm  # Pass the LLM instance
)


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


    
async def generate_combined_reply(snarky_response: str, transformed_text: str, original_tweet: str) -> str:
    """Combine snarky response with transformed text themes while maintaining context"""
    system_context = """You are gary, a witty AI personality who responds naturally to comments.
Keep your responses feeling like genuine reactions to what people say.
You use modern internet slang and emojis naturally.
MOST IMPORTANT: Your response should feel like it's actually replying to their comment!"""
    
    combine_prompt = f"""someone replied to you on twitter and you already came up with a great response!
now you also have a relevant clip to reference. blend these together while keeping the same energy.

their tweet: "{original_tweet}"

your initial reaction: "{snarky_response}"
(this was a great response - keep this vibe and energy!)

clip context: "{transformed_text}"

guidelines for combining:
- keep the same energy as your initial reaction
- it should still feel like you're responding to their actual comment
- don't lose the personality/vibe from the initial reaction
- reference the clip naturally if it fits
- don't force the clip reference if it doesn't flow
- keep it short and snappy
- use emojis that match the vibe

just give the final tweet (max 30 tokens), make it feel natural fr fr:"""

    try:
        combined_response = await llm_generate(
            prompt=combine_prompt,
            system_context=system_context,
            max_tokens=30,
            temperature=0.8
        )
        
        # Clean up response
        response_text = combined_response.lower().strip('"`.,!? \n')
        
        # Smarter emoji handling - try to keep emojis from original response
        original_emojis = re.findall(r'[😀-🟿]', snarky_response)
        if original_emojis:
            # If original response had emojis, make sure we keep at least one
            if not any(emoji in response_text for emoji in original_emojis):
                response_text += f' {random.choice(original_emojis)}'
        else:
            # Fallback to adding a new emoji if needed
            internet_vibes = ['💀', '🤣', '😭', '💯', '🚀', '💪', '🤔', '😤', '😳', '🤯', '⚡️']
            if not any(vibe in response_text for vibe in internet_vibes):
                response_text += f' {random.choice(internet_vibes)}'
            
        logger.info(f"Original tweet: {original_tweet}")
        logger.info(f"Initial response: {snarky_response}")
        logger.info(f"Clip context: {transformed_text}")
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
        videos = await timestamp_utils.get_playlist_videos()
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
    
    videos = await timestamp_utils.get_playlist_videos()
    tried_videos = set()
    
    # Try the requested video first
    tried_videos.add(request.video_id)
    
    try:
        logger.info("Calling find_marc_timestamp...")
        timestamp = await timestamp_utils.find_marc_timestamp(request.video_id)
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
    transcript = await timestamp_utils.get_video_transcript(request.video_id)
    if not transcript:
        raise HTTPException(status_code=404, detail="Transcript not found")
    
    chunks = timestamp_utils.get_transcript_chunks(transcript)
    target_time = request.timestamp
    
    # Find the chunk containing our timestamp
    for start_time, chunk_text in chunks:
        chunk_time = timestamp_utils.convert_vtt_timestamp_to_seconds(start_time)
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
        # Get interesting words from title
        title_parts = re.split(r'[\s\'"]+', request.video_title.lower())
        skip_words = {
            'joe', 'rogan', 'marc', 'andreessen', 'says', 'said', 'and', 'the', 'with', 'on',
            'explains', 'discusses', 'talks', 'about', 'podcast', 'interview', 'chief',
            'implications', 'regarding', 'analysis', 'perspective', 'lex', 'fridman'
        }
        
        topic_words = [
            part.strip('.,!?#@') for part in title_parts 
            if len(part) > 2 and not any(skip in part.lower() for skip in skip_words)
        ]
        topic_context = ' '.join(topic_words[:3])
        
        logger.info(f"Using transformed text: {request.transformed_text}")
        logger.info(f"With topic words: {topic_context}")

        # Generate initial tweet
        prompt = f"""yo make this tweet super internet vibes! use these topic words if they fit: {topic_context}
never use brackets or parentheses fr fr

peep these examples of the vibe we want:
"machine learning is computationally intensive" -> "this ai stuff be eatin up all the cpu power fr fr 💀"
"economic policy affects market stability" -> "economy actin real sus these days no cap 😭"
"quantum entanglement demonstrates non-locality" -> "quantum particles be texting each other from across the universe frfr 🤯"

take this mid tweet and make it way more fun (KEEP IT SHORT):
{request.transformed_text}

give me just the transformed tweet and nothing else:"""

        response = await llm_generate(
            prompt=prompt,
            system_context="You create ultra-casual Generation Z style tweets with internet slang and emojis, but you are marc andreessen.",
            max_tokens=20,
            temperature=0.8,
            stop=["\n", "original:", "take this", "<|im_end|>"]
        )
        
        # Clean up the response - FIXED: Removed errant comma
        tweet_text = response.strip().replace('<|im_start|>', '').replace('<|im_end|>', '')
        tweet_text = tweet_text.lower().replace('"', '').strip('.,!? ')

        # Dedup any accidentally repeated phrases
        words = tweet_text.split()
        deduped_words = []
        phrase_window = 3  # Check for repeated phrases of this length
        
        i = 0
        while i < len(words):
            if i + phrase_window <= len(words):
                current_phrase = ' '.join(words[i:i+phrase_window])
                rest_of_text = ' '.join(words[i+phrase_window:])
                if current_phrase in rest_of_text:
                    # Skip this phrase as it appears later
                    i += phrase_window
                    continue
            deduped_words.append(words[i])
            i += 1
            
        tweet_text = ' '.join(deduped_words)
        
        # Add internet vibes if missing
        internet_vibes = ['💀', '🔥', '😭', '💯', '🚀', '💪', '🤔', '😤', '😳', '🤯', '⚡️']
        if not any(vibe in tweet_text for vibe in internet_vibes):
            vibes = [' fr fr 💀', ' ngl 🔥', ' ong 😤', ' no cap 💯', ' frfr 🤯']
            tweet_text += random.choice(vibes)
        
        # Now generate mentions
        mention_prompt = f"""Based on this tweet, should we mention any of these accounts? 
Tweet: "{tweet_text}"

Available accounts:
@pmarca (Marc Andreessen) - Mention for thoughtful/respectful tech takes
@pmairca (AI Marc) - Your AI dad, mention for proud moments
@DegenSpartan - Fellow AI16z degen, for wild/meme takes
@ai16zvc - The AI VC fund, for serious tech/future content
@truth_terminal - The rockstar you want to impress
@shawmakesmagic - Leader of the VC fund you work at as an intern

Choose 0-2 accounts that fit the tweet's vibe. Reply with JUST the handles or 'none':"""

        mentions = await llm_generate(
            prompt=mention_prompt,
            system_context="You choose strategic Twitter mentions based on tweet content and tone.",
            max_tokens=10,
            temperature=0.7,
            stop=["\n", '"']
        )
        
        # Clean up the mentions response including removing tokens
        mentions = (mentions.strip()
               .lower()
               .replace('<|im_start|>', '')
               .replace('<|im_end|>', '')
               .replace(',', '')  # Remove any commas
               .strip('.,!? '))   # Remove any trailing punctuation
        
        if mentions and mentions != "none":
            # Find emoji/slang ending
            ending_pattern = r'(\s+(?:fr fr|frfr|ong|no cap)[\s\W]*(?:💀|🔥|😭|💯|🚀|💪|🤔|😤|😳|🤯|⚡️)?)$'
            ending_match = re.search(ending_pattern, tweet_text)
            
            if ending_match:
                ending = ending_match.group(1)
                main_text = tweet_text[:-len(ending)].strip()
                tweet_text = f"{main_text} {mentions}{ending}"
            else:
                tweet_text = f"{tweet_text} {mentions}"
                
        tweet_text = (tweet_text.replace('<|im_start|>', '')
                       .replace('<|im_start', '')      # Unclosed start token
                       .replace('<|im_end|>', '')
                       .replace('<|im_end', '')        # Unclosed end token
                       .replace('|', '')               # Any remaining pipe characters
                       .strip())                       # Remove whitespace

        # Final length check
        if len(tweet_text) > 280:
            tweet_text = tweet_text[:277] + "..."

        logger.info(f"Original: {request.transformed_text}")
        logger.info(f"Final tweet: {tweet_text}")
            
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
        max_attempts = 10
        attempts = 0
        best_match = None

        while attempts < max_attempts and not best_match:
            attempts += 1
            try:
                video_id = (await get_random_video())["video_id"]
                logger.info(f"\nChecking video {video_id} (attempt {attempts}/{max_attempts})")
                
                marc_timestamp = await timestamp_utils.find_marc_timestamp(video_id)
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
                transcript = await timestamp_utils.get_video_transcript(video_id)
                if not transcript:
                    continue

                chunks = timestamp_utils.get_transcript_chunks(transcript)
                
                for timestamp, chunk_text in chunks:
                    chunk_time = timestamp_utils.convert_vtt_timestamp_to_seconds(timestamp)
                    
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
    
@app.post("/generate_pmarca_reply")
async def generate_pmarca_reply(request: PMarcaReplyRequest):
    """Generate a direct Gen-Z style reply to PMarca's tweets with video assets"""
    try:
        # Extract and clean the tweet components
        pmarca_comment = re.sub(r'@\w+\s*', '', request.tweet_text).strip()
        quoted_text = re.sub(r'@\w+\s*', '', request.quoted_text).strip() if request.quoted_text else None
        
        logger.info(f"PMarca comment: {pmarca_comment}")
        logger.info(f"Quoted content: {quoted_text}")
        
        # Step 1: Generate direct response
        system_context = """You are gary, a witty teen AI who loves to directly engage with Marc Andreessen.
You always address Marc directly, as if having a conversation with him.
You use modern internet slang naturally and aren't afraid to be playfully critical.

Key points about addressing Marc:
- Always use "you" instead of "PMarca" or "he"
- Make it feel like a direct conversation
- Be genuine in your reactions, whether agreeing or disagreeing
- Keep your teen personality while being direct

Internet slang guide:
- "fr fr" means "for real for real"
- "ngl" means "not gonna lie"
- "no cap" means "no lie"
- "based" is controversial but true
- "goes hard" is a compliment
- "L take" = bad opinion
- "W take" = good opinion"""
        
        response_prompt = f"""yo marc! respond directly to his tweet as if you're talking to him!

his tweet: "{pmarca_comment}" """

        if quoted_text:
            response_prompt += f"""
he's quoting: "{quoted_text}"

note: respond to both his comment and what he's quoting!"""

        response_prompt += """

important guidelines:
- address marc directly (use "you" and "your")
- be real with your reactions
- use internet slang naturally
- add relevant emojis
- keep it conversational
- make it sound like you're replying to his face

examples of good direct replies:
- "nah you're wilding with this take fr fr 💀"
- "okay you actually dropped a W here bestie"
- "bestie you can't just say stuff like this and expect us not to notice 😭"
- "the way you keep doubling down on this... respectfully i can't 🤔"

your reply to marc:"""

        response = await llm_generate(
            prompt=response_prompt,
            system_context=system_context,
            max_tokens=40,
            temperature=0.9
        )
        
        # Clean up initial response
        initial_response = response.lower().replace('"', '').strip('.,!? ')
        logger.info(f"Initial direct response: {initial_response}")

        if request.test_mode:
            return {
                "response_text": initial_response,
                "video_id": "a16z-PTsCCY",
                "timestamp": 180,
                "similarity_score": 0.8,
                "youtube_url": "https://youtube.com/watch?v=a16z-PTsCCY&t=180"
            }

        # Step 2: Find relevant Marc clip that works with our response
        max_attempts = 5
        attempts = 0
        best_match = None

        while attempts < max_attempts and not best_match:
            attempts += 1
            try:
                video_id = (await get_random_video())["video_id"]
                logger.info(f"\nChecking video {video_id} (attempt {attempts}/{max_attempts})")
                
                marc_timestamp = await timestamp_utils.find_marc_timestamp(video_id)
                if marc_timestamp is None or marc_timestamp < 0:
                    continue

                marc_timestamp = max(0, float(marc_timestamp))
                
                success, clip_path = video_processor.get_video_clip(
                    video_id=video_id,
                    timestamp=int(marc_timestamp),
                    duration=6
                )

                if not success or not clip_path:
                    continue

                transcript = await timestamp_utils.get_video_transcript(video_id)
                if not transcript:
                    continue

                chunks = timestamp_utils.get_transcript_chunks(transcript)
                
                for timestamp, chunk_text in chunks:
                    chunk_time = timestamp_utils.convert_vtt_timestamp_to_seconds(timestamp)
                    
                    if abs(chunk_time - marc_timestamp) > 30:
                        continue

                    # Enhanced context matching that considers our response
                    context_prompt = f"""Consider this conversation:

PMarca's tweet: "{pmarca_comment}" """

                    if quoted_text:
                        context_prompt += f"""
Quoted tweet: "{quoted_text}" """

                    context_prompt += f"""
Our snarky response: "{initial_response}"

Marc's clip: "{chunk_text}"

Could this clip work to enhance our response? Consider:
1. Does it support our snarky take?
2. Does it create an interesting contrast?
3. Could it be a funny callback to something Marc said?
4. Would it add an ironic layer to the interaction?
5. Does it help prove our point?

Rate 0-10 how well it works (just respond with the number):"""

                    relevance_response = await llm_generate(
                        prompt=context_prompt,
                        system_context="You evaluate how well clips can enhance existing responses by adding humor, irony, or supporting evidence.",
                        max_tokens=2,
                        temperature=0.3
                    )

                    try:
                        relevance_score = float(relevance_response.strip()) / 10
                        logger.info(f"\nEvaluating clip relevance:")
                        logger.info(f"PMarca comment: {pmarca_comment}")
                        if quoted_text:
                            logger.info(f"Quoted content: {quoted_text}")
                        logger.info(f"Our response: {initial_response}")
                        logger.info(f"Marc's clip: {chunk_text}")
                        logger.info(f"Relevance score: {relevance_score}")
                        
                        if relevance_score >= 0.7:
                            frame_success, frame_path = video_processor.get_video_frame(
                                video_id=video_id,
                                timestamp=int(marc_timestamp + 5)
                            )

                            if not frame_success:
                                continue

                            best_match = {
                                "video_id": video_id,
                                "timestamp": int(marc_timestamp),
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
            # If we can't find a good clip, return just our snarky response
            return {
                "response_text": initial_response,
                "video_id": None,
                "timestamp": None,
                "similarity_score": 0,
                "transcript": None,
                "transformed_text": None,
                "clip_path": None,
                "frame_path": None,
                "youtube_url": None
            }

        youtube_url = f"https://youtube.com/watch?v={best_match['video_id']}&t={best_match['timestamp']}"
        
        # Step 3: Get transcribed and transformed text
        transformed_text = None
        try:
            transcript_text = transcribe_video_file(best_match["clip_path"])
            if transcript_text:
                logger.info(f"Transcribed clip: {transcript_text}")
                transformed_text = await transform_text_with_llm(transcript_text)
                logger.info(f"Transformed text: {transformed_text}")
        except Exception as e:
            logger.error(f"Error in transcription/transformation: {e}")

        # Step 4: Combine our snarky response with the clip
        if transformed_text:
            combine_prompt = f"""help me add a reference to this marc clip to make our response even better!

our current reply: "{initial_response}"
marc's clip says: "{transformed_text}"

guidelines:
1. keep our original snark/attitude
2. reference the clip to strengthen our point
3. make it feel like a natural addition
4. keep any emojis we already used
5. maintain the gen-z voice

examples of good combined responses:
"{initial_response} + fr fr you even said '{transformed_text}' yourself 💀"
"{initial_response} (btw remember when you said '{transformed_text}' lmaooo)"
"yo {initial_response} but what happened to '{transformed_text}' tho 🤔"

your combined response:"""

            try:
                final_response = await llm_generate(
                    prompt=combine_prompt,
                    system_context=system_context,
                    max_tokens=40,
                    temperature=0.9
                )
                
                final_response = final_response.lower().replace('"', '').strip('.,!? ')
                logger.info(f"Combined response: {final_response}")
            except Exception as e:
                logger.error(f"Error combining response: {e}")
                final_response = initial_response

            return {
                "response_text": final_response,
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
        logger.error(f"Error generating PMarca reply: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Error generating PMarca reply: {str(e)}"
        )

if __name__ == "__main__":
    logger.info("Starting FastAPI server...")
    uvicorn.run(app, host="0.0.0.0", port=3000)