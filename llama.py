from fastapi import FastAPI, HTTPException
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

# Initialize video processor
video_processor = VideoProcessor()

class VideoRequest(BaseModel):
    video_id: str
    timestamp: Optional[int] = None

PLAYLIST_ID = "PLTgvaF3a9YMipq-kfDW1Tkrl-sG7eTQqV"
MAX_RETRIES = 3

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

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

async def find_marc_timestamp(video_id: str) -> Optional[int]:
    """Find a timestamp where Marc is speaking"""
    transcript = await get_video_transcript(video_id)
    if not transcript:
        return None
    
    # Get chunks and shuffle them
    chunks = get_chunks(transcript)
    random.shuffle(chunks)
    logger.info(f"Processing {len(chunks)} chunks for video {video_id}")
    
    # Try chunks until we find Marc speaking
    for i, (start_time, chunk) in enumerate(chunks):
        try:
            prompt = f"""Find a timestamp where Marc Andreessen is speaking in this transcript chunk.
If Marc is not speaking, return 0.
If Marc is speaking, return the timestamp.

Transcript chunk:
{chunk}

Timestamp:"""

            response = llm(
                prompt,
                max_tokens=10,
                temperature=0.1,
                stop=["\n", "\r"],
                echo=False
            )
            
            # Log the raw response for debugging
            logger.info(f"Raw LLM response: {json.dumps(response)}")
            
            raw_timestamp = response['choices'][0]['text'].strip()
            logger.info(f"Chunk {i+1} analysis - Raw timestamp: '{raw_timestamp}'")
            
            # If we got a timestamp (not 0), convert and return it immediately
            if raw_timestamp != "0":
                seconds = convert_timestamp_to_seconds(raw_timestamp)
                if seconds > 0:
                    logger.info(f"Found valid timestamp: {raw_timestamp} ({seconds} seconds)")
                    return seconds
                    
        except Exception as e:
            logger.error(f"Error processing chunk {i+1}: {str(e)}")
            continue
    
    logger.info(f"Finished processing all chunks for video {video_id}, Marc not found")
    return None

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
    """Find a timestamp where Marc is speaking, trying multiple videos if needed"""
    videos = await get_playlist_videos()
    tried_videos = set()
    
    # Try the requested video first
    tried_videos.add(request.video_id)
    timestamp = await find_marc_timestamp(request.video_id)
    
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
        logger.info(f"Found timestamp: {timestamp}")
        return {"timestamp": timestamp}
    
    raise HTTPException(
        status_code=422,
        detail="Could not find a suitable timestamp in any of the tried videos"
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

if __name__ == "__main__":
    logger.info("Starting FastAPI server...")
    uvicorn.run(app, host="0.0.0.0", port=3000)