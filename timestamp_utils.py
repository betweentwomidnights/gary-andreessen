# timestamp_utils.py - For timestamp-related functions
from typing import Optional, List, Tuple
import yt_dlp
import logging
import re
import requests
from fastapi import HTTPException
import random

logger = logging.getLogger(__name__)

class TimestampUtils:
    def __init__(self, playlist_id: Optional[str] = None, llm=None):
        self.playlist_id = playlist_id
        self.llm = llm  # We'll need this for find_marc_timestamp
    
    @staticmethod
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

    @staticmethod
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

    
    async def get_playlist_videos(self) -> List[str]:
        """Get list of video IDs from playlist"""
        if not self.playlist_id:
            raise ValueError("Playlist ID not set")
            
        ydl_opts = {
            'extract_flat': True,
            'quiet': True,
        }
    
        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                playlist_url = f"https://www.youtube.com/playlist?list={self.playlist_id}"
                playlist_info = ydl.extract_info(playlist_url, download=False)
                videos = [entry['id'] for entry in playlist_info['entries'] if entry['id']]
                logger.info(f"Found {len(videos)} videos in playlist")
                return videos
        except Exception as e:
            logger.error(f"Error fetching playlist: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Error fetching playlist: {str(e)}")

    async def get_video_transcript(self, video_id: str) -> Optional[str]:
        """Get video transcript, returns None if not found"""
        logger.info(f"Fetching transcript for video {video_id}")
        
        # Configuration for yt-dlp with more options
        ydl_opts = {
            'writesubtitles': True,
            'writeautomaticsub': True,
            'subtitlesformat': 'vtt',
            'subtitleslangs': ['en'],
            'skip_download': True,
            'ignoreerrors': True,  # Don't stop on download errors
            'no_warnings': True,   # Suppress warnings
            'quiet': True,         # Suppress progress output
        }

        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                # First try to get manual captions
                info = ydl.extract_info(f"https://www.youtube.com/watch?v={video_id}", download=False)
                
                if not info:
                    logger.error(f"Could not fetch video info for {video_id}")
                    return None
                
                # Try manual captions first
                if 'subtitles' in info and 'en' in info['subtitles']:
                    captions = info['subtitles']['en']
                    logger.info(f"Found manual captions for {video_id}")
                elif 'automatic_captions' in info and 'en' in info['automatic_captions']:
                    captions = info['automatic_captions']['en']
                    logger.info(f"Found automatic captions for {video_id}")
                else:
                    logger.warning(f"No captions found for {video_id}")
                    return None

                # Try different caption formats in order of preference
                preferred_formats = ['vtt', 'srv3', 'srv2', 'srv1', 'ttml', 'json3']
                
                for format_type in preferred_formats:
                    caption_url = None
                    for caption in captions:
                        if caption['ext'] == format_type:
                            caption_url = caption['url']
                            break
                    
                    if caption_url:
                        try:
                            response = requests.get(caption_url, timeout=10)
                            response.raise_for_status()
                            vtt_content = response.text
                            
                            if not vtt_content.strip():
                                logger.warning(f"Empty caption content for {video_id} in format {format_type}")
                                continue
                            
                            # Parse VTT to text with timestamps
                            lines = vtt_content.split('\n')
                            transcript = []
                            current_time = None
                            current_text = []
                            
                            for line in lines:
                                if '-->' in line:
                                    if current_time and current_text:
                                        clean_text = self.clean_transcript_text(' '.join(current_text))
                                        if clean_text:
                                            transcript.append(f"[{current_time}] {clean_text}")
                                    current_time = line.split('-->')[0].strip()
                                    current_text = []
                                elif line.strip() and not line.startswith('WEBVTT'):
                                    current_text.append(line.strip())
                            
                            # Add the last chunk
                            if current_time and current_text:
                                clean_text = self.clean_transcript_text(' '.join(current_text))
                                if clean_text:
                                    transcript.append(f"[{current_time}] {clean_text}")
                            
                            if transcript:
                                logger.info(f"Successfully extracted transcript for {video_id} using {format_type} format")
                                return '\n'.join(transcript)
                            
                        except requests.RequestException as e:
                            logger.warning(f"Error fetching {format_type} captions for {video_id}: {str(e)}")
                            continue
                        except Exception as e:
                            logger.warning(f"Error processing {format_type} captions for {video_id}: {str(e)}")
                            continue
                
                logger.error(f"Tried all caption formats but could not get transcript for {video_id}")
                return None
                
        except Exception as e:
            logger.error(f"Error fetching transcript for {video_id}: {str(e)}")
            return None
        
    
    def clean_repeated_phrases(self, text: str) -> str:
        """Remove redundant repeated phrases from text."""
        # Split into words to find repetitions
        words = text.split()
        if len(words) < 4:  # Don't process very short texts
            return text
        
        # Look for repeating sequences of 3+ words
        cleaned_words = []
        i = 0
        while i < len(words):
            # Skip if we're near the end
            if i > len(words) - 3:
                cleaned_words.extend(words[i:])
                break
            
            # Look for repeating sequences starting at current position
            found_repeat = False
            for seq_length in range(3, 8):  # Check sequences of 3-7 words
                if i + seq_length * 2 > len(words):
                    break
                
                # Get the sequence and the following sequence
                seq1 = ' '.join(words[i:i+seq_length])
                seq2 = ' '.join(words[i+seq_length:i+seq_length*2])
            
                # If they match, we found a repetition
                if seq1.lower() == seq2.lower():
                    cleaned_words.extend(words[i:i+seq_length])
                    i += seq_length * 2  # Skip both sequences
                    found_repeat = True
                    break
                
            if not found_repeat:
                cleaned_words.append(words[i])
                i += 1
            
        return ' '.join(cleaned_words)

    def get_transcript_chunks(self, transcript: str) -> List[Tuple[str, str]]:
        """Get cleaned chunks from transcript."""
        chunks = []
        if not transcript:
            return chunks
        
        # Split transcript into lines
        lines = transcript.split('\n')
    
        for line in lines:
            try:
                # Extract timestamp and text using regex
                match = re.match(r'\[([\d:\.]+)\]\s*(.+)', line)
                if not match:
                    continue
                
                start_time, text = match.groups()
                text = text.strip()
            
                if not text or not start_time:
                    continue
            
                # Clean up repeated phrases
                cleaned_text = self.clean_repeated_phrases(text)
            
                # Only add if the cleaned text is still substantial
                if len(cleaned_text.split()) >= 5:  # Minimum 5 words
                    chunks.append((start_time, cleaned_text))
            
            except Exception as e:
                logger.error(f"Error processing transcript line: {e}")
                logger.error(f"Line was: {line if 'line' in locals() else 'No line'}")
                continue
    
        return chunks
        
    
    async def find_marc_timestamp(self, video_id: str) -> Optional[int]:
        """Find a timestamp where Marc begins speaking"""
        try:
            if not self.llm:
                raise ValueError("LLM not initialized")
            
            transcript = await self.get_video_transcript(video_id)
            if not transcript:
                logger.info(f"No transcript found for video {video_id}")
                return None
        
            chunks = self.get_transcript_chunks(transcript)
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
            - Almost always uses casual filler words like "um", "you know"
            - Often gets excited about technical topics
            - Frequently relates topics to software or technology
        
            Important: Since Marc is usually being interviewed, be very careful about:
            - Introductory phrases like "Please welcome", "Let me ask you", "What do you think" - these are the host
            - Mentions of 'we' or 'they' will often be Marc speaking."""
    
            for i, (start_time, chunk) in enumerate(chunks):
                try:
                    # Validate chunk before processing
                    if not chunk or len(chunk.split()) < 10:
                        logger.debug(f"Skipping short chunk {i}: {chunk}")
                        continue
            
                    logger.info(f"\nProcessing chunk {i} at {start_time}:")
                    logger.info(f"Chunk text: {chunk}")

                    analysis_prompt = f"""Analyze this podcast segment, considering both Marc's speaking styles and the context that he's usually being interviewed.

    Transcript: "{chunk}"

    Then provide your analysis in exactly this format:
    Is Marc Speaking: [yes/no]
    Confidence Rating: [number 1-10]
    Reasoning: [your explanation]"""

                    # Use the llm instance directly
                    formatted_prompt = f"""<|im_start|>system
    You are a helpful AI assistant specialized in analyzing speech patterns and identifying speakers. 
    You aim to provide clear, accurate, and relevant responses.
    <|im_end|>
    <|im_start|>user
    {analysis_prompt}<|im_end|>
    <|im_start|>assistant
    """
                
                    response = self.llm(
                        formatted_prompt,
                        max_tokens=200,  # Increased for additional analysis
                        temperature=0.1,
                        stop=["Transcript:", "<|im_end|>"],
                        echo=False
                    )
                
                    # Extract just the assistant's response
                    response = response['choices'][0]['text'].strip()
            
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
                    
                        # Look for interviewer indicators
                        interviewer_match = re.search(r'Interviewer Indicators:\s*(.+?)(?=\nReasoning:|$)', cleaned_response, re.DOTALL)
                        has_interviewer_indicators = bool(interviewer_match and any([
                            indicator.strip()
                            for indicator in interviewer_match.group(1).split(',')
                            if indicator.strip()
                        ]))
                
                        logger.info(f"Parsed results - Is Marc: {is_marc_speaking}, Confidence: {confidence}/10")
                        logger.info(f"Interviewer indicators found: {has_interviewer_indicators}")
                
                        # Accept only if we're confident it's Marc and there are no interviewer indicators
                        if is_marc_speaking and confidence >= 7 and not has_interviewer_indicators:
                            chunk_start = self.convert_vtt_timestamp_to_seconds(start_time)
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
     
    def clean_transcript_text(self, text: str) -> str:
        """Clean WebVTT formatting and deduplicate repeated phrases"""
        try:
            if not text:
                return ""
        
            # Remove WebVTT timing tags (<00:02:15.900>)
            text = re.sub(r'<\d{2}:\d{2}:\d{2}\.\d{3}>', '', text)
        
            # Remove WebVTT style tags (<c> and </c>)
            text = re.sub(r'</?c>', '', text)
        
            # Remove HTML tags
            text = re.sub(r'<[^>]+>', ' ', text)
        
            # Remove timestamp markers
            text = re.sub(r'\[\d+:\d+:\d+\.\d+\]', ' ', text)
        
            # Clean up multiple spaces
            text = re.sub(r'\s+', ' ', text)
        
            # Deduplicate repeated phrases using our existing method
            cleaned_text = self.clean_repeated_phrases(text)
        
            return cleaned_text.strip()
        
        except Exception as e:
            logger.error(f"Error in clean_transcript_text: {e}")
            logger.error(f"Input text was: {text}")
            return text  # Return original text if cleaning fails