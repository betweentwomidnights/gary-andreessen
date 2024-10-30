import yt_dlp
import cv2
import os
import tempfile
import logging
from typing import Tuple, Optional

logger = logging.getLogger(__name__)

class VideoProcessor:
    def __init__(self):
        self.temp_dir = tempfile.gettempdir()

    def _download_video_segment(self, video_id: str, start_time: int, duration: int = 6) -> Optional[str]:
        """
        Download a specific segment of a YouTube video
        Returns the path to the downloaded video file
        """
        output_path = os.path.join(self.temp_dir, f"{video_id}_{start_time}_{duration}.mp4")
        
        if os.path.exists(output_path):
            return output_path

        ydl_opts = {
            'format': 'best[ext=mp4]',
            'outtmpl': output_path,
            # Use external downloader with time range
            'external_downloader': 'ffmpeg',
            'external_downloader_args': {
                'ffmpeg_i': ['-ss', str(start_time), '-t', str(duration)]
            },
            'postprocessors': [{
                'key': 'FFmpegVideoConvertor',
                'preferedformat': 'mp4',
            }],
        }

        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                url = f"https://www.youtube.com/watch?v={video_id}"
                ydl.download([url])
                return output_path if os.path.exists(output_path) else None
        except Exception as e:
            logger.error(f"Error downloading video segment: {e}")
            return None

    def get_video_clip(self, video_id: str, timestamp: int, duration: int = 6) -> Tuple[bool, Optional[str]]:
        """
        Extract a clip from the video starting at timestamp
        Returns (success, path_to_clip)
        """
        try:
            # Download the video segment
            video_path = self._download_video_segment(video_id, timestamp, duration)
            if not video_path:
                return False, None
            return True, video_path
        except Exception as e:
            logger.error(f"Error in get_video_clip: {e}")
            return False, None

    def get_video_frame(self, video_id: str, timestamp: int) -> Tuple[bool, Optional[str]]:
        """
        Get a single frame from video
        Returns (success, path_to_frame)
        """
        try:
            # Download the video segment
            video_path = self._download_video_segment(video_id, timestamp, 1)
            if not video_path:
                return False, None

            # Extract the frame
            cap = cv2.VideoCapture(video_path)
            ret, frame = cap.read()
            cap.release()

            if not ret:
                return False, None

            # Save the frame
            frame_path = os.path.join(self.temp_dir, f"{video_id}_{timestamp}_frame.jpg")
            cv2.imwrite(frame_path, frame)

            # Clean up the video file since we only need the frame
            try:
                os.remove(video_path)
            except Exception as e:
                logger.warning(f"Failed to clean up temporary video file: {e}")

            return True, frame_path

        except Exception as e:
            logger.error(f"Error in get_video_frame: {e}")
            return False, None

    def get_last_frame_from_clip(self, clip_path: str) -> Tuple[bool, Optional[str]]:
        """
        Extract the last frame from a video clip
        Returns (success, path_to_frame)
        """
        try:
            cap = cv2.VideoCapture(clip_path)
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            # Set position to last frame
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_count - 1)
            ret, frame = cap.read()
            cap.release()

            if not ret:
                return False, None

            # Save the frame
            frame_path = clip_path.replace('.mp4', '_last_frame.jpg')
            cv2.imwrite(frame_path, frame)

            return True, frame_path

        except Exception as e:
            logger.error(f"Error in get_last_frame_from_clip: {e}")
            return False, None