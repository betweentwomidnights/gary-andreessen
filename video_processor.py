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
            # Verify the existing file is valid
            try:
                cap = cv2.VideoCapture(output_path)
                if not cap.isOpened():
                    logger.warning(f"Existing file at {output_path} is invalid, removing...")
                    os.remove(output_path)
                else:
                    cap.release()
                    return output_path
            except Exception as e:
                logger.warning(f"Error checking existing file: {e}")
                if os.path.exists(output_path):
                    os.remove(output_path)

        ydl_opts = {
            'format': 'best[ext=mp4]',
            'outtmpl': output_path,
            'external_downloader': 'ffmpeg',
            'external_downloader_args': {
                'ffmpeg_i': [
                    '-ss', str(max(0, start_time)),  # Ensure non-negative
                    '-t', str(duration),
                    '-avoid_negative_ts', 'make_zero'  # Handle negative timestamps
                ]
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
                
                # Verify the downloaded file
                if os.path.exists(output_path):
                    cap = cv2.VideoCapture(output_path)
                    if not cap.isOpened():
                        logger.error("Downloaded file exists but cannot be opened")
                        return None
                    cap.release()
                    return output_path
                return None
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
        
    def _extract_frame_method1(self, video_path: str, video_id: str, timestamp: int) -> Tuple[bool, Optional[str]]:
            """Standard OpenCV frame extraction"""
            cap = cv2.VideoCapture(video_path)
            ret, frame = cap.read()
            cap.release()

            if not ret or frame is None:
                return False, None

            frame_path = os.path.join(self.temp_dir, f"{video_id}_{timestamp}_frame.jpg")
            cv2.imwrite(frame_path, frame)
            return True, frame_path

    def _extract_frame_method2(self, video_path: str, video_id: str, timestamp: int) -> Tuple[bool, Optional[str]]:
        """Extract frame using FFmpeg directly"""
        frame_path = os.path.join(self.temp_dir, f"{video_id}_{timestamp}_frame.jpg")
        try:
            import subprocess
            cmd = [
                'ffmpeg', '-i', video_path,
                '-vf', 'select=eq(n\\,0)',
                '-vframes', '1',
                frame_path
            ]
            subprocess.run(cmd, check=True, capture_output=True)
            return True, frame_path
        except Exception as e:
            logger.warning(f"FFmpeg extraction failed: {e}")
            return False, None

    def _extract_frame_method3(self, video_path: str, video_id: str, timestamp: int) -> Tuple[bool, Optional[str]]:
        """Extract frame by seeking to specific position"""
        cap = cv2.VideoCapture(video_path)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        if frame_count > 0:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_count // 2)  # Try middle frame
            ret, frame = cap.read()
            cap.release()

            if ret and frame is not None:
                frame_path = os.path.join(self.temp_dir, f"{video_id}_{timestamp}_frame.jpg")
                cv2.imwrite(frame_path, frame)
                return True, frame_path

        return False, None