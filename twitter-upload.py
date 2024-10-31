import os
import sys
import time
import json
import requests
from requests_oauthlib import OAuth1
from dotenv import load_dotenv

def main():
    video_path = r"C:\Users\Kevin\Downloads\marc_beats_continued_manicdepressive.mp4"
    text = "got sumthin to say @pmarca @ai16z"
    
    # Use OAuth1 for both media upload and tweeting
    oauth = OAuth1(
        'iTkilo9nvRzJnGm1JTkWWJhVP',
        client_secret='ikELeM1ELm8hYAwcFmXVdYyo5LzLt1yo7SoWcAcNNBBBqOtsqj',
        resource_owner_key='1851424448265527296-A5wpLmV5gY6FZ6tb2rrt74SFKuJQaV',
        resource_owner_secret='R3CBHjIMSZEmqYv8r32erxDkomAs3n39owc4ZdNcoIxDL'
    )
    
    print('Starting upload process...')
    print(f'Checking video file: {video_path}')
    
    if not os.path.exists(video_path):
        print("ERROR: Video file not found!")
        return
        
    file_size = os.path.getsize(video_path)
    print(f'Video file size: {file_size} bytes')
    
    # INIT
    print('\nStarting INIT phase...')
    init_data = {
        'command': 'INIT',
        'media_type': 'video/mp4',
        'total_bytes': file_size,
        'media_category': 'tweet_video'
    }
    
    init_url = 'https://upload.twitter.com/1.1/media/upload.json'
    
    try:
        init_req = requests.post(
            init_url,
            data=init_data,
            auth=oauth
        )
        
        print('INIT Response Status:', init_req.status_code)
        print('INIT Response:', init_req.text)
        
        if init_req.status_code not in [200, 202]:
            print('INIT failed. Full response:')
            print(f'Status: {init_req.status_code}')
            print(f'Content: {init_req.text}')
            return
            
        media_id = init_req.json()['media_id_string']
        print(f'Received media_id: {media_id}')
        
        # APPEND
        print('\nStarting APPEND phase...')
        segment_id = 0
        bytes_sent = 0
        chunk_size = 4 * 1024 * 1024  # 4MB chunks
        
        with open(video_path, 'rb') as video_file:
            while bytes_sent < file_size:
                chunk = video_file.read(chunk_size)
                print(f'\nUploading chunk {segment_id + 1}...')
                
                append_data = {
                    'command': 'APPEND',
                    'media_id': media_id,
                    'segment_index': segment_id
                }
                files = {'media': chunk}
                
                append_req = requests.post(
                    init_url,
                    data=append_data,
                    files=files,
                    auth=oauth
                )
                
                print(f'APPEND chunk {segment_id + 1} response:', append_req.status_code)
                if append_req.status_code not in [200, 202, 204]:
                    print(f'APPEND chunk {segment_id + 1} failed:')
                    print(append_req.text)
                    return
                    
                segment_id += 1
                bytes_sent = video_file.tell()
                print(f'Progress: {bytes_sent}/{file_size} bytes')
                
        print('\nAll chunks uploaded successfully!')
        
        # FINALIZE
        print('\nStarting FINALIZE phase...')
        finalize_data = {
            'command': 'FINALIZE',
            'media_id': media_id
        }
        
        finalize_req = requests.post(
            init_url,
            data=finalize_data,
            auth=oauth
        )
        
        print('FINALIZE Response Status:', finalize_req.status_code)
        print('FINALIZE Response:', finalize_req.text)
        
        if finalize_req.status_code not in [200, 202]:
            print('FINALIZE failed')
            return
            
        processing_info = finalize_req.json().get('processing_info', None)
        
        # Check processing status
        while processing_info and processing_info['state'] in ['pending', 'in_progress']:
            check_after_secs = processing_info.get('check_after_secs', 1)
            print(f'\nProcessing... waiting {check_after_secs} seconds')
            print(f'Progress: {processing_info.get("progress_percent", 0)}%')
            time.sleep(check_after_secs)
            
            status_req = requests.get(
                init_url,
                params={
                    'command': 'STATUS',
                    'media_id': media_id
                },
                auth=oauth
            )
            
            processing_info = status_req.json().get('processing_info', None)
            print('Processing status:', status_req.text)
            
            if processing_info and processing_info['state'] == 'failed':
                print('Video processing failed:', processing_info.get('error'))
                return
            elif processing_info and processing_info['state'] == 'succeeded':
                print('Video processing completed successfully!')
                break
        
        # Post Tweet using v1.1 API with OAuth1
        print('\nPosting tweet...')
        tweet_data = {
            'status': text,
            'media_ids': media_id
        }
        
        tweet_req = requests.post(
            'https://api.twitter.com/1.1/statuses/update.json',
            data=tweet_data,
            auth=oauth
        )
        
        print('Tweet Response Status:', tweet_req.status_code)
        print('Tweet Response:', tweet_req.text)
        
        if tweet_req.status_code == 200:
            print('\nTweet posted successfully!')
            tweet_data = tweet_req.json()
            print(f'Tweet ID: {tweet_data.get("id_str")}')
        else:
            print('\nFailed to post tweet')
        
    except Exception as e:
        print(f'An error occurred: {str(e)}')

if __name__ == '__main__':
    main()