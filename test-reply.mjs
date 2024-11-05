import fetch from 'node-fetch';
import { PassThrough } from 'stream';
import ffmpeg from 'fluent-ffmpeg';
import ffmpegPath from 'ffmpeg-static';
import 'dotenv/config';
import VideoProcessor from './videoProcessor.mjs';
import GlitchProcessor from './glitchProcessor.mjs';

// Configure ffmpeg
ffmpeg.setFfmpegPath(ffmpegPath);

const FASTAPI_URL = 'http://0.0.0.0:3000';
const MUSICGEN_URL = 'http://localhost:8001';

const videoProcessor = new VideoProcessor();
const glitchProcessor = new GlitchProcessor();

// Convert WAV buffer to MP3
const convertToMP3 = (inputBuffer) => {
    return new Promise((resolve, reject) => {
        const bufferStream = new PassThrough();
        bufferStream.end(inputBuffer);

        const outputStream = new PassThrough();
        const chunks = [];

        outputStream.on('data', chunk => chunks.push(chunk));
        outputStream.on('end', () => resolve(Buffer.concat(chunks)));
        outputStream.on('error', reject);

        ffmpeg(bufferStream)
            .inputFormat('wav')
            .format('mp3')
            .audioBitrate(128)
            .pipe(outputStream);
    });
};

// Poll for task completion
async function pollForTaskCompletion(taskId) {
    const maxTime = 1200000; // 10 minutes
    const interval = 10000; // 5 seconds
    const startTime = Date.now();

    while (Date.now() - startTime < maxTime) {
        const response = await fetch(`${MUSICGEN_URL}/tasks/${taskId}`);
        if (!response.ok) {
            throw new Error(`Failed to check task status: ${response.statusText}`);
        }

        const data = await response.json();
        
        if (data.status === 'completed' && data.audio) {
            return data.audio;
        } else if (data.status === 'failed') {
            throw new Error('Music generation failed');
        }

        await new Promise(resolve => setTimeout(resolve, interval));
    }
    
    throw new Error('Generation timed out');
}

async function testReplyGeneration() {
    const testComment = "@thepatch_gary @pmarca it would be funny and awesome if the beats starts in the bg and in sync with his speaking";
    
    console.log('\nTesting reply generation...');
    console.log('Test comment:', testComment);
    
    try {
        // Step 1: Get initial reply and clip info
        console.log('🎵 Finding relevant Marc clip...');
        const replyResponse = await fetch(`${FASTAPI_URL}/generate_reply`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ 
                tweet_text: testComment,
                test_mode: false
            })
        });
        
        if (!replyResponse.ok) {
            throw new Error(`Failed to generate reply: ${replyResponse.statusText}`);
        }
        
        const replyData = await replyResponse.json();
        console.log('\nGenerated reply data:', JSON.stringify(replyData, null, 2));

        // Step 2: Get video clip and frame
        console.log('\n🎬 Getting video assets...');
        const { video_id, timestamp, youtube_url } = replyData;

        // Get 6-second clip
        const clipResponse = await fetch(`${FASTAPI_URL}/get_clip`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ video_id, timestamp })
        });

        if (!clipResponse.ok) {
            throw new Error('Failed to get video clip');
        }

        const clipBuffer = Buffer.from(await clipResponse.arrayBuffer());
        console.log('Successfully got video clip');

        // Get last frame
        const frameResponse = await fetch(`${FASTAPI_URL}/get_frame`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ 
                video_id, 
                timestamp: timestamp + 5 
            })
        });

        if (!frameResponse.ok) {
            throw new Error('Failed to get video frame');
        }

        const frameBuffer = Buffer.from(await frameResponse.arrayBuffer());
        console.log('Successfully got last frame');

        // Step 3: Get transcription
        console.log('\n🎤 Transcribing clip...');
        const formData = new FormData();
        formData.append('file', new Blob([clipBuffer], { type: 'video/mp4' }));
        
        const transcriptResponse = await fetch(`${FASTAPI_URL}/transcribe_buffer`, {
            method: 'POST',
            body: formData
        });

        let speechText = null;
        let transformedText = null;
        let tweetText = null;

        if (transcriptResponse.ok) {
            const transcriptData = await transcriptResponse.json();
            if (transcriptData.success && transcriptData.text) {
                speechText = transcriptData.text;
                console.log('Transcription:', speechText);

                // Transform the text
                const transformResponse = await fetch(`${FASTAPI_URL}/transform_text`, {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ text: speechText })
                });

                if (transformResponse.ok) {
                    const transformData = await transformResponse.json();
                    if (transformData.success) {
                        transformedText = transformData.transformed;
                        console.log('Transformed text:', transformedText);

                        // Generate tweet text
                        tweetText = replyData.response_text;
                        console.log('Tweet text:', tweetText);
                    }
                }
            }
        }

        // Step 4: Generate music
        console.log('\n🎵 Generating music...');
        const musicResponse = await fetch(`${MUSICGEN_URL}/generate`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                url: youtube_url,
                currentTime: timestamp,
                model: 'thepatch/vanya_ai_dnb_0.1',
                promptLength: '6',
                duration: '28-30'
            })
        });

        if (!musicResponse.ok) {
            throw new Error('Failed to start music generation');
        }

        const { task_id } = await musicResponse.json();
        console.log('Waiting for music generation...');
        
        const audioData = await pollForTaskCompletion(task_id);
        const audioBuffer = Buffer.from(audioData, 'base64');
        const mp3Buffer = await convertToMP3(audioBuffer);
        console.log('Music generated successfully');

        // Step 5: Create final video
        console.log('\n🎬 Creating video...');
        const baseVideoBuffer = await videoProcessor.createVideoWithAudio(clipBuffer, mp3Buffer, {
            startWithVideo: true,
            effectIntensity: 0.6,
            transitionDuration: 2
        });
        
        console.log('Adding glitch effects...');
        const finalVideoBuffer = await glitchProcessor.processVideoBuffer(baseVideoBuffer);
        
        console.log('\n✨ Test complete! Generated:');
        console.log('- Snarky response:', replyData.response_text);
        console.log('- Video clip at timestamp:', timestamp);
        console.log('- Transcription:', speechText);
        console.log('- Transformed text:', transformedText);
        console.log('- Tweet text:', tweetText);
        console.log('- Generated music and video');

        // Save the video and audio files for inspection
        console.log('\nSaving test files...');
        const fs = await import('fs');
        await fs.promises.writeFile('test_output.mp4', finalVideoBuffer);
        await fs.promises.writeFile('test_output.mp3', mp3Buffer);
        console.log('Files saved as test_output.mp4 and test_output.mp3');

    } catch (error) {
        console.error('Error during test:', error);
        console.error('Full error details:', {
            name: error.name,
            message: error.message,
            stack: error.stack
        });
    }
}

console.log('Starting reply generation test...');
testReplyGeneration();