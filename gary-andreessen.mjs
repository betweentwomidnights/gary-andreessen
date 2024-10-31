import { Client, GatewayIntentBits, AttachmentBuilder } from 'discord.js';
import fetch from 'node-fetch';
import { PassThrough } from 'stream';
import ffmpeg from 'fluent-ffmpeg';
import ffmpegPath from 'ffmpeg-static';
import 'dotenv/config';
import os from 'os';
import fs from 'fs';
import path from 'path';
import VideoProcessor from './videoProcessor.mjs';
import GlitchProcessor from './glitchProcessor.mjs';
import TwitterHandler from './twitter-handler.mjs';

const twitterHandler = new TwitterHandler();

const videoProcessor = new VideoProcessor();

const glitchProcessor = new GlitchProcessor();

// Configure ffmpeg
ffmpeg.setFfmpegPath(ffmpegPath);

// Add a simple cache to store the last generation's info per channel
const lastGenerationCache = new Map();

// API endpoints
const FASTAPI_URL = 'http://0.0.0.0:3000';
const MUSICGEN_URL = 'http://localhost:8001';

// Initialize Discord client
const client = new Client({
    intents: [
        GatewayIntentBits.Guilds,
        GatewayIntentBits.GuildMessages,
        GatewayIntentBits.MessageContent,
        GatewayIntentBits.DirectMessages
    ]
});

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

// Get random video from playlist
async function getRandomVideo() {
    try {
        console.log('Fetching random video from:', `${FASTAPI_URL}/random`);
        const response = await fetch(`${FASTAPI_URL}/random`);
        console.log('Response status:', response.status);
        
        if (!response.ok) {
            const errorText = await response.text();
            console.error('Error response:', errorText);
            throw new Error(`Failed to get random video: ${response.statusText}. Response: ${errorText}`);
        }
        
        const data = await response.json();
        console.log('Received video data:', data);
        
        if (!data.video_id) {
            throw new Error('No video_id in response');
        }
        
        return data.video_id;
    } catch (error) {
        console.error('Detailed error in getRandomVideo:', error);
        if (error.cause) console.error('Error cause:', error.cause);
        throw error;
    }
}

// Analyze video for Marc's timestamp
async function analyzeVideo(videoId) {
    const response = await fetch(`${FASTAPI_URL}/analyze`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ video_id: videoId })
    });
    if (!response.ok) {
        throw new Error(`Failed to analyze video: ${response.statusText}`);
    }
    const data = await response.json();
    return data.timestamp;
}

// Generate music from YouTube URL
async function generateMusic(youtubeUrl, timestamp) {
    const response = await fetch(`${MUSICGEN_URL}/generate`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
            url: youtubeUrl,
            currentTime: timestamp,
            model: 'thepatch/vanya_ai_dnb_0.1',
            promptLength: '6',
            duration: '28-30'
        })
    });
    if (!response.ok) {
        throw new Error(`Failed to generate music: ${response.statusText}`);
    }
    return await response.json();
}

// Add new function to handle continuation
async function continueGeneration(channelId, message) {
    const lastGeneration = lastGenerationCache.get(channelId);
    if (!lastGeneration) {
        await message.reply('theres no recent beat to continue from! generate something first using !generate or by mentioning me.');
        return;
    }

    const processingMessage = await message.reply('🎵 continuing from the previous beat...');

    try {
        // Generate continuation audio
        const response = await fetch(`${MUSICGEN_URL}/continue`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                task_id: lastGeneration.taskId,
                model: 'thepatch/vanya_ai_dnb_0.1',
                prompt_duration: 6,
                audio: lastGeneration.audioData
            })
        });

        if (!response.ok) {
            throw new Error(`Failed to continue generation: ${response.statusText}`);
        }

        const { task_id } = await response.json();
        
        await processingMessage.edit("🎼 extending your beat... (this may take a few minutes, plz dont spam me)");
        
        const audioData = await pollForTaskCompletion(task_id);
        const audioBuffer = Buffer.from(audioData, 'base64');
        const mp3Buffer = await convertToMP3(audioBuffer);

        // Get the original video clip for the continuation
        const clipResponse = await fetch(`${FASTAPI_URL}/get_clip`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ 
                video_id: lastGeneration.videoId, 
                timestamp: lastGeneration.timestamp 
            })
        });

        if (!clipResponse.ok) {
            throw new Error('Failed to get original video clip');
        }

        const originalVideoBuffer = Buffer.from(await clipResponse.arrayBuffer());
        
        // Create base continuation video
        await processingMessage.edit("🎬 creating base video...");
        const baseVideoBuffer = await videoProcessor.createContinuationVideo(originalVideoBuffer, mp3Buffer);
        
        // Apply glitch effects
        await processingMessage.edit("✨ adding glitch effects...");
        const finalVideoBuffer = await glitchProcessor.processVideoBuffer(baseVideoBuffer);
        
        // Update cache with new generation info
        lastGenerationCache.set(channelId, {
            ...lastGeneration,
            taskId: task_id,
            audioData: audioData,
            videoBuffer: finalVideoBuffer  // Update with new video
        });

        // Send both video and audio
        const videoAttachment = new AttachmentBuilder(finalVideoBuffer, { name: 'marc_beats_continued.mp4' });
        const audioAttachment = new AttachmentBuilder(mp3Buffer, { name: 'marc_beats_continued.mp3' });

        // Construct reply message using stored text
        let replyContent = '🎵 heres your continued beat homie\n';
        if (lastGeneration.originalTranscript) {
            replyContent += `🎤 original: "${lastGeneration.originalTranscript}"\n`;
            if (lastGeneration.transcript) {
                replyContent += `✨ transformed: "${lastGeneration.transcript}"\n`;
            }
            if (lastGeneration.tweetText) {
                replyContent += `🐦 tweet draft: "${lastGeneration.tweetText}"\n`;
            }
        }
        replyContent += '📱 say "post that" if u want me to tweet this';

        await message.reply({
            content: replyContent,
            files: [videoAttachment, audioAttachment]
        });

    } catch (error) {
        console.error('Error in continueGeneration:', error);
        await message.reply('uh oh sumthin happened while extending that beat bro. try again');
    } finally {
        if (processingMessage) {
            await processingMessage.delete().catch(() => {});
        }
    }
}

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

// Parse YouTube URL for timestamp
function parseYouTubeUrl(url) {
    try {
        const urlObj = new URL(url);
        let videoId = null;
        let timestamp = 0;

        if (urlObj.hostname === 'youtu.be') {
            videoId = urlObj.pathname.slice(1);
            timestamp = urlObj.searchParams.get('t') || 0;
        } else if (urlObj.hostname.includes('youtube.com')) {
            videoId = urlObj.searchParams.get('v');
            timestamp = urlObj.searchParams.get('t') || 0;

            if (!timestamp && urlObj.hash.includes('t=')) {
                timestamp = urlObj.hash.split('t=')[1];
            }
        } else {
            return { valid: false };
        }

        timestamp = parseTimestamp(timestamp);
        return { valid: true, videoId, timestamp };
    } catch (error) {
        console.error('Error parsing YouTube URL:', error);
        return { valid: false };
    }
}

// Parse timestamp into seconds
function parseTimestamp(timestamp) {
    if (typeof timestamp === 'string') {
        let totalSeconds = 0;
        const timeParts = timestamp.match(/(\d+)(h|m|s)/g);

        if (timeParts) {
            for (const part of timeParts) {
                const value = parseInt(part.slice(0, -1));
                const unit = part.slice(-1);
                if (unit === 'h') totalSeconds += value * 3600;
                else if (unit === 'm') totalSeconds += value * 60;
                else if (unit === 's') totalSeconds += value;
            }
        } else {
            totalSeconds = parseInt(timestamp, 10);
        }
        return totalSeconds;
    }
    return parseInt(timestamp, 10) || 0;
}

const createVideoWithAudio = async (videoOrImage, audioBuffer, useEffects = false) => {
    try {
        if (useEffects) {
            return await videoProcessor.createVideoWithAudio(videoOrImage, audioBuffer, {
                startWithVideo: true,
                effectIntensity: 0.6,
                transitionDuration: 2
            });
        } else {
            return await videoProcessor.createBasicVideoWithAudio(videoOrImage, audioBuffer);
        }
    } catch (error) {
        console.error('Error in video processing:', error);
        throw error;
    }
};

async function getVideoTitle(videoId) {
    try {
        const response = await fetch(`${FASTAPI_URL}/video_title`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ video_id: videoId })
        });
        
        if (!response.ok) {
            throw new Error('Failed to get video title');
        }
        
        const data = await response.json();
        return data.title;
    } catch (error) {
        console.error('Error getting video title:', error);
        return null;
    }
}

async function generateTweetText(transformedText, videoId) {
    try {
        // Get video title
        const videoTitle = await getVideoTitle(videoId);
        if (!videoTitle) {
            console.warn('Could not get video title');
            return transformedText; // Fall back to transformed text only
        }

        // Generate tweet text using both transformed text and video title
        const tweetResponse = await fetch(`${FASTAPI_URL}/generate_tweet`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                transformed_text: transformedText,
                video_title: videoTitle
            })
        });

        if (!tweetResponse.ok) {
            throw new Error('Failed to generate tweet text');
        }

        const tweetData = await tweetResponse.json();
        return tweetData.success ? tweetData.tweet : transformedText;
    } catch (error) {
        console.error('Error generating tweet text:', error);
        return transformedText; // Fall back to transformed text
    }
}

async function handleGenerateCommand(message, youtubeUrl, transcript = null) {
    const { valid, videoId, timestamp } = parseYouTubeUrl(youtubeUrl);
    
    if (!valid) {
        await message.reply('bro i need a valid url. do one with a timestamp as well or ill just start from the beginning of the vid');
        return;
    }

    const processingMessage = await message.reply('🎵 cooking...');

    try {
        // Get the video clip first (6 seconds)
        const clipResponse = await fetch(`${FASTAPI_URL}/get_clip`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ video_id: videoId, timestamp: timestamp })
        });

        if (!clipResponse.ok) {
            throw new Error('Failed to get video clip');
        }

        const clipBuffer = Buffer.from(await clipResponse.arrayBuffer());

        // Get the last frame for continuing later
        const frameResponse = await fetch(`${FASTAPI_URL}/get_frame`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ video_id: videoId, timestamp: timestamp + 5 })
        });

        if (!frameResponse.ok) {
            throw new Error('Failed to get video frame');
        }

        const frameBuffer = Buffer.from(await frameResponse.arrayBuffer());

        // Try to transcribe the clip using Whisper
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
                await processingMessage.edit(`🎤 transcribed: "${speechText}"\n🔄 transforming...`);
                
                // Transform the transcribed text
                try {
                    const transformResponse = await fetch(`${FASTAPI_URL}/transform_text`, {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({ text: speechText })
                    });

                    if (transformResponse.ok) {
                        const transformData = await transformResponse.json();
                        if (transformData.success) {
                            transformedText = transformData.transformed;
                            // Generate tweet text right after getting transformed text
                            tweetText = await generateTweetText(transformedText, videoId);
                            await processingMessage.edit(
                                `🎤 original: "${speechText}"\n` +
                                `✨ transformed: "${transformedText}"\n` +
                                `🐦 tweet draft: "${tweetText}"\n` +
                                `🎵 now cookin up a beat...`
                            );
                        }
                    }
                } catch (error) {
                    console.error('Error transforming text:', error);
                    // Continue with original text if transformation fails
                }
            }
        }

        // Generate the music
        const { task_id } = await generateMusic(youtubeUrl, timestamp);
        if (!task_id) {
            throw new Error('No task ID received from music generator');
        }

        await processingMessage.edit("🎼 ima make your beat now... (this may take a few minutes plz dont spam me)");
        
        const audioData = await pollForTaskCompletion(task_id);
        const audioBuffer = Buffer.from(audioData, 'base64');
        const mp3Buffer = await convertToMP3(audioBuffer);

        // First create the base video with VideoProcessor
        await processingMessage.edit("🎬 creating base video...");
        const baseVideoBuffer = await videoProcessor.createVideoWithAudio(clipBuffer, mp3Buffer, {
            startWithVideo: true,
            effectIntensity: 0.6,
            transitionDuration: 2
        });
        
        // Then apply glitch effects
        await processingMessage.edit("✨ adding glitch effects...");
        const finalVideoBuffer = await glitchProcessor.processVideoBuffer(baseVideoBuffer);
        
        // Store everything in the cache
        lastGenerationCache.set(message.channelId, {
            taskId: task_id,
            audioData: audioData,
            videoId: videoId,
            timestamp: timestamp,
            frameBuffer: frameBuffer,
            videoBuffer: finalVideoBuffer,
            transcript: transformedText,        // Store transformed text
            originalTranscript: speechText,     // Store original transcript
            tweetText: tweetText,              // Store generated tweet text
            mp3Buffer: mp3Buffer
        });

        // Send both video and audio files
        const videoAttachment = new AttachmentBuilder(finalVideoBuffer, { name: 'marc_beats.mp4' });
        const audioAttachment = new AttachmentBuilder(mp3Buffer, { name: 'marc_beats.mp3' });

        // Construct reply message with all information
        let replyContent = `🎵 heres what i think.\n`;
        if (speechText) {
            replyContent += `🎤 original: "${speechText}"\n`;
            if (transformedText) {
                replyContent += `✨ transformed: "${transformedText}"\n`;
            }
            if (tweetText) {
                replyContent += `🐦 tweet draft: "${tweetText}"\n`;
            }
        }
        replyContent += `📺 original clip: ${youtubeUrl}\n`;
        replyContent += `📱 say "post that" if u want me to tweet this`;

        await message.reply({
            content: replyContent,
            files: [videoAttachment, audioAttachment]
        });

    } catch (error) {
        console.error('Error in handleGenerateCommand:', error);
        await message.reply('sorry, sumthin went wrong while generating your beat. try again mang');
    } finally {
        if (processingMessage) {
            await processingMessage.delete().catch(() => {});
        }
    }
}

// Update the messageCreate event handler
client.on('messageCreate', async (message) => {
    if (message.author.bot) return;

    // Handle !generate command
    if (message.content.startsWith('!generate')) {
        const args = message.content.split(' ');
        if (args.length < 2) {
            await message.reply('bro i need a valid url. do one with a timestamp as well or ill just start from the beginning of the vid.');
            return;
        }
        await handleGenerateCommand(message, args[1]);
        return;
    }
    
    // Handle continue command
    if (message.mentions.has(client.user) && message.content.toLowerCase().includes('continue')) {
        await continueGeneration(message.channelId, message);
        return;
    }

    // Handle "post that" command
    if (message.mentions.has(client.user) && message.content.toLowerCase().includes('post that')) {
    const lastGeneration = lastGenerationCache.get(message.channelId);
    if (!lastGeneration) {
        await message.reply('nothing to post yet! generate something first by mentioning me.');
        return;
    }

    const processingMsg = await message.reply('📤 getting ready to post...');

    try {
        // Use the stored tweet text, or generate it if somehow missing
        let tweetText = lastGeneration.tweetText;
        if (!tweetText && lastGeneration.transcript) {
            tweetText = await generateTweetText(lastGeneration.transcript, lastGeneration.videoId);
        }

        if (!tweetText) {
            throw new Error('No tweet text available');
        }

        // Post the video
        const result = await twitterHandler.postVideoWithText(
            lastGeneration.videoBuffer,
            tweetText
        );

        // If successful, reply with the tweet URL
        if (result.success) {
            let replyText = `✨ just posted this heat to twitter!\n` +
                           `🐦 tweet: "${tweetText}"\n` +
                           `🔗 check it: https://twitter.com/thepatch_gary/status/${result.tweetId}`;
            
            await processingMsg.edit(replyText);
        } else {
            throw new Error('Failed to post tweet');
        }
    } catch (error) {
        console.error('Error posting to Twitter:', error);
        await processingMsg.edit('😅 oops, something went wrong posting to twitter. try again?');
    }
    return;
}
    
    // Handle other bot mentions (both @ mentions and replies)
    if (message.mentions.has(client.user) || message.type === 'REPLY') {
        try {
            const processingMsg = await message.reply("🎵 gimme a min bruh i got sumthin to say...");
            
            const videoId = await getRandomVideo();
            const timestamp = await analyzeVideo(videoId);
            
            if (!timestamp) {
                await processingMsg.edit("😕 sumthin went wrong...derp try again");
                return;
            }
            
            const youtubeUrl = `https://youtube.com/watch?v=${videoId}&t=${timestamp}`;
            
            // We'll let handleGenerateCommand handle the transcription
            await handleGenerateCommand(message, youtubeUrl);
            
            await processingMsg.delete().catch(() => {});
        } catch (error) {
            console.error('Error handling mention:', error);
            message.reply('oops u broke me dog...try again');
        }
    }
});



// Startup handler
client.once('ready', async () => {
    console.log('🤖 Bot is online!');
    console.log('📊 Connected to servers:');
    client.guilds.cache.forEach(guild => {
        console.log(`   - ${guild.name} (${guild.id})`);
    });
    
    // Initialize Twitter handler
    try {
        const success = await twitterHandler.initialize();
        if (success && twitterHandler.isInitialized) {
            console.log('🐦 Twitter handler initialized successfully');
        } else {
            console.error('🐦 Twitter handler initialization failed');
        }
    } catch (error) {
        console.error('Failed to initialize Twitter handler:', error);
    }
});

// Start the bot
client.login(process.env.DISCORD_TOKEN)
    .catch(error => {
        console.error('Failed to start bot:', error);
        process.exit(1);
    });