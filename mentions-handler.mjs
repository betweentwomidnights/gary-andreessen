import { Scraper } from './twitter-client/dist/node/esm/index.mjs';
import fetch from 'node-fetch';
import fs from 'fs/promises';
import 'dotenv/config';
import TwitterHandler from './twitter-handler.mjs';
import VideoProcessor from './videoProcessor.mjs';
import GlitchProcessor from './glitchProcessor.mjs';
import ffmpeg from 'fluent-ffmpeg';
import { PassThrough } from 'stream';
import GenerationStore from './generationStore.mjs';

// Constants
const FASTAPI_URL = 'http://0.0.0.0:3000';
const MUSICGEN_URL = 'http://localhost:8001';
const CHECK_INTERVAL = 600 * 1000; // Check every minute while testing
const PROCESSED_MENTIONS_FILE = 'processed_mentions.json';
const MAX_MENTIONS_TO_CHECK = 100;

class MentionsHandler {
    constructor() {
        this.scraper = null;
        this.twitterHandler = new TwitterHandler();
        this.processedMentions = new Set();
        this.username = process.env.TWITTER_USERNAME;
        this.videoProcessor = new VideoProcessor();
        this.glitchProcessor = new GlitchProcessor();
        this.generationStore = new GenerationStore();
    }

    async initialize() {
        try {
            console.log('Initializing YouTube Mentions Handler...');
            
            // Initialize Twitter components
            await this.twitterHandler.initialize();
            this.scraper = await this.initializeScraper();
            
            if (!this.scraper) {
                throw new Error('Failed to initialize scraper');
            }

            // Initialize generation store
            const storeInitialized = await this.generationStore.initialize();
            if (!storeInitialized) {
                throw new Error('Failed to initialize generation store');
            }

            // Load previously processed mentions
            await this.loadProcessedMentions();

            // Set up periodic cleanup of old generations
            setInterval(() => this.generationStore.cleanup(), 3600000); // Clean every hour

            console.log('YouTube Mentions Handler initialized successfully');
            return true;
        } catch (error) {
            console.error('Failed to initialize YouTube Mentions Handler:', error);
            return false;
        }
    }

    async initializeScraper() {
        const scraper = new Scraper();
        
        try {
            const cookiesJson = await fs.readFile('cookies.json', 'utf-8');
            const cookies = JSON.parse(cookiesJson);
            await scraper.setCookies(cookies);
            return scraper;
        } catch (error) {
            console.log('Existing cookies failed, performing fresh login');
            
            try {
                await scraper.login(
                    process.env.TWITTER_USERNAME,
                    process.env.TWITTER_PASSWORD
                );
                
                const cookies = await scraper.getCookies();
                await this.saveCookies(cookies);
                return scraper;
            } catch (loginError) {
                console.error('Login failed:', loginError);
                return null;
            }
        }
    }

    async saveCookies(cookies) {
        const cookiesArray = cookies.map(cookie => ({
            key: cookie.key,
            value: cookie.value,
            domain: cookie.domain,
            path: cookie.path
        }));
        
        await fs.writeFile('cookies.json', JSON.stringify(cookiesArray, null, 2));
    }

    async loadProcessedMentions() {
        try {
            const data = await fs.readFile(PROCESSED_MENTIONS_FILE, 'utf-8');
            this.processedMentions = new Set(JSON.parse(data));
            console.log(`Loaded ${this.processedMentions.size} previously processed mentions`);
        } catch (error) {
            console.log('No existing processed mentions file, starting fresh');
            this.processedMentions = new Set();
            await this.saveProcessedMentions();
        }
    }

    async saveProcessedMentions() {
        await fs.writeFile(
            PROCESSED_MENTIONS_FILE,
            JSON.stringify([...this.processedMentions])
        );
    }

    extractYoutubeInfo(text) {
        try {
            // Regular expressions for both youtube.com and youtu.be URLs
            const youtubeRegex = /(?:https?:\/\/)?(?:www\.)?(?:youtube\.com\/watch\?v=|youtu\.be\/)([^&\s]+)(?:&t=(\d+)|[\s])?/;
            const match = text.match(youtubeRegex);
            
            if (!match) return null;
            
            const videoId = match[1];
            let timestamp = 0;
            
            // Check for timestamp in various formats
            const timePatterns = [
                /[?&]t=(\d+)/,           // t=123
                /[?&]start=(\d+)/,       // start=123
                /\?t=(\d+)s/,            // t=123s
                /\?start=(\d+)s/         // start=123s
            ];

            for (const pattern of timePatterns) {
                const timeMatch = text.match(pattern);
                if (timeMatch) {
                    timestamp = parseInt(timeMatch[1], 10);
                    break;
                }
            }
            
            return {
                videoId,
                timestamp,
                url: `https://youtube.com/watch?v=${videoId}`
            };
        } catch (error) {
            console.error('Error extracting YouTube info:', error);
            return null;
        }
    }

    async pollForTaskCompletion(taskId) {
        const maxTime = 1200000; // 20 minutes
        const interval = 10000; // 10 seconds
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

    async convertToMP3(inputBuffer) {
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
    }

    async generateVideo(videoId, timestamp, youtubeUrl) {
    try {
        console.log(`Generating video for YouTube ID: ${videoId} at timestamp: ${timestamp}`);
        
        // Get video clip
        const clipResponse = await fetch(`${FASTAPI_URL}/get_clip`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ video_id: videoId, timestamp })
        });

        if (!clipResponse.ok) {
            throw new Error('Failed to get video clip');
        }

        const clipBuffer = Buffer.from(await clipResponse.arrayBuffer());

        // Generate music
        console.log('Starting music generation...');
        const musicResponse = await fetch(`${MUSICGEN_URL}/generate`, {
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

        if (!musicResponse.ok) {
            throw new Error('Failed to start music generation');
        }

        const { task_id } = await musicResponse.json();
        console.log('Waiting for music generation...');
        
        const audioData = await this.pollForTaskCompletion(task_id);
        const audioBuffer = Buffer.from(audioData, 'base64');
        const mp3Buffer = await this.convertToMP3(audioBuffer);

        // Create final video
        const baseVideoBuffer = await this.videoProcessor.createVideoWithAudio(
            clipBuffer,
            mp3Buffer,
            {
                startWithVideo: true,
                effectIntensity: 0.6,
                transitionDuration: 2
            }
        );
        
        // Add glitch effects
        const finalVideoBuffer = await this.glitchProcessor.processVideoBuffer(baseVideoBuffer);
        
        return { 
            success: true, 
            videoBuffer: finalVideoBuffer,
            taskId: task_id,
            audioData: audioData
        };
    } catch (error) {
        console.error('Error generating video:', error);
        return { success: false };
    }
}

    async handleContinueGeneration(mention, replyToTweetId) {
        try {
            console.log(`Handling continue request for tweet: ${replyToTweetId}`);
            
            // Get the previous generation
            const previousGen = await this.generationStore.getGeneration(replyToTweetId);
            if (!previousGen) {
                console.log('No previous generation found');
                return false;
            }

            // Generate continuation audio
            const response = await fetch(`${MUSICGEN_URL}/continue`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    task_id: previousGen.task_id,
                    model: 'thepatch/vanya_ai_dnb_0.1',
                    prompt_duration: 6,
                    audio: previousGen.audio_data
                })
            });

            if (!response.ok) {
                throw new Error('Failed to continue generation');
            }

            const { task_id } = await response.json();
            
            // Wait for audio generation
            const audioData = await this.pollForTaskCompletion(task_id);
            const audioBuffer = Buffer.from(audioData, 'base64');
            const mp3Buffer = await this.convertToMP3(audioBuffer);

            // Get the original video clip
            const clipResponse = await fetch(`${FASTAPI_URL}/get_clip`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ 
                    video_id: previousGen.video_id, 
                    timestamp: previousGen.timestamp 
                })
            });

            if (!clipResponse.ok) {
                throw new Error('Failed to get original video clip');
            }

            const clipBuffer = Buffer.from(await clipResponse.arrayBuffer());
            
            // Create base continuation video
            const baseVideoBuffer = await this.videoProcessor.createContinuationVideo(
                clipBuffer,
                mp3Buffer
            );
            
            // Apply glitch effects
            const finalVideoBuffer = await this.glitchProcessor.processVideoBuffer(baseVideoBuffer);

            // Post the continuation
            const result = await this.twitterHandler.postVideoWithText(
                finalVideoBuffer,
                'heres that continued beat u asked for 🎵',
                mention.id
            );

            if (result.success) {
                // Store the continuation data
                await this.generationStore.saveGeneration(result.tweetId, {
                    videoId: previousGen.video_id,
                    timestamp: previousGen.timestamp,
                    taskId: task_id,
                    audioData: audioData,
                    videoBuffer: finalVideoBuffer
                });
            }

            return result.success;
        } catch (error) {
            console.error('Error handling continue generation:', error);
            return false;
        }
    }

    async findBotTweetInThread(mention) {
    try {
        console.log('🔍 Tracing conversation thread for mention:', mention.id);
        
        // Get conversation thread
        const conversation = await this.scraper.getTweet(mention.id);
        console.log('Found conversation:', conversation);
        
        // The tweet object should contain references to the conversation
        // Look for the most recent tweet from the bot in the thread
        if (conversation.referenced_tweets) {
            console.log('Referenced tweets:', conversation.referenced_tweets);
            
            // Find the most recent tweet from the bot in the thread
            for (const ref of conversation.referenced_tweets) {
                const referencedTweet = await this.scraper.getTweet(ref.id);
                if (referencedTweet.author_id === this.username) {
                    console.log('✅ Found bot tweet in thread:', referencedTweet.id);
                    return referencedTweet.id;
                }
            }
        }
        
        console.log('❌ No bot tweet found in thread');
        return null;
    } catch (error) {
        console.error('Error tracing conversation:', error);
        return null;
    }
}

    async handleMentions() {
    try {
        console.log('\n🔍 Checking for YouTube URL mentions...');
        
        const searchQuery = `@${this.username}`;
        console.log('📊 Search query:', searchQuery);
        
        let mentionCount = 0;
        const mentionsGenerator = this.scraper.searchTweets(
            searchQuery,
            MAX_MENTIONS_TO_CHECK
        );
        
        for await (const mention of mentionsGenerator) {
            mentionCount++;
            console.log('\n📌 Found mention:', {
                id: mention.id,
                text: mention.text?.substring(0, 100) + '...',
                from: mention.username,
                inReplyToStatusId: mention.inReplyToStatusId
            });

            if (mention.username === this.username || this.processedMentions.has(mention.id)) {
                console.log('⏭️ Skipping mention (already processed or self-mention)');
                continue;
            }

            // Check if this is a continuation request
            if (mention.text.toLowerCase().includes('continue')) {
                console.log('🔄 Detected continue request');
                
                // Get the full conversation data for this tweet
                const conversation = await this.scraper.getTweet(mention.id);
                console.log('Found conversation data for continue request');

                if (!conversation?.inReplyToStatus) {
                    console.log('❌ No reply status found in conversation');
                    continue;
                }

                // Get the tweet this is replying to
                const replyToTweet = conversation.inReplyToStatus;
                console.log('Replying to tweet from:', replyToTweet.username);

                // Check if the reply is to our bot
                if (replyToTweet.username !== this.username) {
                    console.log('❌ Continue reply not to bot tweet:', replyToTweet.username);
                    continue;
                }

                const botTweetId = replyToTweet.id;
                console.log(`✨ Found bot tweet to continue from: ${botTweetId}`);

                // Look up the generation in our store
                const previousGen = await this.generationStore.getGeneration(botTweetId);
                if (!previousGen) {
                    console.log('❌ No stored generation found for tweet:', botTweetId);
                    
                    // Optionally reply to let the user know
                    await this.twitterHandler.postVideoWithText(
                        null,
                        "sorry, i can't find the original beat to continue from! might be too old or something went wrong 🎵",
                        mention.id
                    );
                    
                    continue;
                }

                console.log('✅ Found previous generation data:', {
                    videoId: previousGen.video_id,
                    timestamp: previousGen.timestamp,
                    taskId: previousGen.task_id
                });

                const success = await this.handleContinueGeneration(
                    mention,
                    botTweetId
                );
                
                if (success) {
                    this.processedMentions.add(mention.id);
                    await this.saveProcessedMentions();
                    console.log('✅ Successfully processed continue request');
                } else {
                    console.log('❌ Failed to process continue request');
                    
                    // Optionally notify the user of the failure
                    await this.twitterHandler.postVideoWithText(
                        null,
                        "sorry, something went wrong while trying to continue the beat 😅 try again?",
                        mention.id
                    );
                }
                
                continue;
            }

            // If not a continue request, look for YouTube URLs
            console.log('🔍 Looking for YouTube URL in mention');
            let youtubeInfo = null;
            
            if (mention.urls && mention.urls.length > 0) {
                for (const url of mention.urls) {
                    console.log('Checking URL:', url);
                    youtubeInfo = this.extractYoutubeInfo(url);
                    if (youtubeInfo) break;
                }
            }

            if (!youtubeInfo) {
                youtubeInfo = this.extractYoutubeInfo(mention.text);
            }

            if (!youtubeInfo) {
                console.log('❌ No valid YouTube URL found in mention');
                continue;
            }

                const videoResponse = await this.generateVideo(
                    youtubeInfo.videoId,
                    youtubeInfo.timestamp,
                    youtubeInfo.url
                );

                if (videoResponse?.success) {
                    const result = await this.twitterHandler.postVideoWithText(
                        videoResponse.videoBuffer,
                        'thats dope but this is cooler',
                        mention.id
                    );
                    
                    if (result.success) {
                        // Store the generation data
                        await this.generationStore.saveGeneration(result.tweetId, {
                            videoId: youtubeInfo.videoId,
                            timestamp: youtubeInfo.timestamp,
                            taskId: videoResponse.taskId,
                            audioData: videoResponse.audioData,
                            videoBuffer: videoResponse.videoBuffer
                        });
                        
                        this.processedMentions.add(mention.id);
                        await this.saveProcessedMentions();
                    }
                    
                    await new Promise(resolve => setTimeout(resolve, 5000));
                }
            }
            
            console.log(`\n📊 Search complete - Found ${mentionCount} total mentions`);
            
        } catch (error) {
            console.error('❌ Error handling mentions:', error);
            console.error('Error details:', {
                name: error.name,
                message: error.message,
                stack: error.stack
            });
        }
    }

// Update extractYoutubeInfo to handle the URL variants we're seeing
extractYoutubeInfo(text) {
    try {
        console.log('Extracting YouTube info from:', text);
        
        // Regular expressions for both youtube.com and youtu.be URLs with various parameter formats
        const youtubeRegex = /(?:https?:\/\/)?(?:www\.)?(?:youtube\.com\/watch\?v=|youtu\.be\/)([a-zA-Z0-9_-]+)(?:[^\s]*)?/;
        const match = text.match(youtubeRegex);
        
        if (!match) {
            console.log('No YouTube URL pattern match found');
            return null;
        }
        
        const videoId = match[1];
        let timestamp = 0;
        
        // Check for timestamp in various formats
        const timePatterns = [
            /[?&]t=(\d+)/,           // t=123
            /[?&]start=(\d+)/,       // start=123
            /\?t=(\d+)s/,            // t=123s
            /\?start=(\d+)s/,        // start=123s
            /&t=(\d+)s?/             // &t=123 or &t=123s
        ];

        for (const pattern of timePatterns) {
            const timeMatch = text.match(pattern);
            if (timeMatch) {
                timestamp = parseInt(timeMatch[1], 10);
                console.log('Found timestamp:', timestamp);
                break;
            }
        }
        
        const result = {
            videoId,
            timestamp,
            url: `https://youtube.com/watch?v=${videoId}`
        };
        
        console.log('Extracted YouTube info:', result);
        return result;
        
    } catch (error) {
        console.error('Error extracting YouTube info:', error);
        return null;
    }
}

// Make sure to close the store when shutting down
    async cleanup() {
        await this.generationStore.close();
    }



    async start() {
        console.log('🚀 Starting YouTube Mentions Handler...');
        
        // Initial check
        await this.handleMentions();
        
        // Set up periodic checking
        setInterval(() => this.handleMentions(), CHECK_INTERVAL);
        
        console.log(`✨ Monitoring mentions every ${CHECK_INTERVAL / 1000} seconds...`);
    }
}

const handler = new MentionsHandler();
handler.initialize().then(() => {
    handler.start();
});

// Handle cleanup on shutdown
process.on('SIGINT', async () => {
    console.log('Shutting down...');
    await handler.cleanup();
    process.exit(0);
});