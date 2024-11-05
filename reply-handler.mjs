import { Scraper } from './twitter-client/dist/node/esm/index.mjs';
import fetch from 'node-fetch';
import fs from 'fs/promises';
import 'dotenv/config';
import TwitterHandler from './twitter-handler.mjs';
import VideoProcessor from './videoProcessor.mjs';
import GlitchProcessor from './glitchProcessor.mjs';
import ffmpeg from 'fluent-ffmpeg';
import { PassThrough } from 'stream';

// Constants
const FASTAPI_URL = 'http://0.0.0.0:3000';
const MUSICGEN_URL = 'http://localhost:8001'
const CHECK_INTERVAL = 20 * 60 * 1000; // Check every 5 minutes
const REPLIED_IDS_FILE = 'replied_tweets.json';
const MAX_TWEETS_TO_CHECK = 20;

class TwitterReplyHandler {
    constructor() {
        this.scraper = null;
        this.twitterHandler = new TwitterHandler();
        this.repliedIds = new Set();
        this.username = process.env.TWITTER_USERNAME;
        this.videoProcessor = new VideoProcessor();
        this.glitchProcessor = new GlitchProcessor();
    }

    async initialize() {
        try {
            // Initialize the tweet poster
            await this.twitterHandler.initialize();
            
            // Initialize the scraper
            this.scraper = await this.initializeScraper();
            if (!this.scraper) {
                throw new Error('Failed to initialize scraper');
            }

            // Load previously replied tweets
            await this.loadRepliedIds();

            console.log('Twitter Reply Handler initialized successfully');
            return true;
        } catch (error) {
            console.error('Failed to initialize Twitter Reply Handler:', error);
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

    async loadRepliedIds() {
        try {
            const data = await fs.readFile(REPLIED_IDS_FILE, 'utf-8');
            const ids = JSON.parse(data);
            this.repliedIds = new Set(ids);
            console.log(`Loaded ${this.repliedIds.size} previously replied tweets`);
        } catch (error) {
            this.repliedIds = new Set();
            await this.saveRepliedIds();
        }
    }

    async saveRepliedIds() {
        await fs.writeFile(
            REPLIED_IDS_FILE,
            JSON.stringify([...this.repliedIds])
        );
    }

    async handleNewReplies() {
        try {
            console.log('Checking for new replies...');
            
            const tweetsGenerator = this.scraper.getTweets(this.username, MAX_TWEETS_TO_CHECK);
            
            for await (const tweet of tweetsGenerator) {
                console.log('\nChecking our tweet:', {
                    id: tweet.id,
                    text: tweet.text?.substring(0, 50) + '...'
                });
                
                try {
                    const tweetData = await this.scraper.getTweet(tweet.id);
                    
                    if (tweetData && tweetData.conversationId && tweetData.replies > 0) {
                        console.log(`Found ${tweetData.replies} replies to tweet ${tweet.id}`);
                        
                        const repliesGenerator = this.scraper.searchTweets(
                            `conversation_id:${tweetData.conversationId}`,
                            10
                        );

                        for await (const reply of repliesGenerator) {
                            if (reply.username === this.username || this.repliedIds.has(reply.id)) {
                                console.log('Skipping reply:', reply.id);
                                continue;
                            }

                            console.log('Processing new reply:', {
                                id: reply.id,
                                text: reply.text,
                                from: reply.username
                            });

                            // Generate response using our pipeline
                            const response = await this.generateResponse(reply.text);
                            
                            if (response) {
                                // Generate video
                                
                                const videoResponse = await this.generateVideo(
                                    response.videoId,
                                    response.timestamp,
                                    response.youtubeUrl  // Add this parameter
                                );

                                if (videoResponse?.success) {
                                    // Post reply using TwitterHandler
                                    await this.twitterHandler.postVideoWithText(
                                        videoResponse.videoBuffer,
                                        response.responseText,
                                        reply.id
                                    );
                                    
                                    console.log('Successfully posted reply video to:', reply.id);
                                    
                                    // Mark as replied
                                    this.repliedIds.add(reply.id);
                                    await this.saveRepliedIds();
                                    
                                    // Add delay between processing replies
                                    await new Promise(resolve => setTimeout(resolve, 5000));
                                }
                            }
                        }
                    }
                } catch (error) {
                    console.error(`Error processing tweet ${tweet.id}:`, error);
                    continue;
                }
            }
        } catch (error) {
            console.error('Error handling replies:', error);
        }
    }

    async generateResponse(tweetText) {
        try {
            console.log('Generating response for tweet:', tweetText);
            
            // Step 1: Get initial reply and clip info
            console.log('🎵 Finding relevant Marc clip...');
            const replyResponse = await fetch(`${FASTAPI_URL}/generate_reply`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ 
                    tweet_text: tweetText,
                    test_mode: false
                })
            });
            
            if (!replyResponse.ok) {
                throw new Error(`Failed to generate reply: ${replyResponse.statusText}`);
            }
            
            const replyData = await replyResponse.json();
            console.log('Generated reply data:', JSON.stringify(replyData, null, 2));

            // Return the essential data needed for video generation
            return {
                videoId: replyData.video_id,
                timestamp: replyData.timestamp,
                responseText: replyData.response_text,
                youtubeUrl: replyData.youtube_url  // Add this field
            };

        } catch (error) {
            console.error('Error generating response:', error);
            return null;
        }
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

    async pollForTaskCompletion(taskId) {
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

    async generateVideo(videoId, timestamp, youtubeUrl) {  // Add youtubeUrl parameter
    try {
        // Step 1: Get video clip and frame
        const clipResponse = await fetch(`${FASTAPI_URL}/get_clip`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ video_id: videoId, timestamp })
        });

        if (!clipResponse.ok) {
            throw new Error('Failed to get video clip');
        }

        const clipBuffer = Buffer.from(await clipResponse.arrayBuffer());
        console.log('Successfully got video clip');

        // Step 2: Get frame for the end of the clip
        const frameResponse = await fetch(`${FASTAPI_URL}/get_frame`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ 
                video_id: videoId, 
                timestamp: timestamp + 5 
            })
        });

        if (!frameResponse.ok) {
            throw new Error('Failed to get video frame');
        }

        // Step 3: Get music
        const musicResponse = await fetch(`${MUSICGEN_URL}/generate`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                url: youtubeUrl,  // Use the YouTube URL here
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
        console.log('Music generated successfully');

        // Step 4: Create final video with VideoProcessor
        const baseVideoBuffer = await this.videoProcessor.createVideoWithAudio(clipBuffer, mp3Buffer, {
            startWithVideo: true,
            effectIntensity: 0.6,
            transitionDuration: 2
        });
        
        // Step 5: Add glitch effects
        const finalVideoBuffer = await this.glitchProcessor.processVideoBuffer(baseVideoBuffer);
        
        return { success: true, videoBuffer: finalVideoBuffer };
    } catch (error) {
        console.error('Error generating video:', error);
        return { success: false };
    }
}

    async start() {
        console.log('Starting Twitter Reply Handler...');
        
        // Initial check
        await this.handleNewReplies();
        
        // Set up periodic checking
        setInterval(() => this.handleNewRplies(), CHECK_INTERVAL);
    }
}

// Start the handler
const handler = new TwitterReplyHandler();
handler.initialize().then(() => {
    handler.start();
});