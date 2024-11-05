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
const MUSICGEN_URL = 'http://localhost:8001';
const CHECK_INTERVAL = 15 * 60 * 1000; // Check every 15 minutes
const REPLIED_IDS_FILE = 'pmarca_replied_tweets.json';
const MAX_TWEETS_TO_CHECK = 10;
const PMARCA_USERNAME = 'pmarca';

class PMarcaHandler {
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

            console.log('PMarca Handler initialized successfully');
            return true;
        } catch (error) {
            console.error('Failed to initialize PMarca Handler:', error);
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
            console.log(`Loaded ${this.repliedIds.size} previously replied PMarca tweets`);
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

    async checkPMarcaTweets() {
    try {
        console.log('Checking PMarca\'s recent tweets...');
        
        const tweetsGenerator = this.scraper.getTweets(PMARCA_USERNAME, MAX_TWEETS_TO_CHECK);
        
        for await (const tweet of tweetsGenerator) {
            if (this.repliedIds.has(tweet.id)) {
                console.log('Already replied to tweet:', tweet.id);
                continue;
            }

            // Get full tweet data immediately
            const fullTweetData = await this.scraper.getTweet(tweet.id);
            
            if (!fullTweetData) {
                console.log('Failed to fetch full tweet data, skipping...');
                continue;
            }

            // Skip pure retweets
            if (tweet.isRetweet && !fullTweetData.quotedStatus) {
                console.log('Skipping pure retweet:', tweet.id);
                continue;
            }

            // Skip replies that aren't quote tweets
            if (tweet.replyTo && !fullTweetData.quotedStatus) {
                console.log('Skipping reply:', tweet.id);
                continue;
            }

            // Extract PMarca's comment and quoted content
            let pmarcaComment = '';
            let quotedContent = null;

            // Get PMarca's comment, filtering out just URLs
            if (fullTweetData.text && !fullTweetData.text.startsWith('https://')) {
                pmarcaComment = fullTweetData.text;
            }

            // Get quoted content if available
            if (fullTweetData.quotedStatus) {
                quotedContent = fullTweetData.quotedStatus.text;
                
                // If PMarca's comment is empty/just a URL, but he included media
                if (!pmarcaComment && fullTweetData.photos?.length > 0) {
                    pmarcaComment = '📸'; // Indicate he shared media
                } else if (!pmarcaComment) {
                    pmarcaComment = '👀'; // Default reaction if no text/media
                }

                console.log('Found quote tweet:', {
                    pmarcaComment,
                    quotedContent,
                    hasMedia: !!fullTweetData.photos?.length,
                    quotedAuthor: fullTweetData.quotedStatus.username
                });
            }

            // Only proceed if we have some content to work with
            if (!quotedContent && !pmarcaComment) {
                console.log('No meaningful content found, skipping...');
                continue;
            }

            // Generate response using our pipeline
            const response = await this.generateResponse(pmarcaComment, quotedContent);
            
            if (response) {
                // Generate video
                const videoResponse = await this.generateVideo(
                    response.videoId,
                    response.timestamp,
                    response.youtubeUrl
                );

                if (videoResponse?.success) {
                    // Post reply using TwitterHandler
                    await this.twitterHandler.postVideoWithText(
                        videoResponse.videoBuffer,
                        response.responseText,
                        tweet.id
                    );
                    
                    console.log('Successfully posted reply video to PMarca tweet:', tweet.id);
                    
                    // Mark as replied
                    this.repliedIds.add(tweet.id);
                    await this.saveRepliedIds();
                    
                    // Add delay between processing tweets
                    await new Promise(resolve => setTimeout(resolve, 30000));
                }
            }
        }
    } catch (error) {
        console.error('Error checking PMarca tweets:', error);
    }
}


    async generateResponse(pmarcaComment, quotedContent = null) {
    try {
        console.log('Generating response for PMarca tweet:', {
            pmarcaComment,
            quotedContent
        });
        
        // Use the PMarca-specific endpoint
        const replyResponse = await fetch(`${FASTAPI_URL}/generate_pmarca_reply`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ 
                tweet_text: pmarcaComment,
                quoted_text: quotedContent,
                test_mode: false
            })
        });
        
        if (!replyResponse.ok) {
            throw new Error(`Failed to generate reply: ${replyResponse.statusText}`);
        }
        
        const replyData = await replyResponse.json();
        console.log('Generated reply data:', JSON.stringify(replyData, null, 2));

        return {
            videoId: replyData.video_id,
            timestamp: replyData.timestamp,
            responseText: replyData.response_text,
            youtubeUrl: replyData.youtube_url
        };

    } catch (error) {
        console.error('Error generating response:', error);
        return null;
    }
}

    // Reuse existing video generation methods
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

    async generateVideo(videoId, timestamp, youtubeUrl) {
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

            // Step 3: Generate music with more intense settings for PMarca
            const musicResponse = await fetch(`${MUSICGEN_URL}/generate`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    url: youtubeUrl,
                    currentTime: timestamp,
                    model: 'thepatch/vanya_ai_dnb_0.1',
                    promptLength: '8',  // Longer prompt for more context
                    duration: '30-32'   // Slightly longer duration
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
                effectIntensity: 0.8,  // More intense effects for PMarca
                transitionDuration: 2.5 // Slightly longer transitions
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
        console.log('Starting PMarca Handler...');
        
        // Initial check
        await this.checkPMarcaTweets();
        
        // Set up periodic checking
        setInterval(() => this.checkPMarcaTweets(), CHECK_INTERVAL);
    }
}

// Start the handler
const handler = new PMarcaHandler();
handler.initialize().then(() => {
    handler.start();
});