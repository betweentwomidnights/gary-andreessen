import FormData from 'form-data';
import crypto from 'crypto';
import { promisify } from 'util';
const sleep = promisify(setTimeout);

class TwitterHandler {
    constructor() {
        this.MEDIA_ENDPOINT_URL = 'https://upload.twitter.com/1.1/media/upload.json';
        this.POST_TWEET_URL = 'https://api.twitter.com/2/tweets';  // Changed to v2 endpoint
        this.isInitialized = false;
        this.mediaId = null;
        this.processingInfo = null;
        this.bearerToken = process.env.TWITTER_BEARER_TOKEN;
    }

    async initialize() {
        try {
            console.log('Initializing Twitter handler...');
            // Verify credentials are present
            const required = [
                'TWITTER_API_KEY',
                'TWITTER_API_SECRET',
                'TWITTER_ACCESS_TOKEN',
                'TWITTER_ACCESS_TOKEN_SECRET'
            ];
            
            for (const key of required) {
                if (!process.env[key]) {
                    throw new Error(`Missing required environment variable: ${key}`);
                }
            }
            
            this.isInitialized = true;
            return true;
        } catch (error) {
            this.isInitialized = false;
            console.error('Error in Twitter handler initialization:', error);
            throw error;
        }
    }

    generateOAuthHeaders(method, url, params = {}) {
        const oauthNonce = crypto.randomBytes(32).toString('hex');
        const timestamp = Math.floor(Date.now() / 1000).toString();

        const oauthParams = {
            oauth_consumer_key: process.env.TWITTER_API_KEY,
            oauth_token: process.env.TWITTER_ACCESS_TOKEN,
            oauth_signature_method: 'HMAC-SHA1',
            oauth_timestamp: timestamp,
            oauth_nonce: oauthNonce,
            oauth_version: '1.0'
        };

        const allParams = { ...params, ...oauthParams };
        const paramString = Object.keys(allParams)
            .sort()
            .map(key => `${encodeURIComponent(key)}=${encodeURIComponent(allParams[key])}`)
            .join('&');

        const sigBaseStr = [
            method,
            encodeURIComponent(url),
            encodeURIComponent(paramString)
        ].join('&');

        const signingKey = `${encodeURIComponent(process.env.TWITTER_API_SECRET)}&${encodeURIComponent(process.env.TWITTER_ACCESS_TOKEN_SECRET)}`;
        const signature = crypto.createHmac('sha1', signingKey)
            .update(sigBaseStr)
            .digest('base64');

        return 'OAuth ' + Object.entries({
            ...oauthParams,
            oauth_signature: signature
        })
        .map(([key, value]) => `${key}="${encodeURIComponent(value)}"`)
        .join(', ');
    }

    async uploadInit(fileSize) {
        console.log('Starting INIT phase...');
        const params = {
            command: 'INIT',
            total_bytes: fileSize,
            media_type: 'video/mp4',
            media_category: 'tweet_video'
        };

        const response = await fetch(this.MEDIA_ENDPOINT_URL, {
            method: 'POST',
            headers: {
                'Authorization': this.generateOAuthHeaders('POST', this.MEDIA_ENDPOINT_URL, params),
                'Content-Type': 'application/x-www-form-urlencoded'
            },
            body: new URLSearchParams(params)
        });

        const responseText = await response.text();
        console.log('INIT Response:', responseText);

        if (response.status !== 202) {
            throw new Error(`INIT failed: ${responseText}`);
        }

        const data = JSON.parse(responseText);
        this.mediaId = data.media_id_string;
        return data;
    }

    async uploadAppend(chunk, segmentIndex) {
        console.log(`Uploading chunk ${segmentIndex + 1}...`);
        
        // Create form data using URLSearchParams for the non-media parts
        const params = new URLSearchParams();
        params.append('command', 'APPEND');
        params.append('media_id', this.mediaId);
        params.append('segment_index', segmentIndex);
        
        // Create a boundary for multipart form data
        const boundary = '----WebKitFormBoundary' + crypto.randomBytes(16).toString('hex');
        
        // Construct the multipart form data manually
        const formDataContent = Buffer.concat([
            // Command parameter
            Buffer.from(`--${boundary}\r\n`),
            Buffer.from('Content-Disposition: form-data; name="command"\r\n\r\n'),
            Buffer.from('APPEND\r\n'),
            
            // Media ID parameter
            Buffer.from(`--${boundary}\r\n`),
            Buffer.from('Content-Disposition: form-data; name="media_id"\r\n\r\n'),
            Buffer.from(`${this.mediaId}\r\n`),
            
            // Segment index parameter
            Buffer.from(`--${boundary}\r\n`),
            Buffer.from('Content-Disposition: form-data; name="segment_index"\r\n\r\n'),
            Buffer.from(`${segmentIndex}\r\n`),
            
            // Media content
            Buffer.from(`--${boundary}\r\n`),
            Buffer.from('Content-Disposition: form-data; name="media"; filename="chunk.mp4"\r\n'),
            Buffer.from('Content-Type: video/mp4\r\n\r\n'),
            chunk,
            Buffer.from('\r\n'),
            
            // End boundary
            Buffer.from(`--${boundary}--\r\n`)
        ]);

        const response = await fetch(this.MEDIA_ENDPOINT_URL, {
            method: 'POST',
            headers: {
                'Authorization': this.generateOAuthHeaders('POST', this.MEDIA_ENDPOINT_URL),
                'Content-Type': `multipart/form-data; boundary=${boundary}`,
                'Content-Length': formDataContent.length.toString()
            },
            body: formDataContent
        });

        const responseText = await response.text();
        console.log('APPEND Response:', responseText);

        if (response.status !== 204) {
            const error = response.status === 200 ? responseText : `Status ${response.status}: ${responseText}`;
            throw new Error(`APPEND failed: ${error}`);
        }
    }

    async uploadFinalize() {
        console.log('Starting FINALIZE phase...');
        const params = {
            command: 'FINALIZE',
            media_id: this.mediaId
        };

        const response = await fetch(this.MEDIA_ENDPOINT_URL, {
            method: 'POST',
            headers: {
                'Authorization': this.generateOAuthHeaders('POST', this.MEDIA_ENDPOINT_URL, params),
                'Content-Type': 'application/x-www-form-urlencoded'
            },
            body: new URLSearchParams(params)
        });

        const responseText = await response.text();
        console.log('FINALIZE Response:', responseText);

        if (response.status !== 200) {
            throw new Error(`FINALIZE failed: ${responseText}`);
        }

        const data = JSON.parse(responseText);
        this.processingInfo = data.processing_info;
        return data;
    }

    async checkStatus() {
        if (!this.processingInfo) return;

        const state = this.processingInfo.state;
        console.log('Media processing status:', state);

        if (state === 'succeeded') return;
        if (state === 'failed') {
            throw new Error('Video processing failed');
        }

        const checkAfterSecs = this.processingInfo.check_after_secs || 1;
        console.log(`Checking again after ${checkAfterSecs} seconds...`);
        await sleep(checkAfterSecs * 1000);

        const params = {
            command: 'STATUS',
            media_id: this.mediaId
        };

        const response = await fetch(`${this.MEDIA_ENDPOINT_URL}?${new URLSearchParams(params)}`, {
            method: 'GET',
            headers: {
                'Authorization': this.generateOAuthHeaders('GET', this.MEDIA_ENDPOINT_URL, params)
            }
        });

        if (!response.ok) {
            const error = await response.text();
            throw new Error(`STATUS check failed: ${error}`);
        }

        const data = await response.json();
        console.log('Status response:', data);
        this.processingInfo = data.processing_info;
        
        // Recursive call to keep checking
        return this.checkStatus();
    }

    async uploadMediaToTwitter(videoBuffer) {
        try {
            const CHUNK_SIZE = 4 * 1024 * 1024; // 4MB chunks
            
            // Initialize upload
            await this.uploadInit(videoBuffer.length);
            console.log('Got media_id:', this.mediaId);

            // Upload chunks
            let segment = 0;
            for (let i = 0; i < videoBuffer.length; i += CHUNK_SIZE) {
                const chunk = videoBuffer.slice(i, Math.min(i + CHUNK_SIZE, videoBuffer.length));
                console.log(`Preparing chunk ${segment + 1}/${Math.ceil(videoBuffer.length / CHUNK_SIZE)}`);
                console.log(`Chunk size: ${chunk.length} bytes`);
                await this.uploadAppend(chunk, segment);
                segment++;
            }

            // Finalize upload
            const finalizeResponse = await this.uploadFinalize();
            console.log('Upload finalized');

            // Check processing status
            await this.checkStatus();
            console.log('Video processing completed');

            return this.mediaId;
        } catch (error) {
            console.error('Error in media upload:', error);
            throw error;
        }
    }

    async generateV2OAuthHeaders(method, url, params = {}) {
        const oauthNonce = crypto.randomBytes(32).toString('hex');
        const timestamp = Math.floor(Date.now() / 1000).toString();

        const oauthParams = {
            oauth_consumer_key: process.env.TWITTER_API_KEY,
            oauth_token: process.env.TWITTER_ACCESS_TOKEN,
            oauth_signature_method: 'HMAC-SHA1',
            oauth_timestamp: timestamp,
            oauth_nonce: oauthNonce,
            oauth_version: '1.0'
        };

        // For v2, we need to include the JSON body in the signature
        const signingParams = { ...params };
        const bodyParams = params.json ? JSON.parse(params.json) : {};
        Object.assign(signingParams, bodyParams);
        delete signingParams.json;

        const allParams = { ...signingParams, ...oauthParams };
        const paramString = Object.keys(allParams)
            .sort()
            .map(key => `${encodeURIComponent(key)}=${encodeURIComponent(allParams[key])}`)
            .join('&');

        const sigBaseStr = [
            method,
            encodeURIComponent(url),
            encodeURIComponent(paramString)
        ].join('&');

        const signingKey = `${encodeURIComponent(process.env.TWITTER_API_SECRET)}&${encodeURIComponent(process.env.TWITTER_ACCESS_TOKEN_SECRET)}`;
        const signature = crypto.createHmac('sha1', signingKey)
            .update(sigBaseStr)
            .digest('base64');

        return 'OAuth ' + Object.entries({
            ...oauthParams,
            oauth_signature: signature
        })
        .map(([key, value]) => `${key}="${encodeURIComponent(value)}"`)
        .join(', ');
    }

    async tweet(text, mediaId) {
        console.log('Posting tweet using v2 API with OAuth 1.0a...');
        
        const tweetData = {
            text: text,
            media: {
                media_ids: [mediaId]
            }
        };

        console.log('Tweet payload:', JSON.stringify(tweetData, null, 2));

        // Convert the JSON body to a string for OAuth signing
        const jsonBody = JSON.stringify(tweetData);
        
        // Generate OAuth 1.0a headers
        const authHeader = this.generateOAuthHeaders('POST', this.POST_TWEET_URL);

        const headers = {
            'Authorization': authHeader,
            'Content-Type': 'application/json',
            'Accept': 'application/json'
        };

        console.log('Using headers:', {
            ...headers,
            'Authorization': authHeader.substring(0, 50) + '...' // Truncate for logging
        });

        const response = await fetch(this.POST_TWEET_URL, {
            method: 'POST',
            headers: headers,
            body: jsonBody
        });

        const responseText = await response.text();
        console.log('Tweet Response:', responseText);

        if (!response.ok) {
            throw new Error(`Tweet failed: ${responseText}`);
        }

        const responseData = JSON.parse(responseText);
        
        // Handle v2 response format
        return {
            id_str: responseData.data.id,
            text: responseData.data.text
        };
    }

    async postVideoWithText(videoBuffer, text) {
        if (!this.isInitialized) {
            await this.initialize();
        }

        try {
            const formattedText = text ? `"${text}" @pmarca @ai16z` : "got sumthin to say @pmarca @ai16z";
            
            // First upload and process the video (using v1.1 API)
            const mediaId = await this.uploadMediaToTwitter(videoBuffer);
            console.log('Video uploaded successfully, media_id:', mediaId);
            
            // Then post the tweet with the processed video (using v2 API)
            const tweetResult = await this.tweet(formattedText, mediaId);

            return {
                success: true,
                tweetId: tweetResult.id_str,
                text: formattedText
            };
        } catch (error) {
            console.error('Error posting to Twitter:', error);
            throw error;
        }
    }
}

export default TwitterHandler;