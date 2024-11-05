import { Scraper } from './twitter-client/dist/node/esm/index.mjs';
import fs from 'fs/promises';
import 'dotenv/config';

async function initializeScraper() {
    const scraper = new Scraper();
    
    try {
        // Try existing cookies first
        const cookiesJson = await fs.readFile('cookies.json', 'utf-8');
        const cookies = JSON.parse(cookiesJson);
        await scraper.setCookies(cookies);
        return scraper;
    } catch (error) {
        // If cookies failed, do fresh login
        console.log('Existing cookies failed, performing fresh login');
        
        await scraper.login(
            process.env.TWITTER_USERNAME,
            process.env.TWITTER_PASSWORD
        );
        
        // Save new cookies
        const cookies = await scraper.getCookies();
        await saveCookies(cookies);
        return scraper;
    }
}

async function saveCookies(cookies) {
    const cookiesArray = cookies.map(cookie => ({
        key: cookie.key,
        value: cookie.value,
        domain: cookie.domain,
        path: cookie.path
    }));
    
    await fs.writeFile('cookies.json', JSON.stringify(cookiesArray, null, 2));
}

async function testReplyMethods() {
    const tweetId = '1851850100290576540'; // The tweet with 4 replies
    const scraper = await initializeScraper();
    
    console.log('\nTesting different methods to find replies...');

    // Method 1: Direct getTweet
    try {
        console.log('\nMethod 1: getTweet');
        const tweet = await scraper.getTweet(tweetId);
        console.log('Tweet data:', JSON.stringify(tweet, null, 2));
    } catch (error) {
        console.error('Error with getTweet:', error);
    }

    // Method 2: Search for replies using conversation_id
    try {
        console.log('\nMethod 2: Search for replies');
        const tweet = await scraper.getTweet(tweetId);
        if (tweet.conversationId) {
            const replies = await scraper.searchTweets(
                `conversation_id:${tweet.conversationId}`,
                10
            );
            for await (const reply of replies) {
                console.log('Reply:', {
                    id: reply.id,
                    text: reply.text,
                    author: reply.username
                });
            }
        }
    } catch (error) {
        console.error('Error searching for replies:', error);
    }
}

console.log('Starting reply detection test...');
testReplyMethods();