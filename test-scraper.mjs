import { Scraper } from './twitter-client/dist/node/esm/index.mjs';
import fs from 'fs/promises';
import 'dotenv/config';

async function saveCookies(cookies) {
    try {
        const cookiesArray = cookies.map(cookie => ({
            key: cookie.key,
            value: cookie.value,
            domain: cookie.domain,
            path: cookie.path
        }));
        
        await fs.writeFile('cookies.json', JSON.stringify(cookiesArray, null, 2));
        console.log('Cookies saved successfully!');
    } catch (error) {
        console.error('Error saving cookies:', error);
    }
}

async function tryWithExistingCookies() {
    console.log('Trying to use existing cookies...');
    const scraper = new Scraper();
    
    try {
        // Try to load and set cookies
        const cookiesJson = await fs.readFile('cookies.json', 'utf-8');
        const cookies = JSON.parse(cookiesJson);
        await scraper.setCookies(cookies);
        
        // Test if cookies work by getting user ID
        await scraper.getUserIdByScreenName(process.env.TWITTER_USERNAME);
        
        console.log('Existing cookies work!');
        return scraper;
    } catch (error) {
        console.log('Existing cookies failed, will need to login');
        return null;
    }
}

async function testScraper() {
    let scraper;
    const username = process.env.TWITTER_USERNAME;
    
    try {
        // Try to use existing cookies first
        scraper = await tryWithExistingCookies();
        
        // If that didn't work, do a fresh login
        if (!scraper) {
            console.log('Creating new scraper instance...');
            scraper = new Scraper();
            
            console.log('\nAttempting to login...');
            await scraper.login(
                process.env.TWITTER_USERNAME,
                process.env.TWITTER_PASSWORD
            );
            
            // Save new cookies
            const cookies = await scraper.getCookies();
            console.log('Got new cookies:', cookies.length);
            await saveCookies(cookies);
        }

        // Test getting tweets
        console.log('\nTesting tweet retrieval...');
        const userId = await scraper.getUserIdByScreenName(username);
        console.log('Got user ID:', userId);

        const tweets = await scraper.getUserTweets(userId, 10);
        console.log('\nTweets found:', tweets?.tweets?.length ?? 0);

        if (tweets?.tweets?.length > 0) {
            tweets.tweets.forEach((tweet, index) => {
                console.log(`\nTweet ${index + 1}:`, {
                    id: tweet.id,
                    text: tweet.text?.substring(0, 100) + '...',
                    createdAt: tweet.createdAt
                });
            });
        }

    } catch (error) {
        console.error('Error occurred:', {
            message: error?.message || 'Unknown error',
            type: error?.constructor?.name || 'Unknown type',
            details: error?.toString()
        });
    }
}

console.log('Starting scraper test...');
testScraper();