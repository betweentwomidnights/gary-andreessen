import sqlite3 from 'sqlite3';
import { open } from 'sqlite';
import fs from 'fs/promises';

class GenerationStore {
    constructor() {
        this.db = null;
    }

    async initialize() {
        try {
            // Open SQLite database
            this.db = await open({
                filename: 'generations.db',
                driver: sqlite3.Database
            });

            // Create tables if they don't exist
            await this.db.exec(`
                CREATE TABLE IF NOT EXISTS generations (
                    tweet_id TEXT PRIMARY KEY,
                    video_id TEXT NOT NULL,
                    timestamp INTEGER NOT NULL,
                    task_id TEXT NOT NULL,
                    audio_data TEXT NOT NULL,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                );

                CREATE TABLE IF NOT EXISTS generation_files (
                    tweet_id TEXT PRIMARY KEY,
                    video_buffer BLOB,
                    FOREIGN KEY(tweet_id) REFERENCES generations(tweet_id)
                );
            `);

            return true;
        } catch (error) {
            console.error('Failed to initialize GenerationStore:', error);
            return false;
        }
    }

    async saveGeneration(tweetId, generationData) {
        try {
            // Store main generation data
            await this.db.run(
                `INSERT INTO generations (
                    tweet_id, video_id, timestamp, task_id, audio_data
                ) VALUES (?, ?, ?, ?, ?)`,
                [
                    tweetId,
                    generationData.videoId,
                    generationData.timestamp,
                    generationData.taskId,
                    generationData.audioData
                ]
            );

            // Store video buffer separately
            if (generationData.videoBuffer) {
                await this.db.run(
                    `INSERT INTO generation_files (tweet_id, video_buffer) 
                     VALUES (?, ?)`,
                    [tweetId, generationData.videoBuffer]
                );
            }

            return true;
        } catch (error) {
            console.error('Error saving generation:', error);
            return false;
        }
    }

    async getGeneration(tweetId) {
        try {
            // Get main generation data
            const generation = await this.db.get(
                'SELECT * FROM generations WHERE tweet_id = ?',
                tweetId
            );

            if (!generation) {
                return null;
            }

            // Get video buffer if it exists
            const fileData = await this.db.get(
                'SELECT video_buffer FROM generation_files WHERE tweet_id = ?',
                tweetId
            );

            return {
                ...generation,
                videoBuffer: fileData?.video_buffer
            };
        } catch (error) {
            console.error('Error retrieving generation:', error);
            return null;
        }
    }

    async getRecentGeneration(tweetId) {
        try {
            // Get the most recent generation for a given tweet
            const generation = await this.db.get(`
                SELECT * FROM generations 
                WHERE tweet_id = ?
                ORDER BY created_at DESC 
                LIMIT 1
            `, tweetId);

            return generation || null;
        } catch (error) {
            console.error('Error getting recent generation:', error);
            return null;
        }
    }

    async cleanup() {
        try {
            // Delete generations older than 24 hours
            const yesterday = new Date();
            yesterday.setHours(yesterday.getHours() - 24);

            await this.db.run(`
                DELETE FROM generation_files 
                WHERE tweet_id IN (
                    SELECT tweet_id 
                    FROM generations 
                    WHERE created_at < ?
                )
            `, yesterday.toISOString());

            await this.db.run(
                'DELETE FROM generations WHERE created_at < ?',
                yesterday.toISOString()
            );

            return true;
        } catch (error) {
            console.error('Error during cleanup:', error);
            return false;
        }
    }

    async close() {
        if (this.db) {
            await this.db.close();
        }
    }
}

export default GenerationStore;