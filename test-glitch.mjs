import fs from 'fs';
import decode from 'audio-decode';
import ffmpeg from 'fluent-ffmpeg';
import path from 'path';
import { fileURLToPath } from 'url';
import { dirname } from 'path';

const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);

async function extractAudioFromVideo(inputPath) {
    const outputPath = path.join(__dirname, 'temp_audio.wav');
    
    return new Promise((resolve, reject) => {
        ffmpeg(inputPath)
            .toFormat('wav')
            .on('end', () => resolve(outputPath))
            .on('error', reject)
            .save(outputPath);
    });
}

function analyzeChunk(audioData, offset, length) {
    // Get a chunk of the audio data
    const chunk = audioData.slice(offset, offset + length);
    
    // Calculate RMS (root mean square) for amplitude
    const rms = Math.sqrt(chunk.reduce((sum, sample) => sum + (sample * sample), 0) / length);
    
    // Simple way to detect bass - look at every 4th sample to approximate low frequencies
    const bassSum = chunk.reduce((sum, sample, i) => {
        if (i % 4 === 0) return sum + Math.abs(sample);
        return sum;
    }, 0);
    
    return {
        amplitude: rms,
        bassLevel: bassSum / (length / 4),
        timestamp: offset / 44100 // assuming 44.1kHz sample rate
    };
}

async function analyzeAudioFile(audioPath) {
    try {
        // Read and decode the audio file
        const audioData = await fs.promises.readFile(audioPath);
        const audioBuffer = await decode(audioData);
        
        console.log('Audio decoded:', {
            sampleRate: audioBuffer.sampleRate,
            length: audioBuffer.length,
            duration: audioBuffer.length / audioBuffer.sampleRate,
            channels: audioBuffer.numberOfChannels
        });

        // Get the first channel's data
        const channelData = audioBuffer.getChannelData(0);
        
        // Analyze in chunks (30fps = analyze every 1470 samples at 44.1kHz)
        const samplesPerFrame = Math.floor(audioBuffer.sampleRate / 30);
        const frames = [];
        
        for (let i = 0; i < channelData.length; i += samplesPerFrame) {
            const analysis = analyzeChunk(channelData, i, Math.min(samplesPerFrame, channelData.length - i));
            frames.push(analysis);
        }

        // Log some sample data
        console.log('\nFirst few frames of analysis:');
        console.log(frames.slice(0, 5));
        
        // Log some statistics
        const maxAmplitude = Math.max(...frames.map(f => f.amplitude));
        const maxBass = Math.max(...frames.map(f => f.bassLevel));
        console.log('\nStats:', {
            totalFrames: frames.length,
            maxAmplitude,
            maxBass,
            durationSeconds: frames.length / 30
        });

        return frames;
        
    } catch (error) {
        console.error('Error analyzing audio:', error);
        throw error;
    }
}

async function test() {
    try {
        console.log('Extracting audio from video...');
        const audioPath = await extractAudioFromVideo('./marc_beats.mp4');
        
        console.log('\nAnalyzing audio...');
        const analysis = await analyzeAudioFile(audioPath);
        
        // Clean up
        fs.unlinkSync(audioPath);
        
        return analysis;
        
    } catch (error) {
        console.error('Error:', error);
    }
}

test();