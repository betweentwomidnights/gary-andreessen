import fs from 'fs';
import decode from 'audio-decode';
import ffmpeg from 'fluent-ffmpeg';
import path from 'path';
import { createCanvas, loadImage } from 'canvas';
import os from 'os';

class GlitchProcessor {
    constructor() {
        // We'll initialize tempDir in processVideoBuffer now, not in constructor
    }

    async processVideoBuffer(videoBuffer, startTime = 6) {
        // Create a new temp directory for each processing attempt
        this.tempDir = path.join(os.tmpdir(), `glitch_${Date.now()}`);
        
        try {
            // Ensure temp directory exists
            if (fs.existsSync(this.tempDir)) {
                // If it exists (rare case), clean it up first
                this.cleanup();
            }
            fs.mkdirSync(this.tempDir);

            const inputPath = path.join(this.tempDir, 'input.mp4');
            fs.writeFileSync(inputPath, videoBuffer);

            // Get video dimensions first
            const dimensions = await this.getVideoDimensions(inputPath);
            this.canvas = createCanvas(dimensions.width, dimensions.height);
            this.ctx = this.canvas.getContext('2d');

            // Process video
            const audioData = await this.analyzeAudio(inputPath);
            const normalizedAudioData = this.normalizeAudioData(audioData);
            await this.extractFrames(inputPath);
            await this.processFrames(normalizedAudioData, startTime);
            const outputBuffer = await this.createFinalVideo(inputPath);

            // Cleanup and return
            this.cleanup();
            return outputBuffer;

        } catch (error) {
            console.error('Error in processVideoBuffer:', error);
            this.cleanup();
            throw error;
        }
    }

    getVideoDimensions(inputPath) {
        return new Promise((resolve, reject) => {
            ffmpeg.ffprobe(inputPath, (err, metadata) => {
                if (err) reject(err);
                const stream = metadata.streams.find(s => s.codec_type === 'video');
                resolve({
                    width: stream.width,
                    height: stream.height
                });
            });
        });
    }

    normalizeAudioData(audioData) {
        const maxAmplitude = Math.max(...audioData.map(frame => frame.amplitude));
        const maxBass = Math.max(...audioData.map(frame => frame.bassLevel));
        
        return audioData.map(frame => ({
            ...frame,
            amplitude: frame.amplitude / maxAmplitude,
            bassLevel: frame.bassLevel / maxBass
        }));
    }

    async analyzeAudio(inputPath) {
        const audioPath = path.join(this.tempDir, 'temp_audio.wav');
        
        await new Promise((resolve, reject) => {
            ffmpeg(inputPath)
                .toFormat('wav')
                .on('end', resolve)
                .on('error', reject)
                .save(audioPath);
        });

        const audioData = await fs.promises.readFile(audioPath);
        const audioBuffer = await decode(audioData);
        
        const channelData = audioBuffer.getChannelData(0);
        const samplesPerFrame = Math.floor(audioBuffer.sampleRate / 30);
        const frames = [];
        
        for (let i = 0; i < channelData.length; i += samplesPerFrame) {
            const chunk = channelData.slice(i, i + samplesPerFrame);
            
            // Calculate RMS amplitude
            const rms = Math.sqrt(chunk.reduce((sum, sample) => sum + (sample * sample), 0) / chunk.length);
            
            // Bass detection using longer intervals
            const bassSum = chunk.reduce((sum, sample, idx) => {
                if (idx % 8 === 0) return sum + Math.abs(sample);
                return sum;
            }, 0);
            
            frames.push({
                amplitude: rms,
                bassLevel: bassSum / (chunk.length / 8),
                timestamp: i / audioBuffer.sampleRate
            });
        }

        fs.unlinkSync(audioPath);
        return frames;
    }

    async extractFrames(inputPath) {
        return new Promise((resolve, reject) => {
            ffmpeg(inputPath)
                .outputOptions(['-vf fps=30'])
                .output(path.join(this.tempDir, 'frame_%d.png'))
                .on('end', resolve)
                .on('error', reject)
                .run();
        });
    }

    async applyEffects(bassLevel, amplitude) {
        const imageData = this.ctx.getImageData(0, 0, this.canvas.width, this.canvas.height);
        const data = imageData.data;
        
        // RGB shift based on bass level
        const shift = Math.floor(bassLevel * 30);
        for (let i = 0; i < data.length; i += 4) {
            if (i + shift * 4 < data.length) {
                data[i] = data[i + shift * 4];     // red
                data[i + 2] = data[i + shift * 4]; // blue
            }
        }
        
        this.ctx.putImageData(imageData, 0, 0);
        
        // Chromatic aberration on heavy bass hits
        if (bassLevel > 0.4) {
            const tempCanvas = createCanvas(this.canvas.width, this.canvas.height);
            const tempCtx = tempCanvas.getContext('2d');
            tempCtx.drawImage(this.canvas, 0, 0);
            
            this.ctx.globalCompositeOperation = 'screen';
            this.ctx.fillStyle = 'rgba(255,0,0,0.5)';
            this.ctx.drawImage(tempCanvas, -3 * bassLevel, 0);
            
            this.ctx.fillStyle = 'rgba(0,0,255,0.5)';
            this.ctx.drawImage(tempCanvas, 3 * bassLevel, 0);
            
            this.ctx.globalCompositeOperation = 'source-over';
        }
        
        // Vertical glitch blocks on high amplitude
        if (amplitude > 0.5) {
            const numBlocks = Math.floor(amplitude * 8);
            for (let i = 0; i < numBlocks; i++) {
                const blockHeight = Math.max(1, Math.random() * (this.canvas.height / 20));
                const y = Math.min(
                    this.canvas.height - blockHeight,
                    Math.random() * this.canvas.height
                );
                const shift = (Math.random() - 0.5) * 40 * amplitude;
                
                if (blockHeight > 0 && y < this.canvas.height) {
                    const blockData = this.ctx.getImageData(0, y, this.canvas.width, blockHeight);
                    this.ctx.putImageData(blockData, shift, y);
                }
            }
        }
    }

    async processFrames(audioData, startTime = 6) {
        const frames = fs.readdirSync(this.tempDir)
            .filter(f => f.startsWith('frame_'))
            .sort((a, b) => {
                const numA = parseInt(a.match(/\d+/)[0]);
                const numB = parseInt(b.match(/\d+/)[0]);
                return numA - numB;
            });

        const startFrame = Math.floor(startTime * 30);

        for (let i = 0; i < audioData.length; i++) {
            const frameData = audioData[i];
            const timestamp = i / 30;
            
            const framePath = path.join(this.tempDir, frames[Math.min(i, frames.length - 1)]);
            const frame = await loadImage(framePath);
            this.ctx.drawImage(frame, 0, 0);
            
            if (i >= startFrame) {
                await this.applyEffects(frameData.bassLevel, frameData.amplitude);
            }
            
            const frameNumber = (i + 1).toString();
            const out = fs.createWriteStream(path.join(this.tempDir, `processed_frame_${frameNumber}.png`));
            const stream = this.canvas.createPNGStream();
            await new Promise((resolve, reject) => {
                stream.pipe(out).on('finish', resolve).on('error', reject);
            });
        }
    }

    async createFinalVideo(inputPath) {
        const audioPath = path.join(this.tempDir, 'audio.mp3');
        await new Promise((resolve, reject) => {
            ffmpeg(inputPath)
                .output(audioPath)
                .on('end', resolve)
                .on('error', reject)
                .run();
        });

        const outputPath = path.join(this.tempDir, 'output_glitch.mp4');
        
        await new Promise((resolve, reject) => {
            ffmpeg()
                .input(path.join(this.tempDir, 'processed_frame_%d.png'))
                .inputFPS(30)
                .input(audioPath)
                .outputOptions(['-c:v libx264', '-pix_fmt yuv420p'])
                .output(outputPath)
                .on('end', resolve)
                .on('error', reject)
                .run();
        });

        const outputBuffer = fs.readFileSync(outputPath);
        return outputBuffer;
    }

    cleanup() {
        if (!this.tempDir || !fs.existsSync(this.tempDir)) {
            return;
        }

        try {
            const files = fs.readdirSync(this.tempDir);
            for (const file of files) {
                try {
                    fs.unlinkSync(path.join(this.tempDir, file));
                } catch (error) {
                    console.error('Error cleaning up file:', file, error);
                }
            }
            fs.rmdirSync(this.tempDir);
        } catch (error) {
            console.error('Error in cleanup:', error);
        }
    }
}

export default GlitchProcessor;