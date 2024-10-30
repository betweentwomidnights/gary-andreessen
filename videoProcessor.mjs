import ffmpeg from 'fluent-ffmpeg';
import ffmpegPath from 'ffmpeg-static';
import { PassThrough } from 'stream';
import os from 'os';
import fs from 'fs';
import path from 'path';

ffmpeg.setFfmpegPath(ffmpegPath);

class VideoProcessor {
    constructor() {
        this.tempDir = os.tmpdir();
    }

    async createVideoWithAudio(inputVideo, audioBuffer, options = {}) {
        const {
            baseEffectIntensity = 0.4,
            maxEffectIntensity = 2.0
        } = options;

        return new Promise((resolve, reject) => {
            const tempFiles = {
                audio: path.join(this.tempDir, `temp_${Date.now()}_audio.mp3`),
                output: path.join(this.tempDir, `temp_${Date.now()}_output.mp4`),
                inputVideo: typeof inputVideo === 'string' ? inputVideo : path.join(this.tempDir, `temp_${Date.now()}_input.mp4`)
            };

            if (Buffer.isBuffer(inputVideo)) {
                fs.writeFileSync(tempFiles.inputVideo, inputVideo);
            }

            fs.writeFileSync(tempFiles.audio, audioBuffer);

            ffmpeg.ffprobe(tempFiles.audio, (err, metadata) => {
                if (err) {
                    this.cleanupTempFiles(tempFiles);
                    return reject(err);
                }

                const audioDuration = metadata.format.duration;
                console.log('Audio duration:', audioDuration, 'seconds');

                const filterComplex = [
                    '-filter_complex',
                    `[0:v]split[raw][foreffects];` +
                    `[raw]trim=0:6[firstpart];` +
                    `[foreffects]trim=5.9:6,setpts=PTS-STARTPTS[lastframe];` +
                    `[1:a]asplit[a1][a2];` +
                    `[a1]atrim=0:6[firstaudio];` +
                    `[a2]atrim=6:${audioDuration},asetpts=PTS-STARTPTS[secondaudio];` +
                    `[lastframe]loop=${Math.ceil(audioDuration-6)}:1:0,` +
                    `negate,hue=s=1.5,eq=contrast=1.5[secondpart];` +
                    `[firstpart][secondpart]concat=n=2:v=1[outv];` +
                    `[firstaudio][secondaudio]concat=n=2:v=0:a=1[outa]`,
                    '-map', '[outv]',
                    '-map', '[outa]'
                ];

                const command = ffmpeg()
                    .input(tempFiles.inputVideo)
                    .input(tempFiles.audio)
                    .outputOptions([
                        '-c:v libx264',
                        '-preset fast',
                        '-c:a aac',
                        '-b:a 192k',
                        '-pix_fmt yuv420p',
                        '-t', String(audioDuration),
                        ...filterComplex
                    ])
                    .fps(30);

                command
                    .save(tempFiles.output)
                    .on('start', (cmd) => console.log('FFmpeg command:', cmd))
                    .on('end', () => {
                        const outputBuffer = fs.readFileSync(tempFiles.output);
                        this.cleanupTempFiles(tempFiles);
                        resolve(outputBuffer);
                    })
                    .on('error', (err, stdout, stderr) => {
                        console.error('FFmpeg error:', err);
                        console.error('FFmpeg stderr:', stderr);
                        this.cleanupTempFiles(tempFiles);
                        reject(err);
                    });
            });
        });
    }

    async createContinuationVideo(originalVideo, newAudioBuffer) {
        return new Promise((resolve, reject) => {
            const tempFiles = {
                originalVideo: path.join(this.tempDir, `temp_${Date.now()}_original.mp4`),
                newAudio: path.join(this.tempDir, `temp_${Date.now()}_newaudio.mp3`),
                output: path.join(this.tempDir, `temp_${Date.now()}_output.mp4`)
            };

            fs.writeFileSync(tempFiles.originalVideo, originalVideo);
            fs.writeFileSync(tempFiles.newAudio, newAudioBuffer);

            ffmpeg.ffprobe(tempFiles.newAudio, (err, metadata) => {
                if (err) {
                    this.cleanupTempFiles(tempFiles);
                    return reject(err);
                }

                const newAudioDuration = metadata.format.duration;
                console.log('New audio duration:', newAudioDuration, 'seconds');

                // Calculate how long we need to loop the last frame
                const loopDuration = newAudioDuration - 6; // Subtract original video duration

                const filterComplex = [
                    '-filter_complex',
                    `[0:v]split=2[originalVideo][forLoop];` +
                    
                    `[originalVideo]trim=0:6,setpts=PTS-STARTPTS[firstPart];` +
                    
                    `[forLoop]trim=5.9:6,setpts=PTS-STARTPTS,` +
                    `loop=${Math.ceil(loopDuration)}:1:0,` +
                    `negate,hue=s=1.5,eq=contrast=1.5[loopedPart];` +
                    
                    `[firstPart][loopedPart]concat=n=2:v=1[outv];` +
                    
                    `[1:a]asetpts=PTS-STARTPTS[outa]`,
                    '-map', '[outv]',
                    '-map', '[outa]'
                ];

                ffmpeg()
                    .input(tempFiles.originalVideo)
                    .input(tempFiles.newAudio)
                    .outputOptions([
                        '-c:v libx264',
                        '-preset fast',
                        '-c:a aac',
                        '-b:a 192k',
                        '-pix_fmt yuv420p',
                        '-t', String(newAudioDuration),
                        ...filterComplex
                    ])
                    .fps(30)
                    .save(tempFiles.output)
                    .on('start', (cmd) => console.log('FFmpeg command:', cmd))
                    .on('end', () => {
                        const outputBuffer = fs.readFileSync(tempFiles.output);
                        this.cleanupTempFiles(tempFiles);
                        resolve(outputBuffer);
                    })
                    .on('error', (err, stdout, stderr) => {
                        console.error('FFmpeg error:', err);
                        console.error('FFmpeg stderr:', stderr);
                        this.cleanupTempFiles(tempFiles);
                        reject(err);
                    });
            });
        });
    }

    cleanupTempFiles(files) {
        Object.values(files).forEach(file => {
            try {
                if (fs.existsSync(file)) {
                    fs.unlinkSync(file);
                }
            } catch (e) {
                console.error('Error cleaning up temp file:', e);
            }
        });
    }
}

export default VideoProcessor;