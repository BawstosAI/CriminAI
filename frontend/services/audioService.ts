/**
 * Audio Service - Handles microphone capture and audio streaming
 * 
 * Captures audio from the microphone in the format required by Gradium STT:
 * - Sample Rate: 24000 Hz
 * - Format: PCM 16-bit signed integer (little-endian)
 * - Channels: Mono
 * - Chunk Size: 1920 samples (80ms at 24kHz)
 */

const TARGET_SAMPLE_RATE = 24000;
const FRAME_SIZE = 1920; // 80ms at 24kHz
const FRAME_DURATION_MS = 80;
const SPEECH_THRESHOLD = 0.006;
const MIN_SPEECH_FRAMES = 1; // 80ms above threshold to start speech
const SILENCE_TIMEOUT_FRAMES = Math.ceil(1200 / FRAME_DURATION_MS); // ~1.2s silence to end speech

type AudioCallback = (audioBase64: string) => void;
type SpeechCallback = (state: 'start' | 'end') => void;

class AudioService {
  private audioContext: AudioContext | null = null;
  private mediaStream: MediaStream | null = null;
  private workletNode: AudioWorkletNode | null = null;
  private scriptProcessor: ScriptProcessorNode | null = null;
  private sourceNode: MediaStreamAudioSourceNode | null = null;
  private onAudioChunk: AudioCallback | null = null;
  private onSpeechStateChange: SpeechCallback | null = null;
  private isCapturing = false;
  private lastOnAudioChunk: AudioCallback | null = null;
  private lastOnSpeechStateChange: SpeechCallback | null = null;

  /**
   * Request microphone permission and set up audio capture
   */
  async initialize(): Promise<boolean> {
    try {
      // Request microphone access
      this.mediaStream = await navigator.mediaDevices.getUserMedia({
        audio: {
          sampleRate: TARGET_SAMPLE_RATE,
          channelCount: 1,
          echoCancellation: true,
          noiseSuppression: true,
          autoGainControl: true,
        },
      });

      // Create audio context with target sample rate
      this.audioContext = new AudioContext({
        sampleRate: TARGET_SAMPLE_RATE,
      });

      console.log(`AudioService: Initialized with sample rate ${this.audioContext.sampleRate}Hz`);
      return true;

    } catch (error) {
      console.error('AudioService: Failed to initialize:', error);
      return false;
    }
  }

  /**
   * Start capturing audio and sending chunks
   */
  async startCapture(onAudioChunk: AudioCallback, onSpeechStateChange?: SpeechCallback): Promise<boolean> {
    if (!this.audioContext || !this.mediaStream) {
      console.error('AudioService: Not initialized');
      return false;
    }

    if (this.isCapturing) {
      console.warn('AudioService: Already capturing');
      return true;
    }

    this.onAudioChunk = onAudioChunk;
    this.onSpeechStateChange = onSpeechStateChange || null;
    this.lastOnAudioChunk = onAudioChunk;
    this.lastOnSpeechStateChange = onSpeechStateChange || null;
    this.isCapturing = true;

    try {
      if (this.audioContext.state === 'suspended') {
        await this.audioContext.resume();
      }
      // Create source node from microphone
      this.sourceNode = this.audioContext.createMediaStreamSource(this.mediaStream);

      // Use smaller buffer size for lower latency (2048 instead of 4096)
      const bufferSize = 2048;
      this.scriptProcessor = this.audioContext.createScriptProcessor(bufferSize, 1, 1);

      // Audio buffer for accumulating samples
      let audioBuffer: Float32Array = new Float32Array(0);

      let speechFrames = 0;
      let silenceFrames = 0;
      let speaking = false;
      let noiseFloor = 0.005;

      this.scriptProcessor.onaudioprocess = (event) => {
        if (!this.isCapturing) return;

        const inputData = event.inputBuffer.getChannelData(0);
        
        // Resample if needed (usually browser already gives us 24kHz)
        const resampledData = this.resample(
          inputData,
          this.audioContext!.sampleRate,
          TARGET_SAMPLE_RATE
        );

        // Append to buffer
        const newBuffer = new Float32Array(audioBuffer.length + resampledData.length);
        newBuffer.set(audioBuffer);
        newBuffer.set(resampledData, audioBuffer.length);
        audioBuffer = newBuffer;

        // Send complete frames immediately when ready
        while (audioBuffer.length >= FRAME_SIZE) {
          const frame = audioBuffer.slice(0, FRAME_SIZE);
          audioBuffer = audioBuffer.slice(FRAME_SIZE);

          const rms = this.calculateRms(frame);
          if (!speaking) {
            noiseFloor = noiseFloor * 0.98 + rms * 0.02;
          }
          const dynamicThreshold = Math.max(SPEECH_THRESHOLD, noiseFloor * 2.0);
          const speechDetected = rms > dynamicThreshold;

          if (speechDetected) {
            speechFrames++;
            silenceFrames = 0;
            if (!speaking && speechFrames >= MIN_SPEECH_FRAMES) {
              speaking = true;
              silenceFrames = 0;
              this.onSpeechStateChange?.('start');
            }
          } else {
            speechFrames = 0;
            if (speaking) {
              silenceFrames++;
              if (silenceFrames >= SILENCE_TIMEOUT_FRAMES) {
                speaking = false;
                silenceFrames = 0;
                this.onSpeechStateChange?.('end');
              }
            }
          }

          // Light preamp to boost quiet speech
          const boosted = frame.map((v) => Math.max(-1, Math.min(1, v * 1.4)));

          // Convert to PCM 16-bit
          const pcmData = this.floatToPCM16(boosted);
          
          // Send as base64 immediately
          const base64 = this.arrayBufferToBase64(pcmData.buffer as ArrayBuffer);
          this.onAudioChunk?.(base64);
        }
      };

      // Connect: microphone -> processor -> destination (needed for processor to work)
      this.sourceNode.connect(this.scriptProcessor);
      this.scriptProcessor.connect(this.audioContext.destination);

      console.log('AudioService: Capture started');
      return true;

    } catch (error) {
      console.error('AudioService: Failed to start capture:', error);
      this.isCapturing = false;
      return false;
    }
  }

  /**
   * Stop capturing audio
   */
  stopCapture(): void {
    this.isCapturing = false;

    if (this.scriptProcessor) {
      this.scriptProcessor.disconnect();
      this.scriptProcessor = null;
    }

    if (this.sourceNode) {
      this.sourceNode.disconnect();
      this.sourceNode = null;
    }

    console.log('AudioService: Capture stopped');
  }

  /**
   * Clean up all resources
   */
  dispose(): void {
    this.stopCapture();

    if (this.mediaStream) {
      this.mediaStream.getTracks().forEach(track => track.stop());
      this.mediaStream = null;
    }

    if (this.audioContext) {
      this.audioContext.close();
      this.audioContext = null;
    }

    console.log('AudioService: Disposed');
  }

  /**
   * Ensure the audio context is running (e.g., after autoplay policies pause it).
   */
  async ensureActive(): Promise<void> {
    if (this.audioContext && this.audioContext.state === 'suspended') {
      try {
        await this.audioContext.resume();
        console.log('AudioService: Resumed suspended AudioContext');
      } catch (err) {
        console.warn('AudioService: Failed to resume AudioContext', err);
      }
    }
    // If for some reason we lost capture, restart with last callbacks
    if (!this.isCapturing && this.lastOnAudioChunk) {
      await this.startCapture(this.lastOnAudioChunk, this.lastOnSpeechStateChange || undefined);
    }
  }

  /**
   * Resample audio data to target sample rate
   */
  private resample(
    inputData: Float32Array,
    inputRate: number,
    outputRate: number
  ): Float32Array {
    if (inputRate === outputRate) {
      return inputData;
    }

    const ratio = inputRate / outputRate;
    const outputLength = Math.floor(inputData.length / ratio);
    const output = new Float32Array(outputLength);

    for (let i = 0; i < outputLength; i++) {
      const srcIndex = i * ratio;
      const srcIndexFloor = Math.floor(srcIndex);
      const srcIndexCeil = Math.min(srcIndexFloor + 1, inputData.length - 1);
      const frac = srcIndex - srcIndexFloor;

      // Linear interpolation
      output[i] = inputData[srcIndexFloor] * (1 - frac) + inputData[srcIndexCeil] * frac;
    }

    return output;
  }

  /**
   * Convert Float32Array (-1 to 1) to Int16Array PCM
   */
  private floatToPCM16(floatData: Float32Array): Int16Array {
    const pcm = new Int16Array(floatData.length);
    for (let i = 0; i < floatData.length; i++) {
      // Clamp to [-1, 1] and convert to 16-bit
      const s = Math.max(-1, Math.min(1, floatData[i]));
      pcm[i] = s < 0 ? s * 0x8000 : s * 0x7FFF;
    }
    return pcm;
  }

  private calculateRms(frame: Float32Array): number {
    let sum = 0;
    for (let i = 0; i < frame.length; i++) {
      const sample = frame[i];
      sum += sample * sample;
    }
    return Math.sqrt(sum / frame.length);
  }

  /**
   * Convert ArrayBuffer to base64 string
   */
  private arrayBufferToBase64(buffer: ArrayBuffer): string {
    const bytes = new Uint8Array(buffer);
    let binary = '';
    for (let i = 0; i < bytes.length; i++) {
      binary += String.fromCharCode(bytes[i]);
    }
    return btoa(binary);
  }

  get capturing(): boolean {
    return this.isCapturing;
  }
}

export const audioService = new AudioService();
