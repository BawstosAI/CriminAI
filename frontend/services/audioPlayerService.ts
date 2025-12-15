/**
 * Audio Player Service - Plays PCM audio from the server
 * 
 * Handles streaming PCM audio playback from Gradium TTS:
 * - Sample Rate: 48000 Hz
 * - Format: PCM 16-bit signed integer (little-endian)
 * - Channels: Mono
 */

const TTS_SAMPLE_RATE = 48000;

type PlaybackStateCallback = (isPlaying: boolean) => void;

class AudioPlayerService {
  private audioContext: AudioContext | null = null;
  private audioQueue: AudioBuffer[] = [];
  private isPlaying = false;
  private currentSource: AudioBufferSourceNode | null = null;
  private nextPlayTime = 0;
  private onPlaybackStateChange: PlaybackStateCallback | null = null;

  /**
   * Initialize the audio context
   */
  async initialize(): Promise<boolean> {
    try {
      this.audioContext = new AudioContext({
        sampleRate: TTS_SAMPLE_RATE,
      });
      console.log(`AudioPlayerService: Initialized with sample rate ${this.audioContext.sampleRate}Hz`);
      return true;
    } catch (error) {
      console.error('AudioPlayerService: Failed to initialize:', error);
      return false;
    }
  }

  /**
   * Set callback for playback state changes
   */
  setPlaybackStateCallback(callback: PlaybackStateCallback) {
    this.onPlaybackStateChange = callback;
  }

  /**
   * Queue PCM audio data for playback
   * @param base64Audio Base64-encoded PCM audio data
   */
  async queueAudio(base64Audio: string): Promise<void> {
    if (!this.audioContext) {
      console.log('AudioPlayerService: Auto-initializing on queueAudio');
      await this.initialize();
    }

    if (!this.audioContext) {
      console.error('AudioPlayerService: Not initialized');
      return;
    }

    // Resume if suspended (e.g., browser autoplay policy)
    if (this.audioContext.state === 'suspended') {
      console.log('AudioPlayerService: Resuming suspended context');
      await this.audioContext.resume();
    }

    try {
      // Decode base64 to ArrayBuffer
      const binaryString = atob(base64Audio);
      const bytes = new Uint8Array(binaryString.length);
      for (let i = 0; i < binaryString.length; i++) {
        bytes[i] = binaryString.charCodeAt(i);
      }
      
      console.log(`AudioPlayerService: Processing ${bytes.length} bytes of PCM data`);

      // Convert PCM 16-bit to Float32
      const pcm16 = new Int16Array(bytes.buffer);
      const float32 = new Float32Array(pcm16.length);
      for (let i = 0; i < pcm16.length; i++) {
        // Convert 16-bit int to float (-1 to 1)
        float32[i] = pcm16[i] / 32768;
      }

      // Create audio buffer
      const audioBuffer = this.audioContext.createBuffer(1, float32.length, TTS_SAMPLE_RATE);
      audioBuffer.getChannelData(0).set(float32);
      
      console.log(`AudioPlayerService: Queued buffer with ${float32.length} samples (${audioBuffer.duration.toFixed(2)}s)`);

      // Queue for playback
      this.audioQueue.push(audioBuffer);

      // Start playback if not already playing
      if (!this.isPlaying) {
        this.startPlayback();
      }
    } catch (error) {
      console.error('AudioPlayerService: Failed to queue audio:', error);
    }
  }

  /**
   * Start playing queued audio
   */
  private startPlayback(): void {
    if (!this.audioContext || this.audioQueue.length === 0) {
      return;
    }

    console.log('AudioPlayerService: Starting playback');
    this.isPlaying = true;
    this.nextPlayTime = this.audioContext.currentTime;
    this.onPlaybackStateChange?.(true);
    this.playNext();
  }

  /**
   * Play the next buffer in the queue
   */
  private playNext(): void {
    if (!this.audioContext) return;

    const buffer = this.audioQueue.shift();
    if (!buffer) {
      // Queue empty, stop playback
      this.isPlaying = false;
      this.onPlaybackStateChange?.(false);
      return;
    }

    // Create source node
    const source = this.audioContext.createBufferSource();
    source.buffer = buffer;
    source.connect(this.audioContext.destination);

    // Schedule playback
    source.start(this.nextPlayTime);
    this.nextPlayTime += buffer.duration;

    // When this buffer ends, play next
    source.onended = () => {
      if (this.audioQueue.length > 0) {
        this.playNext();
      } else {
        this.isPlaying = false;
        this.onPlaybackStateChange?.(false);
      }
    };

    this.currentSource = source;
  }

  /**
   * Stop playback and clear queue
   */
  stop(): void {
    this.audioQueue = [];
    
    if (this.currentSource) {
      try {
        this.currentSource.stop();
      } catch (e) {
        // Already stopped
      }
      this.currentSource = null;
    }

    this.isPlaying = false;
    this.onPlaybackStateChange?.(false);
  }

  /**
   * Resume audio context (needed after user gesture)
   */
  async resume(): Promise<void> {
    if (this.audioContext?.state === 'suspended') {
      await this.audioContext.resume();
    }
  }

  /**
   * Clean up resources
   */
  dispose(): void {
    this.stop();
    if (this.audioContext) {
      this.audioContext.close();
      this.audioContext = null;
    }
  }

  get playing(): boolean {
    return this.isPlaying;
  }
}

export const audioPlayerService = new AudioPlayerService();
