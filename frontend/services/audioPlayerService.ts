/**
 * Audio Player Service - Plays PCM audio from the server
 *
 * Handles streaming PCM audio playback from Kyutai TTS:
 * - Sample Rate: 24000 Hz (Kyutai TTS output)
 * - Format: PCM 16-bit signed integer (little-endian)
 * - Channels: Mono
 */

const TTS_SAMPLE_RATE = 24000;
const MIN_BUFFER_BEFORE_PLAY = 3; // Buffer a few chunks for smoother playback

type PlaybackStateCallback = (isPlaying: boolean) => void;
type AudioLevelCallback = (level: number) => void;

class AudioPlayerService {
  private audioContext: AudioContext | null = null;
  private audioQueue: AudioBuffer[] = [];
  private activeSources: Set<AudioBufferSourceNode> = new Set();
  private isPlaying = false;
  private nextPlayTime = 0;
  private onPlaybackStateChange: PlaybackStateCallback | null = null;
  private onAudioLevel: AudioLevelCallback | null = null;
  private isInitialized = false;
  private pendingChunks = 0;
  private activeTtsId: number | null = null;
  private analyser: AnalyserNode | null = null;
  private fftBuffer: Float32Array | null = null;
  private targetLevel = 0;
  private displayLevel = 0;
  private levelLoopId: number | null = null;

  /**
   * Initialize the audio context (call early, e.g., on user gesture)
   */
  async initialize(): Promise<boolean> {
    if (this.isInitialized && this.audioContext) {
      return true;
    }
    try {
      this.audioContext = new AudioContext({
        sampleRate: TTS_SAMPLE_RATE,
        latencyHint: 'interactive',
      });
      this.isInitialized = true;
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
   * Set callback for TTS audio level updates
   */
  setAudioLevelCallback(callback: AudioLevelCallback | null) {
    this.onAudioLevel = callback;
  }

  /**
   * Start a new TTS stream and clear any previous playback.
   */
  startStream(ttsId: number): void {
    if (this.activeTtsId === ttsId) {
      return;
    }
    this.stop();
    this.activeTtsId = ttsId;
    if (this.audioContext) {
      this.nextPlayTime = this.audioContext.currentTime;
    } else {
      this.nextPlayTime = 0;
    }
  }

  /**
   * Queue PCM audio data for playback
   * @param base64Audio Base64-encoded PCM audio data
   * @param ttsId Optional stream id to ensure we only play current audio
   */
  async queueAudio(base64Audio: string, ttsId?: number): Promise<void> {
    if (typeof ttsId === 'number') {
      if (this.activeTtsId !== ttsId) {
        this.startStream(ttsId);
      }
    } else if (this.activeTtsId === null) {
      // No active stream id; refuse to play unknown audio
      return;
    }

    if (!this.audioContext || !this.isInitialized) {
      console.log('AudioPlayerService: Auto-initializing on queueAudio');
      await this.initialize();
    }

    if (!this.audioContext) {
      console.error('AudioPlayerService: Not initialized');
      return;
    }

    if (this.audioContext.state === 'suspended') {
      await this.audioContext.resume();
    }

    try {
      const binaryString = atob(base64Audio);
      const len = binaryString.length;
      const bytes = new Uint8Array(len);
      for (let i = 0; i < len; i++) {
        bytes[i] = binaryString.charCodeAt(i);
      }

      const pcm16 = new Int16Array(bytes.buffer);
      const numSamples = pcm16.length;
      const float32 = new Float32Array(numSamples);
      const scale = 1 / 32768;
      for (let i = 0; i < numSamples; i++) {
        float32[i] = pcm16[i] * scale;
      }

      const audioBuffer = this.audioContext.createBuffer(1, numSamples, TTS_SAMPLE_RATE);
      audioBuffer.getChannelData(0).set(float32);

      this.updateAudioLevel(float32);

      this.audioQueue.push(audioBuffer);
      this.pendingChunks++;

      if (this.isPlaying) {
        this.scheduleNextBuffers();
      } else if (this.pendingChunks >= MIN_BUFFER_BEFORE_PLAY) {
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

    this.isPlaying = true;
    this.pendingChunks = 0;
    this.nextPlayTime = this.audioContext.currentTime + 0.03;
    this.onPlaybackStateChange?.(true);
    this.scheduleNextBuffers();
  }

  /**
   * Schedule the next buffers in the queue for seamless playback
   */
  private scheduleNextBuffers(): void {
    if (!this.audioContext || !this.isPlaying) return;

    const maxScheduleAhead = 5;
    let scheduled = 0;

    while (this.audioQueue.length > 0 && scheduled < maxScheduleAhead) {
      const buffer = this.audioQueue.shift();
      if (!buffer) break;

      const source = this.audioContext.createBufferSource();
      source.buffer = buffer;
      this.ensureAnalyser(source);

      const startTime = this.nextPlayTime;
      source.start(startTime);
      this.nextPlayTime += buffer.duration;
      scheduled++;

      this.activeSources.add(source);
      source.onended = () => {
        this.activeSources.delete(source);
        if (this.audioQueue.length > 0) {
          this.scheduleNextBuffers();
        } else if (this.activeSources.size === 0) {
          this.isPlaying = false;
          this.onPlaybackStateChange?.(false);
          this.updateAudioLevel(null);
        }
      };
    }
  }

  /**
   * Stop playback and clear queue
   */
  stop(): void {
    this.audioQueue = [];
    this.pendingChunks = 0;

    this.activeSources.forEach((source) => {
      try {
        source.stop();
      } catch (e) {
        // Source might already be stopped
      }
    });
    this.activeSources.clear();

    this.isPlaying = false;
    this.activeTtsId = null;
    if (this.audioContext) {
      this.nextPlayTime = this.audioContext.currentTime;
    } else {
      this.nextPlayTime = 0;
    }
    this.onPlaybackStateChange?.(false);
    this.updateAudioLevel(null);
    if (this.levelLoopId) {
      cancelAnimationFrame(this.levelLoopId);
      this.levelLoopId = null;
    }
    this.displayLevel = 0;
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
    this.isInitialized = false;
  }

  get playing(): boolean {
    return this.isPlaying;
  }

  private updateAudioLevel(samples: Float32Array | null) {
    if (!this.onAudioLevel) return;

    if (!samples || samples.length === 0) {
      this.targetLevel = 0;
      this.ensureLevelLoop();
      return;
    }

    let sum = 0;
    const stride = 64;
    let count = 0;
    for (let i = 0; i < samples.length; i += stride) {
      const s = samples[i];
      sum += s * s;
      count += 1;
    }
    const rms = Math.sqrt(sum / Math.max(1, count));
    const instantaneousLevel = Math.min(1, rms * 3.5);

    this.targetLevel = instantaneousLevel;
    this.ensureLevelLoop();
  }

  private ensureLevelLoop() {
    if (!this.audioContext) {
      this.onAudioLevel?.(0);
      return;
    }

    const step = () => {
      if (!this.audioContext) {
        this.displayLevel = 0;
        this.onAudioLevel?.(0);
        this.levelLoopId = null;
        return;
      }

      // Sample live waveform for up-to-date loudness
      if (this.analyser && this.fftBuffer) {
        this.analyser.getFloatTimeDomainData(this.fftBuffer);
        let sum = 0;
        let count = 0;
        for (let i = 0; i < this.fftBuffer.length; i += 32) {
          const v = this.fftBuffer[i];
          sum += v * v;
          count++;
        }
        const rms = Math.sqrt(sum / Math.max(1, count));
        const liveLevel = Math.min(1, rms * 3.5);
        this.targetLevel = Math.max(liveLevel, this.targetLevel * 0.9);
      }

      const alpha = this.targetLevel > this.displayLevel ? 0.4 : 0.15;
      this.displayLevel += (this.targetLevel - this.displayLevel) * alpha;

      this.onAudioLevel?.(Math.max(0, Math.min(1, this.displayLevel)));

      if (this.targetLevel < 0.01 && this.displayLevel < 0.02 && !this.isPlaying) {
        this.displayLevel = 0;
        this.onAudioLevel?.(0);
        this.levelLoopId = null;
        return;
      }

      this.levelLoopId = requestAnimationFrame(step);
    };

    if (!this.levelLoopId) {
      this.levelLoopId = requestAnimationFrame(step);
    }
  }

  private ensureAnalyser(source?: AudioBufferSourceNode) {
    if (!this.audioContext || !source) return;
    if (!this.analyser) {
      this.analyser = this.audioContext.createAnalyser();
      this.analyser.fftSize = 1024;
      this.analyser.smoothingTimeConstant = 0.0;
      this.fftBuffer = new Float32Array(this.analyser.fftSize);
    }

    source.connect(this.analyser);
    this.analyser.connect(this.audioContext.destination);
  }
}

export const audioPlayerService = new AudioPlayerService();
