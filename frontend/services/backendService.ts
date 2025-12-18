/**
 * Backend Service - WebSocket connection for text and audio interaction
 */

import { BackendMessage, TurnState } from '../types';
import { audioService } from './audioService';
import { audioPlayerService } from './audioPlayerService';

type MessageCallback = (msg: BackendMessage) => void;
type StateCallback = (state: TurnState) => void;
type TranscriptCallback = (text: string, isUser: boolean, isPartial?: boolean) => void;

// WebSocket URLs for different modes
const TEXT_WS_URL = import.meta.env.VITE_TEXT_WS_URL || 'ws://localhost:8092';
const AUDIO_WS_URL = import.meta.env.VITE_AUDIO_WS_URL || 'ws://localhost:8093';

class BackendService {
  private ws: WebSocket | null = null;
  private onMessage: MessageCallback | null = null;
  private onStateChange: StateCallback | null = null;
  private onTranscript: TranscriptCallback | null = null;
  private _connected = false;
  private reconnectAttempts = 0;
  private maxReconnectAttempts = 3;
  private isAudioMode = false;

  /**
   * Connect to the text WebSocket server
   */
  connect(
    onMessage: MessageCallback,
    onStateChange?: StateCallback,
    onTranscript?: TranscriptCallback,
  ): Promise<boolean> {
    this.isAudioMode = false;
    return this._connectToServer(TEXT_WS_URL, onMessage, onStateChange, onTranscript);
  }

  /**
   * Connect to the audio WebSocket server
   */
  async connectAudio(
    onMessage: MessageCallback,
    onStateChange?: StateCallback,
    onTranscript?: TranscriptCallback,
  ): Promise<boolean> {
    this.isAudioMode = true;

    // Initialize both audio services upfront to avoid latency later
    const [audioInitialized, playerInitialized] = await Promise.all([
      audioService.initialize(),
      audioPlayerService.initialize(),
    ]);
    
    if (!audioInitialized) {
      console.error('BackendService: Failed to initialize audio capture');
      return false;
    }
    
    if (!playerInitialized) {
      console.error('BackendService: Failed to initialize audio player');
      // Continue anyway - player will auto-init when needed
    }

    return this._connectToServer(AUDIO_WS_URL, onMessage, onStateChange, onTranscript);
  }

  private _connectToServer(
    wsUrl: string,
    onMessage: MessageCallback,
    onStateChange?: StateCallback,
    onTranscript?: TranscriptCallback,
  ): Promise<boolean> {
    return new Promise((resolve) => {
      this.onMessage = onMessage;
      this.onStateChange = onStateChange || null;
      this.onTranscript = onTranscript || null;

      try {
        console.log(`BackendService: Connecting to ${wsUrl}...`);
        this.ws = new WebSocket(wsUrl);

        this.ws.onopen = () => {
          console.log('BackendService: Connected');
          this._connected = true;
          this.reconnectAttempts = 0;
          resolve(true);
        };

        this.ws.onmessage = (event) => {
          try {
            const data = JSON.parse(event.data);
            this.handleServerMessage(data);
          } catch (e) {
            console.error('Failed to parse message:', e);
          }
        };

        this.ws.onclose = () => {
          console.log('BackendService: Disconnected');
          this._connected = false;
          this.ws = null;
          // Stop audio if in audio mode
          if (this.isAudioMode) {
            audioService.stopCapture();
          }
        };

        this.ws.onerror = (error) => {
          console.error('BackendService: WebSocket error:', error);
          this._connected = false;
          resolve(false);
        };

        // Timeout after 5 seconds
        setTimeout(() => {
          if (!this._connected) {
            console.error('BackendService: Connection timeout');
            resolve(false);
          }
        }, 5000);

      } catch (e) {
        console.error('BackendService: Failed to connect:', e);
        resolve(false);
      }
    });
  }

  private handleServerMessage(data: any) {
    console.log('BackendService: Received:', data.type);

    switch (data.type) {
      case 'assistant_message':
        if (this.onTranscript) {
          this.onTranscript(data.text, false, false);
        }
        if (this.onMessage) {
          this.onMessage({
            type: 'transcript',
            text: data.text,
          });
        }
        if (this.onStateChange) {
          this.onStateChange(TurnState.LIVE);
        }
        break;

      case 'transcription':
        // Audio mode: STT transcription result
        if (this.onTranscript) {
          this.onTranscript(data.text, true, data.is_partial);
        }
        break;

      case 'stt_ready':
        // Audio mode: STT is ready, start capturing
        console.log('BackendService: STT ready, starting audio capture');
        this.startAudioCapture();
        if (this.onStateChange) {
          this.onStateChange(TurnState.LIVE);
        }
        break;

      case 'thinking':
        // Audio mode: LLM is processing
        if (this.onStateChange) {
          this.onStateChange(TurnState.PROCESSING);
        }
        break;

      case 'typing':
        if (this.onStateChange) {
          this.onStateChange(TurnState.PROCESSING);
        }
        break;

      case 'stop_audio':
        // Explicitly stop and clear audio queue
        console.log('BackendService: Stopping audio playback');
        audioPlayerService.stop();
        if (typeof data.tts_id === 'number') {
          audioPlayerService.startStream(data.tts_id);
        }
        break;

      case 'bot_speaking_start':
        // Audio mode: Bot starting to speak
        console.log('BackendService: Bot speaking start - audio player ready');
        if (this.onStateChange) {
          this.onStateChange(TurnState.BOT_SPEAKING);
        }
        if (typeof data.tts_id === 'number') {
          audioPlayerService.startStream(data.tts_id);
        } else {
          // Fallback: ensure any previous playback is cleared
          audioPlayerService.stop();
        }
        // Ensure context is resumed (needed after user gesture)
        audioPlayerService.resume();
        break;

      case 'bot_speaking_end':
        // Audio mode: Bot finished speaking
        console.log('BackendService: Bot speaking end');
        if (this.onStateChange) {
          this.onStateChange(TurnState.LIVE);
        }
        break;

      case 'audio':
        // Audio mode: Incoming TTS audio chunk
        if (data.audio && this.isAudioMode) {
          console.log(`BackendService: Received audio chunk, ${data.audio.length} chars`);
          audioPlayerService.queueAudio(data.audio, data.tts_id);
        }
        break;

      case 'interview_complete':
        if (this.onMessage) {
          this.onMessage({
            type: 'transcript',
            text: data.text,
            sketch_prompt: data.text, // The prompt itself
          });
        }
        break;

      case 'generating_image':
        if (this.onStateChange) {
          this.onStateChange(TurnState.PROCESSING);
        }
        break;

      case 'image_ready':
        if (this.onMessage) {
          this.onMessage({
            type: 'final_render_ready',
            image_url: data.image_url, // Use base64 data URL from server
            sketch_prompt: data.text,
          });
        }
        break;

      case 'error':
        if (this.onMessage) {
          this.onMessage({
            type: 'error',
            text: data.text,
          });
        }
        break;

      case 'pong':
        // Heartbeat response
        break;
    }
  }

  sendTextMessage(text: string) {
    if (!this.ws || !this._connected) {
      console.error('BackendService: Not connected');
      return;
    }
    this.ws.send(JSON.stringify({
      type: 'user_message',
      text: text,
    }));
    
    // Notify UI of user message
    if (this.onTranscript) {
      this.onTranscript(text, true, false);
    }
  }

  requestImageGeneration(prompt?: string) {
    if (!this.ws || !this._connected) {
      console.error('BackendService: Not connected');
      return;
    }
    this.ws.send(JSON.stringify({
      type: 'generate_image',
      prompt: prompt,
    }));
  }

  /**
   * Start capturing audio from microphone and sending to server
   */
  private startAudioCapture(): void {
    if (!this.isAudioMode) return;

    audioService.startCapture((audioBase64: string) => {
      this.sendAudioChunk(audioBase64);
    });
  }

  /**
   * Send an audio chunk to the server
   */
  sendAudioChunk(audioData: string) {
    if (!this.ws || !this._connected) return;
    
    this.ws.send(JSON.stringify({
      type: 'audio',
      audio: audioData,
    }));
  }

  /**
   * Signal end of user turn (for audio mode)
   */
  endUserTurn() {
    if (!this.ws || !this._connected) return;
    
    this.ws.send(JSON.stringify({
      type: 'end_turn',
    }));
  }

  /**
   * Stop audio capture
   */
  stopListening() {
    if (this.isAudioMode) {
      audioService.stopCapture();
    }
  }

  endSession() {
    this.stopListening();
    if (this.ws) {
      this.ws.close();
    }
  }

  reset() {
    this.endSession();
    this._connected = false;
    this.ws = null;
    this.isAudioMode = false;
  }

  disconnect() {
    if (this.isAudioMode) {
      audioService.dispose();
      audioPlayerService.dispose();
    }
    this.reset();
  }

  /**
   * Stop any audio playback
   */
  stopAudioPlayback() {
    audioPlayerService.stop();
  }

  get connected(): boolean {
    return this._connected;
  }

  get audioMode(): boolean {
    return this.isAudioMode;
  }
}

export const backendService = new BackendService();
