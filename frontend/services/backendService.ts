import { BackendMessage, TurnState } from '../types';

type MessageCallback = (msg: BackendMessage) => void;
type StateCallback = (state: TurnState) => void;
type TranscriptCallback = (text: string, isUser: boolean) => void;
type VADStatusCallback = (pausePrediction: number, conversationState: string) => void;

class BackendService {
  private ws: WebSocket | null = null;
  private onMessage: MessageCallback | null = null;
  private onStateChange: StateCallback | null = null;
  private onTranscript: TranscriptCallback | null = null;
  private onVADStatus: VADStatusCallback | null = null;
  private sessionId: string | null = null;
  private audioContext: AudioContext | null = null;
  private isConnected = false;
  private reconnectAttempts = 0;
  private maxReconnectAttempts = 5;
  private isBotSpeaking = false;
  private pendingTranscripts: string[] = [];

  // Configure backend URL - use environment variable or default
  private get wsUrl(): string {
    const host = import.meta.env.VITE_BACKEND_HOST || window.location.hostname || 'localhost';
    const port = import.meta.env.VITE_BACKEND_PORT || '8000';
    return `ws://${host}:${port}/v1/realtime`;
  }

  connect(
    onMessage: MessageCallback,
    onStateChange?: StateCallback,
    onTranscript?: TranscriptCallback,
    onVADStatus?: VADStatusCallback
  ) {
    this.onMessage = onMessage;
    this.onStateChange = onStateChange || null;
    this.onTranscript = onTranscript || null;
    this.onVADStatus = onVADStatus || null;
    this.initWebSocket();
  }

  private initWebSocket() {
    try {
      console.log(`Connecting to backend at ${this.wsUrl}`);
      this.ws = new WebSocket(this.wsUrl);

      this.ws.onopen = () => {
        console.log('WebSocket connected');
        this.isConnected = true;
        this.reconnectAttempts = 0;
      };

      this.ws.onmessage = (event) => {
        try {
          const data = JSON.parse(event.data);
          this.handleMessage(data);
        } catch (err) {
          console.error('Failed to parse message:', err);
        }
      };

      this.ws.onclose = () => {
        console.log('WebSocket disconnected');
        this.isConnected = false;
        this.sessionId = null;
        
        // Attempt reconnection
        if (this.reconnectAttempts < this.maxReconnectAttempts) {
          this.reconnectAttempts++;
          console.log(`Reconnecting... attempt ${this.reconnectAttempts}`);
          setTimeout(() => this.initWebSocket(), 2000);
        }
      };

      this.ws.onerror = (error) => {
        console.error('WebSocket error:', error);
      };
    } catch (err) {
      console.error('Failed to connect:', err);
    }
  }

  private handleMessage(data: any) {
    console.log('📥 Received from backend:', JSON.stringify(data, null, 2));

    switch (data.type) {
      case 'session_state':
        console.log(`📊 Session state: ${data.data?.status}, mode: ${data.data?.mode}`);
        this.sessionId = data.session_id;
        if (data.data?.status === 'listening') {
          this.isBotSpeaking = false;
          this.onStateChange?.(TurnState.LIVE);
        } else if (data.data?.status === 'processing') {
          this.onStateChange?.(TurnState.PROCESSING);
        }
        break;

      case 'speech_started':
        // User started speaking - detected by VAD
        console.log('🎤 User speech started (VAD detected)');
        this.onStateChange?.(TurnState.LIVE);
        break;

      case 'speech_ended':
        // User finished speaking - detected by VAD
        console.log('🎤 User speech ended:', data.data?.text);
        this.onStateChange?.(TurnState.PROCESSING);
        // Send full transcript
        if (data.data?.text) {
          this.onTranscript?.(data.data.text, true);
        }
        break;

      case 'bot_speaking':
        // Bot is about to speak
        console.log('🤖 Bot speaking');
        this.isBotSpeaking = true;
        this.onStateChange?.(TurnState.BOT_SPEAKING);
        break;

      case 'bot_finished':
        // Bot finished speaking
        console.log('🤖 Bot finished');
        this.isBotSpeaking = false;
        this.onStateChange?.(TurnState.LIVE);
        break;

      case 'transcript':
        // Partial or final transcript from STT
        const text = data.data?.text || '';
        const isFinal = data.data?.is_final || false;
        const pauseProb = data.data?.pause_probability || 0;
        
        console.log(`📝 Transcript: "${text}" (final=${isFinal}, pause=${pauseProb.toFixed(2)})`);
        
        if (text) {
          // Collect partial transcripts
          this.pendingTranscripts.push(text);
        }
        break;

      case 'vad_status':
        // VAD status update for debugging/visualization
        const pausePrediction = data.data?.pause_prediction || 0;
        const conversationState = data.data?.conversation_state || '';
        console.log(`📊 VAD: pause=${pausePrediction.toFixed(2)}, state=${conversationState}`);
        this.onVADStatus?.(pausePrediction, conversationState);
        break;

      case 'assistant_text':
        // Bot's text response - will be spoken via TTS
        console.log('🤖 Assistant response:', data.data.text?.substring(0, 100));
        this.onTranscript?.(data.data.text, false);
        this.onMessage?.({
          type: 'bot_audio',
          text: data.data.text,
          suspect_profile: data.data.suspect_profile
        });
        break;

      case 'assistant_audio':
        // Raw audio from TTS
        this.playAudio(data.data.audio);
        break;

      case 'sketch_ready':
        // Final sketch is ready
        this.onMessage?.({
          type: 'final_render_ready',
          image_url: `/api/generate?prompt=${encodeURIComponent(data.data.prompt)}`,
          sketch_prompt: data.data.prompt,
          suspect_profile: data.data.suspect_profile
        });
        break;

      case 'error':
        console.error('Backend error:', data.data.message);
        break;
    }
  }

  // Start voice mode - begins listening
  startVoiceMode() {
    if (!this.ws || this.ws.readyState !== WebSocket.OPEN) {
      console.error('WebSocket not connected');
      return false;
    }

    this.pendingTranscripts = [];
    this.isBotSpeaking = false;

    this.ws.send(JSON.stringify({
      type: 'start_listening'
    }));

    // Send initial greeting request
    this.ws.send(JSON.stringify({
      type: 'text_input',
      data: { text: 'Hello, I would like to start creating a forensic sketch.' }
    }));

    return true;
  }

  // Send audio chunk to backend - with VAD consideration
  sendAudioChunk(audioData: string) {
    if (!this.ws || this.ws.readyState !== WebSocket.OPEN) {
      console.warn('⚠️ Cannot send audio: WebSocket not open');
      return;
    }

    // Don't send audio while bot is speaking (could be echo)
    if (this.isBotSpeaking) {
      console.log('⏸️ Skipping audio send - bot is speaking');
      return;
    }

    console.log(`🎙️ Sending audio chunk: ${audioData.length} chars (base64)`);

    this.ws.send(JSON.stringify({
      type: 'audio_chunk',
      data: { audio: audioData }
    }));
  }

  // Signal end of speech segment (manual stop)
  stopListening() {
    if (!this.ws || this.ws.readyState !== WebSocket.OPEN) {
      return;
    }

    // Send any pending transcripts
    this.pendingTranscripts = [];

    this.ws.send(JSON.stringify({
      type: 'stop_listening'
    }));
  }

  // Check if bot is currently speaking
  get botSpeaking(): boolean {
    return this.isBotSpeaking;
  }

  // End the session
  endSession() {
    if (!this.ws || this.ws.readyState !== WebSocket.OPEN) {
      return;
    }

    this.ws.send(JSON.stringify({
      type: 'end_session'
    }));
  }

  // Play audio from base64
  private async playAudio(base64Audio: string) {
    try {
      if (!this.audioContext) {
        this.audioContext = new (window.AudioContext || (window as any).webkitAudioContext)();
      }

      // Decode base64 to array buffer
      const binaryString = atob(base64Audio);
      const len = binaryString.length;
      const bytes = new Uint8Array(len);
      for (let i = 0; i < len; i++) {
        bytes[i] = binaryString.charCodeAt(i);
      }

      // Create audio blob and play
      const audioBlob = new Blob([bytes], { type: 'audio/wav' });
      const audioUrl = URL.createObjectURL(audioBlob);
      const audio = new Audio(audioUrl);
      
      audio.onended = () => {
        URL.revokeObjectURL(audioUrl);
        this.onStateChange?.(TurnState.LIVE);
      };

      await audio.play();
    } catch (err) {
      console.error('Failed to play audio:', err);
    }
  }

  // Use browser TTS as fallback
  speakText(text: string): Promise<void> {
    return new Promise((resolve) => {
      if (!('speechSynthesis' in window)) {
        console.warn('Speech synthesis not supported');
        resolve();
        return;
      }

      const utterance = new SpeechSynthesisUtterance(text);
      utterance.rate = 0.9;
      utterance.pitch = 0.8;
      utterance.onend = () => resolve();
      utterance.onerror = () => resolve();
      
      window.speechSynthesis.speak(utterance);
    });
  }

  reset() {
    this.endSession();
    this.sessionId = null;
  }

  disconnect() {
    if (this.ws) {
      this.ws.close();
      this.ws = null;
    }
    if (this.audioContext && this.audioContext.state !== 'closed') {
      this.audioContext.close();
      this.audioContext = null;
    }
  }

  get connected(): boolean {
    return this.isConnected && this.ws?.readyState === WebSocket.OPEN;
  }
}

export const backendService = new BackendService();
