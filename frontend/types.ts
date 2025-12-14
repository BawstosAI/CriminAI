export enum AppMode {
  INITIAL = 'INITIAL',
  CONVERSATION = 'CONVERSATION',
  RENDER = 'RENDER'
}

export enum TurnState {
  IDLE = 'IDLE', // Before mic permission
  LIVE = 'LIVE', // Mic is open, listening continuously
  BOT_SPEAKING = 'BOT_SPEAKING', // Bot is currently outputting audio
  PROCESSING = 'PROCESSING' // Processing user speech
}

export interface BackendMessage {
  type: 'bot_audio' | 'final_render_ready' | 'transcript' | 'error';
  text?: string;
  audio_url?: string;
  video_url?: string;
  image_url?: string;
  sketch_prompt?: string;
  suspect_profile?: Record<string, any>;
}

export interface ConversationMessage {
  role: 'user' | 'assistant';
  text: string;
  timestamp: number;
}

export interface Partner {
  name: string;
  active: boolean;
}