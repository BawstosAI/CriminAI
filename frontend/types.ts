export enum AppMode {
  INITIAL = 'INITIAL',
  CONVERSATION = 'CONVERSATION',
  RENDER = 'RENDER'
}

export enum TurnState {
  IDLE = 'IDLE', // Before mic permission
  LIVE = 'LIVE', // Mic is open, listening continuously
  BOT_SPEAKING = 'BOT_SPEAKING' // Bot is currently outputting audio (mic stays open)
}

export interface BackendMessage {
  type: 'bot_audio' | 'final_render_ready';
  audio_url?: string;
  video_url?: string;
  image_url?: string;
}

export interface Partner {
  name: string;
  active: boolean;
}