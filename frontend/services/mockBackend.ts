import { BackendMessage } from '../types';

type MessageCallback = (msg: BackendMessage) => void;

class MockBackendService {
  private turnCount = 0;
  private maxTurns = 3; // Short conversation for MVP
  private onMessage: MessageCallback | null = null;
  private isProcessing = false;

  connect(callback: MessageCallback) {
    this.onMessage = callback;
    this.reset();
  }

  reset() {
    this.turnCount = 0;
    this.isProcessing = false;
  }

  // With continuous streaming, we receive chunks constantly (~every 1s)
  sendAudioChunk(base64Data: any) {
    if (this.isProcessing) return; // Don't interrupt if already thinking/speaking

    // Simulate VAD (Voice Activity Detection) logic:
    // Random chance that this chunk represents the "end of a user sentence"
    // In a real app, the backend analyzes silence/semantics.
    const isEndOfSentence = Math.random() > 0.7; // 30% chance per chunk to trigger response

    if (isEndOfSentence) {
      this.triggerBotResponse();
    }
  }

  private triggerBotResponse() {
    this.isProcessing = true;

    // Simulate "Thinking" time (1-2 seconds)
    setTimeout(() => {
      this.turnCount++;

      if (this.turnCount >= this.maxTurns) {
        // Trigger final render
        if (this.onMessage) {
          this.onMessage({
            type: 'final_render_ready',
            video_url: 'https://picsum.photos/seed/video-placeholder/800/600', 
            image_url: 'https://picsum.photos/seed/suspect-final/800/800'
          });
        }
      } else {
        // Trigger bot response
        if (this.onMessage) {
          this.onMessage({
            type: 'bot_audio',
            audio_url: 'mock_audio_blob' 
          });
        }
        
        // After bot is done "speaking" (simulated), we allow processing again
        // We assume bot speech takes about 3-4 seconds
        setTimeout(() => {
            this.isProcessing = false;
        }, 4000);
      }
    }, 1500); 
  }
}

export const mockBackend = new MockBackendService();