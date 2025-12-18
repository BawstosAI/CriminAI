import React, { useState, useRef, useEffect } from 'react';
import BinaryBackground from './components/BinaryBackground';
import DetectiveAvatar from './components/DetectiveAvatar';
import FinalRender from './components/FinalRender';
import { AppMode, TurnState, ConversationMessage, InteractionMode, BackendMessage } from './types';
import { backendService } from './services/backendService';

const App: React.FC = () => {
  const [appMode, setAppMode] = useState<AppMode>(AppMode.INITIAL);
  const [interactionMode, setInteractionMode] = useState<InteractionMode | null>(null);
  const [turnState, setTurnState] = useState<TurnState>(TurnState.IDLE);
  const [botAudioLevel, setBotAudioLevel] = useState(0);
  const [finalMedia, setFinalMedia] = useState<{video?: string, image?: string}>({});
  const [messages, setMessages] = useState<ConversationMessage[]>([]);
  const [currentBotText, setCurrentBotText] = useState<string>('');
  const [inputText, setInputText] = useState<string>('');
  const [isConnected, setIsConnected] = useState(false);
  const [sketchPrompt, setSketchPrompt] = useState<string | null>(null);
  const [connectionError, setConnectionError] = useState<string | null>(null);
  const [partialTranscript, setPartialTranscript] = useState<string>('');
  
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLInputElement>(null);

  // Auto-scroll messages
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  // Focus input when in text mode
  useEffect(() => {
    if (interactionMode === InteractionMode.TEXT && turnState === TurnState.LIVE) {
      inputRef.current?.focus();
    }
  }, [turnState, interactionMode]);

  const handleRestart = () => {
    backendService.disconnect();
    setFinalMedia({});
    setBotAudioLevel(0);
    setTurnState(TurnState.IDLE);
    setAppMode(AppMode.INITIAL);
    setInteractionMode(null);
    setMessages([]);
    setCurrentBotText('');
    setInputText('');
    setIsConnected(false);
    setSketchPrompt(null);
    setConnectionError(null);
    setPartialTranscript('');
  };

  const selectMode = (mode: InteractionMode) => {
    setInteractionMode(mode);
    if (mode === InteractionMode.TEXT) {
      startTextInteraction();
    } else {
      startAudioInteraction();
    }
  };

  // Common message handler for both modes
  const handleBackendMessage = (msg: BackendMessage) => {
    console.log('Message:', msg);
    if (msg.type === 'final_render_ready' && msg.image_url) {
      setFinalMedia({ image: msg.image_url });
      setAppMode(AppMode.RENDER);
    }
    if (msg.sketch_prompt) {
      setSketchPrompt(msg.sketch_prompt);
    }
    if (msg.type === 'error') {
      setConnectionError(msg.text || 'Unknown error');
      // Clear error after 3 seconds
      setTimeout(() => setConnectionError(null), 3000);
    }
  };

  // Common state change handler
  const handleStateChange = (state: TurnState) => {
    setTurnState(state);
    // Clear partial transcript when state changes to non-listening
    if (state !== TurnState.LIVE) {
      setPartialTranscript('');
    }
  };

  // Common transcript handler
  const handleTranscript = (text: string, isUser: boolean, isPartial?: boolean) => {
    if (isPartial) {
      // Show the latest word - replace rather than accumulate
      setPartialTranscript(prev => {
        const words = prev.trim().split(' ').filter(w => w);
        words.push(text);
        // Keep last 10 words to avoid growing too long
        return words.slice(-10).join(' ');
      });
      console.log('Partial transcription:', text);
      return;
    }
    
    // Clear partial transcript when we get final
    if (isUser) {
      setPartialTranscript('');
    }
    
    const newMessage: ConversationMessage = {
      role: isUser ? 'user' : 'assistant',
      text,
      timestamp: Date.now(),
    };
    setMessages(prev => [...prev, newMessage]);
    if (!isUser) {
      setCurrentBotText(text);
    }
  };

  const startTextInteraction = async () => {
    setConnectionError(null);
    setTurnState(TurnState.PROCESSING);
    
    const connected = await backendService.connect(
      handleBackendMessage,
      handleStateChange,
      handleTranscript
    );

    if (connected) {
      setIsConnected(true);
      setAppMode(AppMode.CONVERSATION);
      setTurnState(TurnState.LIVE);
    } else {
      setConnectionError('Failed to connect to backend. Is the server running?');
      setTurnState(TurnState.IDLE);
    }
  };

  const startAudioInteraction = async () => {
    setConnectionError(null);
    setTurnState(TurnState.PROCESSING);
    
    const connected = await backendService.connectAudio(
      handleBackendMessage,
      handleStateChange,
      handleTranscript
    );

    if (connected) {
      setIsConnected(true);
      setAppMode(AppMode.CONVERSATION);
      // TurnState will be set to LIVE when STT is ready
    } else {
      setConnectionError('Failed to connect to audio backend. Check microphone permissions and server.');
      setTurnState(TurnState.IDLE);
    }
  };

  const handleSendMessage = () => {
    if (!inputText.trim() || turnState === TurnState.PROCESSING) return;
    
    backendService.sendTextMessage(inputText.trim());
    setInputText('');
    setTurnState(TurnState.PROCESSING);
  };

  const handleKeyPress = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSendMessage();
    }
  };

  const handleGenerateImage = () => {
    if (sketchPrompt) {
      backendService.requestImageGeneration(sketchPrompt);
      setTurnState(TurnState.PROCESSING);
    }
  };

  // Demo: Show final render (for audio mode)
  const showDemoRender = () => {
    setFinalMedia({ image: '/demo-sketch.png' });
    setAppMode(AppMode.RENDER);
  };

  const renderModeSelection = () => (
    <div className="w-full h-full flex flex-col items-center justify-center space-y-8 z-20">
      <h1 className="text-2xl md:text-4xl font-mono tracking-tighter border-b-2 border-green-500 pb-2">
        PORTRAIT_ROBOT.EXE
      </h1>
      
      <div className="max-w-md text-center font-mono text-xs md:text-sm text-green-700">
        <p>SELECT INTERACTION MODE</p>
      </div>

      <div className="flex gap-6">
        {/* Text Mode Button */}
        <button 
          onClick={() => selectMode(InteractionMode.TEXT)}
          className="group px-8 py-6 border-2 border-green-500 hover:bg-green-500 hover:text-black font-mono transition-all flex flex-col items-center gap-3"
        >
          <svg className="w-8 h-8" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M8 10h.01M12 10h.01M16 10h.01M9 16H5a2 2 0 01-2-2V6a2 2 0 012-2h14a2 2 0 012 2v8a2 2 0 01-2 2h-5l-5 5v-5z" />
          </svg>
          <span className="font-bold">TEXT MODE</span>
          <span className="text-[10px] text-green-600 group-hover:text-green-900">Type your responses</span>
        </button>

        {/* Audio Mode Button */}
        <button 
          onClick={() => selectMode(InteractionMode.AUDIO)}
          className="group px-8 py-6 border-2 border-green-500 hover:bg-green-500 hover:text-black font-mono transition-all flex flex-col items-center gap-3"
        >
          <svg className="w-8 h-8" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 11a7 7 0 01-7 7m0 0a7 7 0 01-7-7m7 7v4m0 0H8m4 0h4m-4-8a3 3 0 01-3-3V5a3 3 0 116 0v6a3 3 0 01-3 3z" />
          </svg>
          <span className="font-bold">AUDIO MODE</span>
          <span className="text-[10px] text-green-600 group-hover:text-green-900">Speak naturally</span>
        </button>
      </div>

      {connectionError && (
        <div className="px-4 py-2 bg-red-900/30 border border-red-500/50 text-red-400 font-mono text-sm">
          {connectionError}
        </div>
      )}

      <p className="text-[10px] text-green-800 font-mono">
        Both modes connect to backend • Choose your preferred interaction style
      </p>
    </div>
  );

  const renderTextChatUI = () => (
    <div className="absolute bottom-0 left-0 right-0 flex flex-col z-20 max-h-[60vh]">
      {/* Messages */}
      <div className="flex-1 overflow-y-auto px-4 pb-2 space-y-3 max-h-[40vh]">
        {messages.map((msg, idx) => (
          <div
            key={idx}
            className={`flex ${msg.role === 'user' ? 'justify-end' : 'justify-start'}`}
          >
            <div
              className={`max-w-[80%] px-4 py-2 font-mono text-sm ${
                msg.role === 'user'
                  ? 'bg-green-900/40 border border-green-500/30 text-green-300'
                  : 'bg-black/80 border border-green-500/50 text-green-400'
              }`}
            >
              {msg.text}
            </div>
          </div>
        ))}
        <div ref={messagesEndRef} />
      </div>

      {/* Sketch prompt display */}
      {sketchPrompt && (
        <div className="mx-4 mb-2 p-3 bg-yellow-900/30 border border-yellow-500/50">
          <p className="text-yellow-400 font-mono text-xs mb-2">📝 SKETCH PROMPT READY:</p>
          <p className="text-yellow-300 font-mono text-sm">{sketchPrompt}</p>
          <button
            onClick={handleGenerateImage}
            disabled={turnState === TurnState.PROCESSING}
            className="mt-2 px-4 py-1 bg-yellow-600/30 hover:bg-yellow-600/50 border border-yellow-500 text-yellow-300 font-mono text-xs disabled:opacity-50"
          >
            {turnState === TurnState.PROCESSING ? '⏳ GENERATING...' : '🎨 GENERATE IMAGE'}
          </button>
        </div>
      )}

      {/* Input area */}
      <div className="p-4 bg-black/60 border-t border-green-500/30">
        {/* Status */}
        <div className="flex items-center gap-2 mb-2">
          <div className={`w-2 h-2 rounded-full ${
            turnState === TurnState.PROCESSING ? 'bg-yellow-500 animate-pulse' :
            isConnected ? 'bg-green-500' : 'bg-red-500'
          }`} />
          <span className="font-mono text-[10px] text-green-600">
            {turnState === TurnState.PROCESSING ? 'PROCESSING...' :
             isConnected ? 'CONNECTED' : 'DISCONNECTED'}
          </span>
        </div>

        {/* Input */}
        <div className="flex gap-2">
          <input
            ref={inputRef}
            type="text"
            value={inputText}
            onChange={(e) => setInputText(e.target.value)}
            onKeyPress={handleKeyPress}
            disabled={turnState === TurnState.PROCESSING || !isConnected}
            placeholder={isConnected ? "Describe the suspect..." : "Connecting..."}
            className="flex-1 bg-black/80 border border-green-500/50 px-4 py-2 font-mono text-sm text-green-400 placeholder-green-700 focus:outline-none focus:border-green-400 disabled:opacity-50"
          />
          <button
            onClick={handleSendMessage}
            disabled={!inputText.trim() || turnState === TurnState.PROCESSING || !isConnected}
            className="px-6 py-2 bg-green-900/30 border border-green-500 hover:bg-green-700/30 font-mono text-sm disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
          >
            SEND
          </button>
        </div>
      </div>
    </div>
  );

  const renderAudioUI = () => (
    <div className="absolute bottom-0 left-0 w-full flex flex-col z-20 px-4 pb-4" style={{maxHeight: '50vh'}}>
      {/* Messages scroll area */}
      <div className="flex-1 overflow-y-auto space-y-2 mb-4 max-h-[30vh]">
        {messages.map((msg, idx) => (
          <div
            key={idx}
            className={`flex ${msg.role === 'user' ? 'justify-end' : 'justify-start'}`}
          >
            <div
              className={`max-w-[80%] px-3 py-2 font-mono text-sm ${
                msg.role === 'user'
                  ? 'bg-blue-900/40 border border-blue-500/30 text-blue-300'
                  : 'bg-black/80 border border-green-500/50 text-green-400'
              }`}
            >
              {msg.text}
            </div>
          </div>
        ))}
        <div ref={messagesEndRef} />
      </div>

      {/* Sketch prompt display */}
      {sketchPrompt && (
        <div className="mx-4 mb-2 p-3 bg-yellow-900/30 border border-yellow-500/50">
          <p className="text-yellow-400 font-mono text-xs mb-2">📝 SKETCH PROMPT READY:</p>
          <p className="text-yellow-300 font-mono text-sm">{sketchPrompt}</p>
          <button
            onClick={handleGenerateImage}
            disabled={turnState === TurnState.PROCESSING}
            className="mt-2 px-4 py-1 bg-yellow-600/30 hover:bg-yellow-600/50 border border-yellow-500 text-yellow-300 font-mono text-xs disabled:opacity-50"
          >
            {turnState === TurnState.PROCESSING ? '⏳ GENERATING...' : '🎨 GENERATE IMAGE'}
          </button>
        </div>
      )}

      {/* Partial transcription display */}
      {partialTranscript && turnState === TurnState.LIVE && (
        <div className="mx-4 mb-2 p-2 bg-blue-900/30 border border-blue-500/30">
          <p className="text-blue-400 font-mono text-xs mb-1">🎤 You're saying:</p>
          <p className="text-blue-300 font-mono text-sm italic">{partialTranscript.trim()}</p>
        </div>
      )}

      {/* Status and controls */}
      <div className="flex flex-col items-center gap-2 bg-black/60 px-4 py-3 border border-green-600/40">
        {/* Status indicator */}
        <div className="flex items-center gap-2">
          <div className={`w-2 h-2 rounded-full shadow-[0_0_4px_currentColor] ${
            turnState === TurnState.BOT_SPEAKING ? 'bg-green-500 animate-pulse' : 
            turnState === TurnState.PROCESSING ? 'bg-yellow-500 animate-pulse' :
            turnState === TurnState.LIVE ? 'bg-red-500 animate-pulse' :
            'bg-gray-500'
          }`}></div>
          <span className="font-mono text-xs font-bold text-green-400 tracking-wider uppercase">
            {turnState === TurnState.BOT_SPEAKING ? "🔊 DETECTIVE SPEAKING" :
             turnState === TurnState.PROCESSING ? "⏳ PROCESSING..." :
             turnState === TurnState.LIVE ? "🎤 LISTENING..." :
             "CONNECTING..."}
          </span>
        </div>

        {/* Manual send button for audio mode */}
        <div className="flex gap-2 mt-2">
          <button
            onClick={() => {
              backendService.endUserTurn();
              setPartialTranscript('');
            }}
            disabled={turnState !== TurnState.LIVE}
            className="px-6 py-2 bg-blue-900/40 border border-blue-500 hover:bg-blue-700/40 font-mono text-sm text-blue-300 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
          >
            📤 SEND MESSAGE
          </button>
        </div>

        <div className="text-[9px] text-green-700 font-mono mt-1">
          Speak clearly, then click "SEND MESSAGE" when done
        </div>
      </div>
    </div>
  );

  return (
    <div className="w-full h-screen bg-black text-green-500 relative overflow-hidden flex flex-col">
      <BinaryBackground intensity={0.5} />

      {/* Header with restart button when in conversation */}
      {appMode === AppMode.CONVERSATION && (
        <header className="absolute top-4 right-4 z-30">
          <button
            onClick={handleRestart}
            className="px-3 py-1 border border-green-500/30 hover:border-green-500 font-mono text-xs text-green-600 hover:text-green-400 transition-colors"
          >
            ✕ RESTART
          </button>
        </header>
      )}

      {/* Main Content Area */}
      <main className="flex-1 relative z-10 w-full h-full">
        
        {appMode === AppMode.INITIAL && renderModeSelection()}

        {appMode === AppMode.CONVERSATION && (
          <div className="w-full h-full flex items-center justify-center">
            {/* The Detective */}
            <div className="relative transform transition-transform duration-1000" style={{
              marginBottom: interactionMode === InteractionMode.TEXT ? '25vh' : '0'
            }}>
               <DetectiveAvatar 
                  isSpeaking={turnState === TurnState.BOT_SPEAKING}
                  audioLevel={botAudioLevel}
               />
               
               {/* Decorative "Scanning" lines */}
               <div className="absolute inset-0 border-t border-b border-green-500/10 animate-[pulse_4s_infinite]"></div>
            </div>
            
            {/* Mode-specific UI */}
            {interactionMode === InteractionMode.TEXT && renderTextChatUI()}
            {interactionMode === InteractionMode.AUDIO && renderAudioUI()}
          </div>
        )}

        {appMode === AppMode.RENDER && (
          <FinalRender 
            videoUrl={finalMedia.video}
            imageUrl={finalMedia.image}
            onRestart={handleRestart}
          />
        )}

      </main>

      {/* Global Vignette */}
      <div className="absolute inset-0 pointer-events-none bg-[radial-gradient(circle_at_center,transparent_0%,rgba(0,0,0,0.8)_100%)] z-30"></div>
    </div>
  );
};

export default App;
