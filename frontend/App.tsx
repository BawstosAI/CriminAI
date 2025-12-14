import React, { useState, useEffect, useRef, useCallback } from 'react';
import BinaryBackground from './components/BinaryBackground';
import DetectiveAvatar from './components/DetectiveAvatar';
import FinalRender from './components/FinalRender';
import { backendService } from './services/backendService';
import { AppMode, TurnState, BackendMessage, ConversationMessage } from './types';

const App: React.FC = () => {
  const [appMode, setAppMode] = useState<AppMode>(AppMode.INITIAL);
  const [turnState, setTurnState] = useState<TurnState>(TurnState.IDLE);
  const [botAudioLevel, setBotAudioLevel] = useState(0);
  const [userAudioLevel, setUserAudioLevel] = useState(0);
  const [finalMedia, setFinalMedia] = useState<{video?: string, image?: string}>({});
  const [messages, setMessages] = useState<ConversationMessage[]>([]);
  const [currentBotText, setCurrentBotText] = useState<string>('');
  const [isConnected, setIsConnected] = useState(false);
  
  // Push-to-talk state
  const [isRecording, setIsRecording] = useState(false);
  const [audioChunks, setAudioChunks] = useState<Blob[]>([]);
  const [canSpeak, setCanSpeak] = useState(true); // User can speak when bot is not talking

  // Refs for Audio
  const mediaRecorderRef = useRef<MediaRecorder | null>(null);
  const audioContextRef = useRef<AudioContext | null>(null);
  const analyserRef = useRef<AnalyserNode | null>(null);
  const dataArrayRef = useRef<Uint8Array | null>(null);
  const animationFrameRef = useRef<number | null>(null);
  const audioStreamRef = useRef<MediaStream | null>(null);
  const audioChunksRef = useRef<Blob[]>([]);

  // Initialize Backend Connection
  useEffect(() => {
    backendService.connect(
      // Message handler
      (msg: BackendMessage) => {
        if (msg.type === 'bot_audio') {
          setTurnState(TurnState.BOT_SPEAKING);
          setCanSpeak(false);
          setCurrentBotText(msg.text || '');
          
          // Use browser TTS to speak the response
          if (msg.text) {
            speakWithVisualization(msg.text);
          }
        } else if (msg.type === 'final_render_ready') {
          stopAudioSystem();
          setFinalMedia({ 
            image: msg.image_url 
          });
          setAppMode(AppMode.RENDER);
        }
      },
      // State change handler
      (state: TurnState) => {
        console.log('Turn state changed:', state);
        setTurnState(state);
        setIsConnected(true);
        
        // Allow user to speak when back to LIVE state
        if (state === TurnState.LIVE) {
          setCanSpeak(true);
        } else if (state === TurnState.BOT_SPEAKING || state === TurnState.PROCESSING) {
          setCanSpeak(false);
        }
      },
      // Transcript handler
      (text: string, isUser: boolean) => {
        if (text.trim()) {
          setMessages(prev => [...prev, {
            role: isUser ? 'user' : 'assistant',
            text,
            timestamp: Date.now()
          }]);
        }
      },
      // VAD status handler (for debugging)
      (pausePrediction: number, conversationState: string) => {
        // Could display VAD status in UI for debugging
        // console.log(`VAD: pause=${pausePrediction.toFixed(2)}, state=${conversationState}`);
      }
    );

    return () => {
      stopAudioSystem();
      backendService.disconnect();
    };
  }, []);

  const stopAudioSystem = () => {
    if (mediaRecorderRef.current && mediaRecorderRef.current.state !== 'inactive') {
        mediaRecorderRef.current.stop();
    }
    if (animationFrameRef.current) {
        cancelAnimationFrame(animationFrameRef.current);
    }
    if (audioContextRef.current && audioContextRef.current.state !== 'closed') {
        audioContextRef.current.close();
    }
    if (audioStreamRef.current) {
        audioStreamRef.current.getTracks().forEach(track => track.stop());
    }
    window.speechSynthesis?.cancel();
  };

  const handleRestart = () => {
    stopAudioSystem();
    setFinalMedia({});
    setBotAudioLevel(0);
    setUserAudioLevel(0);
    setTurnState(TurnState.IDLE);
    setAppMode(AppMode.INITIAL);
    setMessages([]);
    setCurrentBotText('');
    backendService.reset();
  };

  // Speak text with audio level visualization
  const speakWithVisualization = async (text: string) => {
    if (!('speechSynthesis' in window)) {
      console.warn('Speech synthesis not supported');
      setTurnState(TurnState.LIVE);
      setCanSpeak(true); // Re-enable speaking when done
      return;
    }

    return new Promise<void>((resolve) => {
      const utterance = new SpeechSynthesisUtterance(text);
      utterance.rate = 0.9;
      utterance.pitch = 0.9;
      
      // Simulate audio levels while speaking
      const speakInterval = setInterval(() => {
        setBotAudioLevel(Math.random() * 0.6 + 0.2);
      }, 100);

      utterance.onend = () => {
        clearInterval(speakInterval);
        setBotAudioLevel(0);
        setTurnState(TurnState.LIVE);
        setCanSpeak(true); // Re-enable speaking when bot finishes
        resolve();
      };

      utterance.onerror = () => {
        clearInterval(speakInterval);
        setBotAudioLevel(0);
        setTurnState(TurnState.LIVE);
        setCanSpeak(true); // Re-enable speaking on error too
        resolve();
      };

      window.speechSynthesis.speak(utterance);
    });
  };

  const startInteraction = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ 
        audio: {
          sampleRate: 16000,
          channelCount: 1,
          echoCancellation: true,
          noiseSuppression: true,
          autoGainControl: true
        } 
      });
      
      audioStreamRef.current = stream;
      
      // 1. Setup Audio Context for visualization (User Voice)
      audioContextRef.current = new (window.AudioContext || (window as any).webkitAudioContext)();
      const source = audioContextRef.current.createMediaStreamSource(stream);
      analyserRef.current = audioContextRef.current.createAnalyser();
      analyserRef.current.fftSize = 32;
      source.connect(analyserRef.current);
      
      const bufferLength = analyserRef.current.frequencyBinCount;
      dataArrayRef.current = new Uint8Array(bufferLength);

      // Start visualization loop
      const updateLevel = () => {
          if (analyserRef.current && dataArrayRef.current) {
              analyserRef.current.getByteFrequencyData(dataArrayRef.current);
              let sum = 0;
              for(let i = 0; i < bufferLength; i++) {
                  sum += dataArrayRef.current[i];
              }
              const average = sum / bufferLength;
              setUserAudioLevel(average / 255);
          }
          animationFrameRef.current = requestAnimationFrame(updateLevel);
      };
      updateLevel();

      // 2. Start voice mode on backend
      if (!backendService.startVoiceMode()) {
        throw new Error('Failed to start voice mode - backend not connected');
      }

      // 3. Setup MediaRecorder (but don't start yet - push-to-talk)
      const recorder = new MediaRecorder(stream, { mimeType: 'audio/webm' });
      
      recorder.ondataavailable = async (e) => {
        console.log(`🎤 ondataavailable: size=${e.data.size}`);
        if (e.data.size > 0) {
          audioChunksRef.current.push(e.data);
        }
      };

      recorder.onstop = async () => {
        console.log('🔴 Recorder stopped, chunks:', audioChunksRef.current.length);
      };

      mediaRecorderRef.current = recorder;
      
      setAppMode(AppMode.CONVERSATION);
      setTurnState(TurnState.LIVE);
      setCanSpeak(true);

    } catch (err) {
      console.error("Microphone access denied or backend error:", err);
      alert("Microphone access is required for the investigation. Please allow microphone access and ensure the backend is running.");
    }
  };

  // Push-to-talk: Start recording
  const startRecording = () => {
    if (!mediaRecorderRef.current || !canSpeak) return;
    
    console.log('🎙️ Starting recording...');
    audioChunksRef.current = [];
    setIsRecording(true);
    setTurnState(TurnState.LIVE);
    
    try {
      mediaRecorderRef.current.start();
    } catch (e) {
      console.error('Failed to start recording:', e);
      setIsRecording(false);
    }
  };

  // Push-to-talk: Stop recording and send audio
  const stopAndSendRecording = async () => {
    if (!mediaRecorderRef.current || !isRecording) return;
    
    console.log('📤 Stopping and sending recording...');
    setIsRecording(false);
    setCanSpeak(false);
    setTurnState(TurnState.PROCESSING);
    
    // Stop the recorder - this triggers ondataavailable with final data
    mediaRecorderRef.current.stop();
    
    // Wait a moment for the final chunk to be collected
    await new Promise(resolve => setTimeout(resolve, 100));
    
    // Combine all chunks and send
    if (audioChunksRef.current.length > 0) {
      const audioBlob = new Blob(audioChunksRef.current, { type: 'audio/webm' });
      const arrayBuffer = await audioBlob.arrayBuffer();
      const base64Audio = btoa(String.fromCharCode(...new Uint8Array(arrayBuffer)));
      
      console.log(`📤 Sending complete audio: ${base64Audio.length} chars`);
      
      // Send as a complete audio message
      backendService.sendAudioChunk(base64Audio);
      
      // Signal end of speech
      backendService.stopListening();
    }
    
    audioChunksRef.current = [];
  };

  // Interaction UI Logic
  const renderInteractionLayer = () => {
    if (appMode !== AppMode.CONVERSATION) return null;

    // Generate a visual bar string based on user volume
    const totalBars = 20;
    const activeBars = Math.floor(userAudioLevel * totalBars);
    const barString = "I".repeat(activeBars) + ".".repeat(totalBars - activeBars);

    return (
      <div className="absolute bottom-10 left-0 w-full flex flex-col items-center justify-center z-20 gap-2 px-4">
        
        {/* Current bot response text */}
        {currentBotText && turnState === TurnState.BOT_SPEAKING && (
          <div className="max-w-lg bg-black/90 border border-green-500/50 px-4 py-3 mb-4 rounded">
            <p className="font-mono text-green-400 text-sm leading-relaxed">
              {currentBotText}
            </p>
          </div>
        )}

        {/* Text-based VU Meter - only show when recording */}
        {isRecording && (
          <div className="font-mono text-green-500 text-sm md:text-base bg-black/80 px-4 py-2 border-l-2 border-r-2 border-green-900">
              INPUT_LEVEL: [{barString}]
          </div>
        )}

        {/* Push-to-Talk Controls */}
        <div className="flex gap-4 mt-4">
          {!isRecording && canSpeak && (
            <button
              onClick={startRecording}
              className="px-6 py-3 bg-green-900/50 border-2 border-green-500 hover:bg-green-700/50 font-mono font-bold transition-colors flex items-center gap-2"
            >
              <span className="w-3 h-3 bg-red-500 rounded-full"></span>
              HOLD TO SPEAK
            </button>
          )}
          
          {isRecording && (
            <button
              onClick={stopAndSendRecording}
              className="px-6 py-3 bg-red-900/50 border-2 border-red-500 hover:bg-red-700/50 font-mono font-bold transition-colors animate-pulse flex items-center gap-2"
            >
              <span className="w-3 h-3 bg-red-500 rounded-full animate-ping"></span>
              SEND MESSAGE
            </button>
          )}
        </div>

        {/* Connection Status */}
        <div className="flex items-center gap-2 bg-black/60 px-3 py-1 border border-green-600/40 shadow-[0_0_5px_rgba(0,100,0,0.2)] mt-2">
            <div className={`w-1.5 h-1.5 rounded-full shadow-[0_0_4px_currentColor] ${
              isRecording ? 'bg-red-500 animate-pulse' : 
              turnState === TurnState.BOT_SPEAKING ? 'bg-green-500' : 
              turnState === TurnState.PROCESSING ? 'bg-yellow-500' :
              'bg-blue-500'
            }`}></div>
            <span className="font-mono text-[10px] font-bold text-green-400 tracking-widest uppercase">
                {isRecording ? "🎤 RECORDING..." : 
                 turnState === TurnState.BOT_SPEAKING ? "🔊 DETECTIVE SPEAKING" :
                 turnState === TurnState.PROCESSING ? "⏳ PROCESSING" :
                 canSpeak ? "✅ READY - CLICK TO SPEAK" : "⏳ WAIT..."}
            </span>
        </div>
        
        <div className="text-[9px] text-green-900 font-mono mt-1 opacity-60">
            PUSH-TO-TALK MODE ACTIVE
        </div>
      </div>
    );
  };

  return (
    <div className="w-full h-screen bg-black text-green-500 relative overflow-hidden flex flex-col">
      <BinaryBackground intensity={0.5} />

      {/* Main Content Area */}
      <main className="flex-1 relative z-10 w-full h-full">
        
        {appMode === AppMode.INITIAL && (
          <div className="w-full h-full flex flex-col items-center justify-center space-y-8 z-20">
             <h1 className="text-2xl md:text-4xl font-mono tracking-tighter border-b-2 border-green-500 pb-2">
               PORTRAIT_ROBOT.EXE
             </h1>
             <div className="max-w-md text-center font-mono text-xs md:text-sm text-green-700">
               <p>INITIATING FORENSIC INTERFACE...</p>
               <p>MICROPHONE PERMISSION REQUIRED.</p>
               <p>OBJECTIVE: IDENTIFY SUSPECT.</p>
             </div>
             <button 
               onClick={startInteraction}
               className="px-8 py-3 border border-green-500 hover:bg-green-500 hover:text-black font-mono font-bold transition-colors"
             >
               INITIALIZE SYSTEM
             </button>
          </div>
        )}

        {appMode === AppMode.CONVERSATION && (
          <div className="w-full h-full flex items-center justify-center">
            {/* The Detective */}
            <div className="relative transform transition-transform duration-1000">
               <DetectiveAvatar 
                  isSpeaking={turnState === TurnState.BOT_SPEAKING}
                  audioLevel={botAudioLevel}
               />
               
               {/* Decorative "Scanning" lines - Always active in conversation now */}
               <div className="absolute inset-0 border-t border-b border-green-500/10 animate-[pulse_4s_infinite]"></div>
            </div>
            
            {renderInteractionLayer()}
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