import React, { useState, useEffect, useRef, useCallback } from 'react';
import BinaryBackground from './components/BinaryBackground';
import DetectiveAvatar from './components/DetectiveAvatar';
import FinalRender from './components/FinalRender';
import { mockBackend } from './services/mockBackend';
import { AppMode, TurnState, BackendMessage } from './types';

const App: React.FC = () => {
  const [appMode, setAppMode] = useState<AppMode>(AppMode.INITIAL);
  const [turnState, setTurnState] = useState<TurnState>(TurnState.IDLE);
  const [botAudioLevel, setBotAudioLevel] = useState(0);
  const [userAudioLevel, setUserAudioLevel] = useState(0); // To visualize user input
  const [finalMedia, setFinalMedia] = useState<{video?: string, image?: string}>({});

  // Refs for Audio
  const mediaRecorderRef = useRef<MediaRecorder | null>(null);
  const audioContextRef = useRef<AudioContext | null>(null);
  const analyserRef = useRef<AnalyserNode | null>(null);
  const dataArrayRef = useRef<Uint8Array | null>(null);
  const animationFrameRef = useRef<number | null>(null);

  // Initialize Backend Connection
  useEffect(() => {
    mockBackend.connect((msg: BackendMessage) => {
      if (msg.type === 'bot_audio') {
        setTurnState(TurnState.BOT_SPEAKING);
        simulateBotSpeech();
      } else if (msg.type === 'final_render_ready') {
        stopAudioSystem();
        setFinalMedia({ video: msg.video_url, image: msg.image_url });
        setAppMode(AppMode.RENDER);
      }
    });

    return () => stopAudioSystem();
  }, []);

  const stopAudioSystem = () => {
    if (mediaRecorderRef.current && mediaRecorderRef.current.state !== 'inactive') {
        mediaRecorderRef.current.stop();
    }
    if (animationFrameRef.current) {
        cancelAnimationFrame(animationFrameRef.current);
    }
    // Fix: Check if context is already closed before closing
    if (audioContextRef.current && audioContextRef.current.state !== 'closed') {
        audioContextRef.current.close();
    }
  };

  const handleRestart = () => {
    stopAudioSystem();
    setFinalMedia({});
    setBotAudioLevel(0);
    setUserAudioLevel(0);
    setTurnState(TurnState.IDLE);
    setAppMode(AppMode.INITIAL);
    mockBackend.reset(); // Reset backend simulation state
  };

  const simulateBotSpeech = () => {
    let duration = 3000; // 3 seconds of speech
    const interval = setInterval(() => {
      setBotAudioLevel(Math.random());
    }, 100);

    setTimeout(() => {
      clearInterval(interval);
      setBotAudioLevel(0);
      // Return to LIVE state (listening) unless we moved to RENDER
      setTurnState(prev => prev === TurnState.BOT_SPEAKING ? TurnState.LIVE : prev);
    }, duration);
  };

  const startInteraction = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      
      // 1. Setup Audio Context for visualization (User Voice)
      audioContextRef.current = new (window.AudioContext || (window as any).webkitAudioContext)();
      const source = audioContextRef.current.createMediaStreamSource(stream);
      analyserRef.current = audioContextRef.current.createAnalyser();
      analyserRef.current.fftSize = 32; // Low res for binary look
      source.connect(analyserRef.current);
      
      const bufferLength = analyserRef.current.frequencyBinCount;
      dataArrayRef.current = new Uint8Array(bufferLength);

      // Start visualization loop
      const updateLevel = () => {
          if (analyserRef.current && dataArrayRef.current) {
              analyserRef.current.getByteFrequencyData(dataArrayRef.current);
              // Get average volume
              let sum = 0;
              for(let i = 0; i < bufferLength; i++) {
                  sum += dataArrayRef.current[i];
              }
              const average = sum / bufferLength;
              setUserAudioLevel(average / 255); // Normalize 0-1
          }
          animationFrameRef.current = requestAnimationFrame(updateLevel);
      };
      updateLevel();

      // 2. Setup Continuous Recorder
      const recorder = new MediaRecorder(stream);
      
      recorder.ondataavailable = (e) => {
        if (e.data.size > 0) {
            // Continuously stream chunks to backend
            mockBackend.sendAudioChunk("mock_base64_data");
        }
      };

      mediaRecorderRef.current = recorder;
      
      // Start recording with 1s timeslices (triggers ondataavailable every 1s)
      recorder.start(1000); 
      
      setAppMode(AppMode.CONVERSATION);
      setTurnState(TurnState.LIVE);

    } catch (err) {
      console.error("Microphone access denied", err);
      alert("Microphone access is required for the investigation.");
    }
  };

  // Interaction UI Logic
  const renderInteractionLayer = () => {
    if (appMode !== AppMode.CONVERSATION) return null;

    // Generate a visual bar string based on user volume
    // e.g. "||||......"
    const totalBars = 20;
    const activeBars = Math.floor(userAudioLevel * totalBars);
    const barString = "I".repeat(activeBars) + ".".repeat(totalBars - activeBars);

    return (
      <div className="absolute bottom-10 left-0 w-full flex flex-col items-center justify-center z-20 gap-2 px-4">
        
        {/* Text-based VU Meter */}
        <div className="font-mono text-green-500 text-sm md:text-base bg-black/80 px-4 py-2 border-l-2 border-r-2 border-green-900">
            INPUT_LEVEL: [{barString}]
        </div>

        {/* Connection Status - Moved below and reduced size (~3x smaller area/visual weight) */}
        <div className="flex items-center gap-2 bg-black/60 px-3 py-1 border border-green-600/40 shadow-[0_0_5px_rgba(0,100,0,0.2)]">
            <div className={`w-1.5 h-1.5 rounded-full shadow-[0_0_4px_currentColor] ${turnState === TurnState.LIVE ? 'bg-red-500 animate-pulse' : 'bg-green-500'}`}></div>
            <span className="font-mono text-[10px] font-bold text-green-400 tracking-widest uppercase">
                {turnState === TurnState.LIVE ? "MIC LIVE - SECURE" : "RECEIVING"}
            </span>
        </div>
        
        <div className="text-[9px] text-green-900 font-mono mt-1 opacity-60">
            DO NOT TURN OFF THE TERMINAL
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