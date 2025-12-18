import React, { useEffect, useState } from 'react';
import DetectiveAvatar from './DetectiveAvatar';
import SketchReveal from './SketchReveal';

interface FinalRenderProps {
  videoUrl?: string;
  imageUrl?: string;
  onRestart: () => void;
}

const FinalRender: React.FC<FinalRenderProps> = ({ videoUrl, imageUrl, onRestart }) => {
  const [showImage, setShowImage] = useState(false);
  const [idleLevel, setIdleLevel] = useState(0.18);

  useEffect(() => {
    if (!imageUrl && !videoUrl) return;
    // Allow a short lead-in before we reveal the staged drawing
    const timer = setTimeout(() => {
      setShowImage(true);
    }, videoUrl ? 2200 : 900);

    return () => clearTimeout(timer);
  }, [imageUrl, videoUrl]);

  useEffect(() => {
    let tick = 0;
    const pulse = setInterval(() => {
      tick += 1;
      setIdleLevel(0.16 + 0.06 * Math.sin(tick * 0.35));
    }, 80);

    return () => clearInterval(pulse);
  }, []);

  return (
    <div className="w-full h-full flex flex-col md:flex-row overflow-hidden relative z-10 bg-black">
      {/* Left Panel: Bot (Static/Neutral) */}
      {/* Reduced height on mobile (h-[20%]) to prioritize the visual evidence */}
      <div className="w-full h-[20%] md:w-[30%] md:h-full border-b md:border-b-0 md:border-r border-green-900 bg-black/90 flex flex-col items-center justify-center relative p-2 md:p-4 shrink-0 z-20">
        <div className="absolute top-2 left-2 text-[10px] md:text-xs text-green-700 font-mono">SYSTEM_STATUS: IDLE</div>
        
        {/* Scale avatar down slightly to ensure it fits comfortably */}
        <div className="scale-75 md:scale-100 transform origin-center mb-2 md:mb-8">
             <DetectiveAvatar isSpeaking={false} audioLevel={idleLevel} compact={true} />
        </div>
        
        <div className="mt-1 md:mt-4 text-green-500 text-[10px] md:text-xs font-mono animate-pulse text-center leading-tight mb-2 md:mb-6">
          CASE FILE #90210-X <br/>
          ANALYSIS COMPLETE
        </div>

        <button 
          onClick={onRestart}
          className="px-4 py-2 border border-green-700 bg-black/50 hover:bg-green-900 hover:text-white text-green-500 text-[10px] md:text-xs font-mono transition-all uppercase tracking-wider"
        >
          [ INITIATE NEW CASE ]
        </button>
      </div>

      {/* Right Panel: Media */}
      <div className="flex-1 h-[80%] md:h-full relative bg-black flex items-center justify-center p-4 md:p-8 overflow-hidden">
        <div className="absolute top-2 right-2 text-[10px] md:text-xs text-green-700 font-mono z-20">RENDER_OUTPUT_BUFFER</div>
        
        {/* CRT Scanline Effect Overlay */}
        <div className="absolute inset-0 z-10 pointer-events-none bg-[url('https://raw.githubusercontent.com/joshbader/scanlines/master/scanlines.png')] opacity-20"></div>

        {/* Media Container:
            aspect-square: Forces 1:1 ratio
            max-w-full / max-h-full: Ensures it never exceeds parent bounds
            w-auto / h-auto: Allows the browser to pick the limiting dimension
        */}
        <div className="relative aspect-square w-full max-w-[min(78vw,78vh)] min-w-[240px] min-h-[240px] border border-green-800 bg-green-900/10 p-1 shadow-[0_0_20px_rgba(0,50,0,0.5)] flex items-center justify-center box-border">
          
          {/* Inner content wrapper to handle absolute positioning of corners properly relative to the square */}
          <div className="relative w-full h-full overflow-hidden">
              {/* Corner Brackets */}
              <div className="absolute top-0 left-0 w-3 h-3 md:w-4 md:h-4 border-t-2 border-l-2 border-green-500 z-30"></div>
              <div className="absolute top-0 right-0 w-3 h-3 md:w-4 md:h-4 border-t-2 border-r-2 border-green-500 z-30"></div>
              <div className="absolute bottom-0 left-0 w-3 h-3 md:w-4 md:h-4 border-b-2 border-l-2 border-green-500 z-30"></div>
              <div className="absolute bottom-0 right-0 w-3 h-3 md:w-4 md:h-4 border-b-2 border-r-2 border-green-500 z-30"></div>

              {showImage && imageUrl ? (
                <SketchReveal imageUrl={imageUrl} />
              ) : (
                <div className="w-full h-full relative overflow-hidden bg-black flex items-center justify-center">
                  {videoUrl ? (
                    <>
                      <img 
                        src={videoUrl} 
                        alt="Generating..." 
                        className="w-full h-full object-cover blur-sm opacity-40 animate-pulse"
                      />
                      <div className="absolute inset-0 bg-gradient-to-b from-black/60 via-transparent to-black/80"></div>
                    </>
                  ) : (
                    <div className="absolute inset-0 bg-[radial-gradient(circle_at_center,_rgba(0,50,0,0.35),_transparent_55%)]"></div>
                  )}
                  <div className="absolute inset-0 flex flex-col items-center justify-center space-y-2">
                    <div className="text-green-500 font-mono text-xs md:text-lg bg-black/80 px-3 py-2 border border-green-900 shadow-lg tracking-[0.2em] text-center">
                      COMPILING FORENSIC LAYERS...
                    </div>
                    <div className="flex space-x-1">
                      <div className="w-2 h-2 bg-green-500 rounded-full animate-bounce" style={{animationDelay: '0ms'}}></div>
                      <div className="w-2 h-2 bg-green-500 rounded-full animate-bounce" style={{animationDelay: '120ms'}}></div>
                      <div className="w-2 h-2 bg-green-500 rounded-full animate-bounce" style={{animationDelay: '240ms'}}></div>
                    </div>
                  </div>
                </div>
              )}
          </div>
          
          {showImage && imageUrl && (
             <div className="absolute -bottom-6 md:-bottom-8 left-0 w-full text-center">
                 <span className="bg-green-900 text-black px-2 py-1 md:px-3 font-bold text-[10px] md:text-sm tracking-widest whitespace-nowrap">MATCH FOUND (99.8%)</span>
             </div>
          )}
        </div>
      </div>
    </div>
  );
};

export default FinalRender;
