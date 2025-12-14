import React, { useEffect, useState } from 'react';

interface DetectiveAvatarProps {
  isSpeaking: boolean;
  audioLevel: number; // 0 to 1
  compact?: boolean; // For the side-view in render phase
}

const DetectiveAvatar: React.FC<DetectiveAvatarProps> = ({ isSpeaking, audioLevel, compact = false }) => {
  const [grid, setGrid] = useState<string[][]>([]);
  
  // Increased resolution for better definition
  const rows = compact ? 40 : 64;
  const cols = compact ? 60 : 100;

  // Function to calculate if a pixel should be 'on' based on a "Noir" silhouette
  const getCharacter = (r: number, c: number, tick: number, speakLevel: number) => {
    // Normalized coordinates (0 to 1)
    const y = r / rows;
    const x = c / cols;
    
    // Center logic
    const cx = 0.5;
    const dx = x - cx;
    
    // --- SHAPE DEFINITIONS (SDF - Signed Distance Fieldsish) ---

    // 1. Fedora Hat
    // Crown: Rounded top, slightly asymmetrical
    const hatCrown = (y > 0.1 && y < 0.32) && (Math.abs(dx) < 0.14 * (1 + (y-0.1)));
    // Brim: Curved line
    const brimCurve = 0.32 + (dx * dx) * 0.5; // Parabola
    const hatBrim = y > brimCurve && y < brimCurve + 0.04 && Math.abs(dx) < 0.28;

    // 2. The Coat / Shoulders (Mystery Figure)
    // High collar effect
    const collarLeft = x > 0.32 && x < 0.42 && y > 0.52 && y < 0.65; 
    const collarRight = x > 0.58 && x < 0.68 && y > 0.52 && y < 0.65;
    
    // Broad shoulders sloping down
    const shoulderSlope = 0.55 + Math.abs(dx) * 0.8;
    const body = y > shoulderSlope && Math.abs(dx) < 0.45;

    // 3. The Face (The Void)
    // We want the face to be mostly empty (shadow), defined by the hat and collar
    const faceArea = y > 0.35 && y < 0.55 && Math.abs(dx) < 0.15;

    // --- DYNAMIC AUDIO REACTION ---
    
    let probability = 0;

    // Base density for the silhouette
    if (hatCrown) probability = 0.85;
    if (hatBrim) probability = 0.95;
    if (collarLeft || collarRight) probability = 0.8;
    if (body) {
        // Fade out towards the bottom (Matrix rain effect)
        probability = 0.9 - (y * 0.5); 
        // Dither edges
        if (Math.abs(dx) > 0.3) probability *= 0.3; 
    }

    // Face is mostly hidden in shadow
    if (faceArea) probability = 0.05; 

    // --- SPEAKING ANIMATION ---
    // Instead of a mouth, the code intensifies in the face area
    if (isSpeaking && faceArea && y > 0.42 && y < 0.52) {
        // The "Mouth" is a horizontal band of interference
        const mouthWidth = 0.05 + (speakLevel * 0.15); // Width expands with volume
        if (Math.abs(dx) < mouthWidth) {
             probability = 0.8 + (Math.random() * 0.2);
             // Glitch effect: sometimes invert or clear
             if (Math.random() < 0.1) probability = 0;
        }
    }

    // Add general "static" noise to the whole figure to make it alive
    // Higher noise at the edges
    const edgeNoise = Math.abs(dx) * 0.5;
    
    // Final check
    const randomThreshold = Math.random() + (edgeNoise * 0.2); // harder to spawn at edges
    
    if (randomThreshold < probability) {
        // Character selection: 
        // Denser parts get "0", "1", "8", "B". Lighter parts get ".", ":", ","
        if (probability > 0.8) return Math.random() > 0.5 ? '1' : '0';
        if (probability > 0.5) return Math.random() > 0.5 ? '1' : ':';
        return '.';
    }
    
    return ' ';
  };

  useEffect(() => {
    let tick = 0;
    const interval = setInterval(() => {
      tick++;
      const newGrid: string[][] = [];
      
      for (let r = 0; r < rows; r++) {
        const rowArr: string[] = [];
        for (let c = 0; c < cols; c++) {
           rowArr.push(getCharacter(r, c, tick, audioLevel));
        }
        newGrid.push(rowArr);
      }
      setGrid(newGrid);
    }, 60); // 16fps approx for a more cinematic/CRT feel

    return () => clearInterval(interval);
  }, [rows, cols, isSpeaking, audioLevel]);

  return (
    <div className={`flex flex-col items-center justify-center font-mono leading-[0.8] select-none text-green-500 overflow-hidden mix-blend-screen
        ${compact ? 'text-[6px] sm:text-[8px]' : 'text-[8px] sm:text-[10px] md:text-[12px]'}`}
        style={{ textShadow: isSpeaking ? '0 0 4px rgba(0, 255, 0, 0.5)' : 'none' }}
    >
      {grid.map((row, rIdx) => (
        <div key={rIdx} className="whitespace-pre">
          {row.join('')}
        </div>
      ))}
    </div>
  );
};

export default DetectiveAvatar;