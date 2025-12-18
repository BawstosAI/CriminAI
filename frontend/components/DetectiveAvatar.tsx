import React, { useEffect, useRef, useState } from 'react';

interface DetectiveAvatarProps {
  isSpeaking: boolean;
  audioLevel: number; // 0 to 1
  compact?: boolean; // For the side-view in render phase
}

const DetectiveAvatar: React.FC<DetectiveAvatarProps> = ({ isSpeaking, audioLevel, compact = false }) => {
  const [grid, setGrid] = useState<string[][]>([]);
  const [isBlinking, setIsBlinking] = useState(false);
  const containerRef = useRef<HTMLDivElement>(null);
  const isSpeakingRef = useRef(isSpeaking);
  const audioLevelRef = useRef(audioLevel);
  const blinkRef = useRef(false);

  const rows = compact ? 40 : 64;
  const cols = compact ? 60 : 100;

  const noise = (r: number, c: number, t: number) => {
    const value = Math.sin((r * 12.9898 + c * 78.233 + t * 0.35) * 0.9) * 43758.5453;
    return value - Math.floor(value);
  };

  useEffect(() => {
    let timeoutId: number;
    const blink = () => {
      setIsBlinking(true);
      setTimeout(() => setIsBlinking(false), 140);
      timeoutId = window.setTimeout(blink, 2000 + Math.random() * 3000);
    };
    timeoutId = window.setTimeout(blink, 1200 + Math.random() * 1200);
    return () => clearTimeout(timeoutId);
  }, []);

  useEffect(() => {
    isSpeakingRef.current = isSpeaking;
  }, [isSpeaking]);

  useEffect(() => {
    audioLevelRef.current = audioLevel;
  }, [audioLevel]);

  useEffect(() => {
    blinkRef.current = isBlinking;
  }, [isBlinking]);

  const getCharacter = (r: number, c: number, tick: number, speakLevel: number) => {
    const time = tick * 0.05;
    const y = r / rows;
    const x = c / cols;

    const swayX = Math.sin(time * 0.6) * 0.015;
    const swayY = Math.sin(time * 0.45) * 0.01;

    const cx = 0.5 + swayX;
    const dx = x - cx;

    const hatCrown = y > 0.1 + swayY && y < 0.32 + swayY && Math.abs(dx) < 0.14 * (1 + (y - 0.1));
    const brimCurve = 0.32 + swayY + (dx * dx) * 0.5;
    const hatBrim = y > brimCurve && y < brimCurve + 0.04 && Math.abs(dx) < 0.28;

    const collarLeft = x > 0.32 + swayX && x < 0.42 + swayX && y > 0.52 + swayY && y < 0.65 + swayY;
    const collarRight = x > 0.58 + swayX && x < 0.68 + swayX && y > 0.52 + swayY && y < 0.65 + swayY;

    const shoulderSlope = 0.55 + Math.abs(dx) * 0.8;
    const body = y > shoulderSlope && Math.abs(dx) < 0.45;

    const faceArea = y > 0.32 + swayY && y < 0.58 + swayY && Math.abs(dx) < 0.17;

    const eyeJitterX = (noise(r + 3, c + 9, tick * 0.06) - 0.5) * (compact ? 0.01 : 0.015);
    const eyeJitterY = (noise(r + 7, c + 5, tick * 0.06) - 0.5) * (compact ? 0.008 : 0.012);
    const eyeY = 0.41 + swayY + eyeJitterY;
    const eyeXSpacing = 0.06;
    const eyeWidth = 0.024;
    const eyeHeight = blinkRef.current ? 0.004 : 0.016;

    const leftEye = Math.abs(dx + eyeXSpacing + eyeJitterX) < eyeWidth && Math.abs(y - eyeY) < eyeHeight;
    const rightEye = Math.abs(dx - eyeXSpacing + eyeJitterX) < eyeWidth && Math.abs(y - eyeY) < eyeHeight;

    if (faceArea && (leftEye || rightEye)) {
      if (blinkRef.current) return '-';
      const pupil = Math.abs(dx + (leftEye ? eyeXSpacing : -eyeXSpacing) + eyeJitterX) < 0.008 && Math.abs(y - eyeY) < 0.008;
      return pupil ? '1' : '0';
    }

    const mouthCenterY = 0.5 + swayY;
    const mouthLevel = isSpeakingRef.current ? speakLevel : 0;
    const mouthActive = isSpeakingRef.current && mouthLevel > 0.01;
    const waveAmplitude = mouthActive ? (0.01 + mouthLevel * 0.08) : 0;
    const waveThickness = mouthActive ? (0.005 + mouthLevel * 0.045) : 0;
    const lobes = 4.5;

    let inMouth = false;
    if (mouthActive && faceArea && y > 0.44 && y < 0.58) {
      const normalizedX = (dx + 0.18) / 0.36;
      const wave = Math.sin(normalizedX * Math.PI * lobes + time * 2.6);
      const yWave = mouthCenterY + wave * waveAmplitude;
      if (Math.abs(y - yWave) < waveThickness) {
        inMouth = true;
      } else if (normalizedX < 0.08 || normalizedX > 0.92) {
        const capWidth = 0.018 + mouthLevel * 0.03;
        if (Math.abs(x - (cx + (normalizedX < 0.5 ? -0.18 : 0.18))) < capWidth && Math.abs(y - mouthCenterY) < waveThickness * 1.4) {
          inMouth = true;
        }
      }
    }

    if (inMouth) {
      return noise(r + 11, c + 7, tick) > 0.5 ? '1' : '0';
    }

    let probability = 0;
    if (hatCrown) probability = 0.85;
    if (hatBrim) probability = 0.95;
    if (collarLeft || collarRight) probability = 0.8;
    if (body) {
      probability = 0.9 - y * 0.5;
      if (Math.abs(dx) > 0.3) probability *= 0.3;
    }

    if (faceArea) probability = 0.1;

    const edgeNoise = Math.abs(dx) * 0.5;
    const grain = noise(r, c, tick) + edgeNoise * 0.2;
    const charNoise = noise(r + 5, c + 3, tick * 0.25);

    if (grain < probability) {
      if (probability > 0.8) return charNoise > 0.5 ? '1' : '0';
      if (probability > 0.5) return charNoise > 0.5 ? '1' : ':';
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
          rowArr.push(getCharacter(r, c, tick, audioLevelRef.current));
        }
        newGrid.push(rowArr);
      }
      setGrid(newGrid);
    }, 90);

    return () => clearInterval(interval);
  }, [rows, cols]);

  return (
    <div
      ref={containerRef}
      className={`flex flex-col items-center justify-center font-mono leading-[0.8] select-none text-green-500 overflow-hidden mix-blend-screen
        ${compact ? 'text-[6px] sm:text-[8px]' : 'text-[8px] sm:text-[10px] md:text-[12px]'}`}
      style={{ textShadow: isSpeaking ? '0 0 6px rgba(0, 255, 0, 0.6)' : '0 0 3px rgba(0, 255, 0, 0.25)' }}
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
