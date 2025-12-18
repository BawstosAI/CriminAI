import React, { useEffect, useMemo, useState } from 'react';
import { buildSketchStages, SketchStages } from '../services/imageStages';

interface SketchRevealProps {
  imageUrl?: string;
}

const SketchReveal: React.FC<SketchRevealProps> = ({ imageUrl }) => {
  const [stages, setStages] = useState<SketchStages | null>(null);
  const [processing, setProcessing] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;
    if (!imageUrl) {
      setStages(null);
      return;
    }

    setProcessing(true);
    setError(null);

    buildSketchStages(imageUrl)
      .then((result) => {
        if (!cancelled) {
          setStages(result);
        }
      })
      .catch((err) => {
        if (!cancelled) {
          console.error('SketchReveal: failed to build stages', err);
          setError('Unable to process image');
        }
      })
      .finally(() => {
        if (!cancelled) {
          setProcessing(false);
        }
      });

    return () => {
      cancelled = true;
    };
  }, [imageUrl]);

  const styleBlock = useMemo(
    () => `
      .sketch-layer {
        position: absolute;
        inset: 0;
        width: 100%;
        height: 100%;
        object-fit: contain;
      }

      .sketch-layer.lines {
        clip-path: inset(100% 0 0 0);
        animation: sketch-wipe 2.1s ease-out forwards;
        filter: contrast(1.25) brightness(1.05);
        mix-blend-mode: multiply;
      }

      .sketch-layer.charcoal {
        opacity: 0;
        clip-path: inset(100% 0 0 0);
        animation: sketch-wipe 2s ease-out forwards, sketch-fade 1.2s ease-out forwards;
        animation-delay: 1.4s, 1.4s;
        mix-blend-mode: multiply;
        filter: saturate(0.8);
      }

      .sketch-layer.final {
        opacity: 0;
        animation: sketch-fade 1.5s ease-in forwards;
        animation-delay: 3.2s;
        filter: contrast(1.05) saturate(1.05);
      }

      .sketch-grid {
        position: absolute;
        inset: 0;
        background:
          linear-gradient(rgba(0, 255, 120, 0.03) 1px, transparent 1px),
          linear-gradient(90deg, rgba(0, 255, 120, 0.03) 1px, transparent 1px);
        background-size: 32px 32px, 32px 32px;
        opacity: 0.5;
        mix-blend-mode: overlay;
        pointer-events: none;
      }

      .sketch-hatch {
        position: absolute;
        inset: -20%;
        background-image: repeating-linear-gradient(
          135deg,
          rgba(0, 255, 120, 0.08) 0px,
          rgba(0, 255, 120, 0.08) 2px,
          transparent 2px,
          transparent 8px
        );
        mix-blend-mode: soft-light;
        opacity: 0.25;
        animation: drift 14s linear infinite;
      }

      @keyframes sketch-wipe {
        from { clip-path: inset(100% 0 0 0); }
        to { clip-path: inset(0 0 0 0); }
      }

      @keyframes sketch-fade {
        from { opacity: 0; }
        to { opacity: 1; }
      }

      @keyframes drift {
        from { transform: translateY(0); }
        to { transform: translateY(-14%); }
      }
    `,
    []
  );

  return (
    <div className="relative w-full h-full overflow-hidden bg-gradient-to-b from-white via-[#f5f5f0] to-white">
      <style>{styleBlock}</style>

      <div className="absolute inset-0 bg-[radial-gradient(circle_at_30%_20%,rgba(0,40,0,0.35),transparent_45%)]" />
      <div className="absolute inset-0 bg-[radial-gradient(circle_at_70%_60%,rgba(0,60,0,0.25),transparent_40%)]" />
      <div className="absolute inset-0 pointer-events-none mix-blend-screen opacity-20 bg-[conic-gradient(from_45deg,rgba(0,255,120,0.1),transparent_40%)] animate-pulse" />

      {processing && (
        <div className="absolute top-3 left-3 z-20 text-[10px] md:text-xs font-mono text-green-400 bg-black/70 border border-green-900 px-2 py-1">
          Decomposing sketch layers...
        </div>
      )}

      {error && (
        <div className="absolute inset-0 flex items-center justify-center z-30">
          <div className="bg-red-900/40 border border-red-700 text-red-200 font-mono text-xs px-3 py-2">
            {error}
          </div>
        </div>
      )}

      {stages ? (
        <div className="absolute inset-0 flex items-center justify-center">
          <div className="relative w-full h-full">
            <div className="sketch-grid" />
            <div className="sketch-hatch" />
            <img className="sketch-layer lines" src={stages.stage1} alt="Construction lines" />
            <img className="sketch-layer charcoal" src={stages.stage2} alt="Charcoal layer" />
            <img className="sketch-layer final" src={stages.final} alt="Final portrait" />
          </div>
        </div>
      ) : (
        <div className="absolute inset-0 flex items-center justify-center">
          <div className="text-green-400 font-mono text-xs md:text-sm bg-black/70 px-3 py-2 border border-green-800">
            {processing ? 'Preparing forensic frames...' : 'Image not available'}
          </div>
        </div>
      )}
    </div>
  );
};

export default SketchReveal;
