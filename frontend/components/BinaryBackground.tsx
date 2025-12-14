import React, { useEffect, useRef } from 'react';

const PARTNERS = ['UNAITE', 'KYUTAI', 'GRADIUM', 'PYANNOTE.AI', 'ITERATE', 'PARETO', 'HEXA', 'FOMO CLUB'];

interface BinaryBackgroundProps {
  intensity: number; // 0 to 1
}

const BinaryBackground: React.FC<BinaryBackgroundProps> = ({ intensity }) => {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    let width = canvas.width = window.innerWidth;
    let height = canvas.height = window.innerHeight;

    const fontSize = 14;
    const partnerFontSize = 18; // Explicit request for partners
    const columns = Math.ceil(width / fontSize);
    
    // Array to store the y-coordinate of the drop for each column
    const drops: number[] = new Array(columns).fill(1);
    
    // Partner logo state
    let partnerMode = false;
    let activePartnerText = '';
    let partnerTimer = 0;
    let partnerCol = 0;
    let partnerRow = 0;
    
    // Trigger partner appearances every 2 seconds
    const partnerInterval = setInterval(() => {
      partnerMode = true;
      activePartnerText = PARTNERS[Math.floor(Math.random() * PARTNERS.length)];
      partnerTimer = 100; // ~1.5s duration at 60fps, clears before next one

      // Calculate safe boundaries to avoid the center (where the detective is)
      const totalCols = Math.ceil(width / fontSize);
      const totalRows = Math.ceil(height / fontSize);
      
      // Define a central "safe zone" to avoid
      const safeXStart = Math.floor(totalCols * 0.3); // 30%
      const safeXEnd = Math.floor(totalCols * 0.7);   // 70%
      const safeYStart = Math.floor(totalRows * 0.2); // 20%
      const safeYEnd = Math.floor(totalRows * 0.8);   // 80%

      let c = 0;
      let r = 0;
      let attempts = 0;
      
      // Try up to 20 times to find a coordinate OUTSIDE the safe zone
      do {
          // Random column (ensure text fits)
          c = Math.floor(Math.random() * (totalCols - activePartnerText.length - 2)) + 1;
          // Random row
          r = Math.floor(Math.random() * (totalRows - 4)) + 2;
          
          attempts++;
          
          // Check if this point falls inside the exclusion zone
          const isInSafeZone = (c > safeXStart && c < safeXEnd) && (r > safeYStart && r < safeYEnd);
          
          if (!isInSafeZone) {
              break; // Found a good spot
          }
      } while (attempts < 20);

      partnerCol = c;
      partnerRow = r;

    }, 2000);

    const draw = () => {
      // Semi-transparent black to create trail effect
      ctx.fillStyle = 'rgba(0, 0, 0, 0.05)';
      ctx.fillRect(0, 0, width, height);

      // 1. Draw the Standard Matrix Rain (Size 14)
      ctx.font = `${fontSize}px monospace`;

      for (let i = 0; i < drops.length; i++) {
        const text = Math.random() > 0.5 ? '1' : '0';
        
        // Standard Rain Color
        const opacity = Math.random() * 0.5 + 0.5;
        if (Math.random() > 0.98) {
            ctx.fillStyle = '#fff'; // Glitch white
        } else {
            ctx.fillStyle = `rgba(0, 255, 0, ${opacity})`;
        }
          
        ctx.fillText(text, i * fontSize, drops[i] * fontSize);

        // Reset drop to top randomly
        if (drops[i] * fontSize > height && Math.random() > 0.975) {
          drops[i] = 0;
        }

        drops[i]++;
      }

      // 2. Draw Partner Logo "Interference" Overlay (Size 20)
      if (partnerMode && activePartnerText) {
          ctx.font = `bold ${partnerFontSize}px monospace`; // Switch to 20px
          ctx.fillStyle = '#ccffcc'; // Bright green
          
          for (let j = 0; j < activePartnerText.length; j++) {
             // "Emerge by density": glitch effect
             if (Math.random() > 0.1) { 
                 const char = activePartnerText[j];
                 // Maintain grid alignment: use the rain grid (14px) for X spacing logic
                 // but draw the character larger. This keeps them "anchored" to the matrix columns.
                 const x = (partnerCol + j) * fontSize;
                 const y = partnerRow * fontSize;
                 
                 // Draw the character
                 ctx.shadowColor = '#0f0';
                 ctx.shadowBlur = 5;
                 ctx.fillText(char, x, y);
                 ctx.shadowBlur = 0; // Reset
                 
                 // Occasionally draw a "scanline" or noise around it
                 if (Math.random() > 0.9) {
                    ctx.fillStyle = 'rgba(0, 255, 0, 0.4)';
                    // Adjust interference box for larger font
                    ctx.fillRect(x, y - partnerFontSize, fontSize, partnerFontSize * 1.2);
                    ctx.fillStyle = '#ccffcc'; // Reset fill
                 }
             }
          }
          
          partnerTimer--;
          if (partnerTimer <= 0) {
            partnerMode = false;
          }
      }
    };

    let animationFrameId: number;
    const renderLoop = () => {
      draw();
      animationFrameId = requestAnimationFrame(renderLoop);
    };
    renderLoop();

    const handleResize = () => {
      width = canvas.width = window.innerWidth;
      height = canvas.height = window.innerHeight;
    };
    window.addEventListener('resize', handleResize);

    return () => {
      window.removeEventListener('resize', handleResize);
      clearInterval(partnerInterval);
      cancelAnimationFrame(animationFrameId);
    };
  }, []);

  return (
    <canvas 
      ref={canvasRef} 
      className="fixed top-0 left-0 w-full h-full z-0 pointer-events-none opacity-40" 
    />
  );
};

export default BinaryBackground;