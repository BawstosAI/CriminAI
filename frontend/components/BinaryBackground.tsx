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

    const drops: number[] = new Array(columns).fill(1);

    let partnerMode = false;
    let activePartnerText = '';
    let partnerTimer = 0;
    let partnerCol = 0;
    let partnerRow = 0;

    let tick = 0;

    const partnerInterval = setInterval(() => {
      partnerMode = true;
      activePartnerText = PARTNERS[Math.floor(Math.random() * PARTNERS.length)];
      partnerTimer = 100;

      const totalCols = Math.ceil(width / fontSize);
      const totalRows = Math.ceil(height / fontSize);

      const safeXStart = Math.floor(totalCols * 0.3);
      const safeXEnd = Math.floor(totalCols * 0.7);
      const safeYStart = Math.floor(totalRows * 0.2);
      const safeYEnd = Math.floor(totalRows * 0.8);

      let c = 0;
      let r = 0;
      let attempts = 0;

      do {
        c = Math.floor(Math.random() * (totalCols - activePartnerText.length - 2)) + 1;
        r = Math.floor(Math.random() * (totalRows - 4)) + 2;

        attempts++;

        const isInSafeZone = (c > safeXStart && c < safeXEnd) && (r > safeYStart && r < safeYEnd);

        if (!isInSafeZone) {
          break;
        }
      } while (attempts < 20);

      partnerCol = c;
      partnerRow = r;

    }, 2000);

    const draw = () => {
      tick += 1;
      ctx.fillStyle = 'rgba(0, 0, 0, 0.05)';
      ctx.fillRect(0, 0, width, height);

      const pulse = 0.5 + 0.5 * Math.sin(tick * 0.02);
      const centerX = width * (0.5 + Math.sin(tick * 0.004) * 0.08);
      const centerY = height * (0.48 + Math.cos(tick * 0.003) * 0.06);
      const radius = Math.min(width, height) * (0.35 + 0.08 * pulse);

      const gradient = ctx.createRadialGradient(centerX, centerY, radius * 0.15, centerX, centerY, radius);
      gradient.addColorStop(0, `rgba(0, 120, 0, ${0.06 + intensity * 0.08})`);
      gradient.addColorStop(1, 'rgba(0, 0, 0, 0)');
      ctx.fillStyle = gradient;
      ctx.fillRect(0, 0, width, height);

      ctx.font = `${fontSize}px monospace`;

      for (let i = 0; i < drops.length; i++) {
        const text = Math.random() > 0.5 ? '1' : '0';

        const opacity = Math.random() * 0.5 + 0.5;
        if (Math.random() > 0.985) {
          ctx.fillStyle = '#fff';
        } else {
          ctx.fillStyle = `rgba(0, 255, 0, ${opacity})`;
        }

        ctx.fillText(text, i * fontSize, drops[i] * fontSize);

        if (drops[i] * fontSize > height && Math.random() > 0.975) {
          drops[i] = 0;
        }

        drops[i]++;
      }

      if (partnerMode && activePartnerText) {
        ctx.font = `bold ${partnerFontSize}px monospace`;
        ctx.fillStyle = '#ccffcc';

        for (let j = 0; j < activePartnerText.length; j++) {
          if (Math.random() > 0.1) {
            const char = activePartnerText[j];
            const x = (partnerCol + j) * fontSize;
            const y = partnerRow * fontSize;

            ctx.shadowColor = '#0f0';
            ctx.shadowBlur = 5;
            ctx.fillText(char, x, y);
            ctx.shadowBlur = 0;

            if (Math.random() > 0.9) {
              ctx.fillStyle = 'rgba(0, 255, 0, 0.4)';
              ctx.fillRect(x, y - partnerFontSize, fontSize, partnerFontSize * 1.2);
              ctx.fillStyle = '#ccffcc';
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
  }, [intensity]);

  return (
    <canvas
      ref={canvasRef}
      className="fixed top-0 left-0 w-full h-full z-0 pointer-events-none opacity-40"
    />
  );
};

export default BinaryBackground;
