import React, { useRef, useState, useEffect } from 'react';
import { useI18n } from '../lib/i18n';

interface Props {
  imageUrl: string | null;
  onMaskChange: (maskBlob: Blob | null) => void;
}

export function MaskEditor({ imageUrl, onMaskChange }: Props) {
  const { t } = useI18n();
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const containerRef = useRef<HTMLDivElement>(null);
  const [isDrawing, setIsDrawing] = useState(false);
  const [brushSize, setBrushSize] = useState(40);
  const [imageObj, setImageObj] = useState<HTMLImageElement | null>(null);

  useEffect(() => {
    if (!imageUrl) {
      setImageObj(null);
      return;
    }

    setImageObj(null);
    let cancelled = false;
    const img = new Image();
    img.onload = () => {
      if (!cancelled) setImageObj(img);
    };
    img.onerror = () => {
      if (!cancelled) setImageObj(null);
    };
    img.src = imageUrl;

    return () => {
      cancelled = true;
      img.onload = null;
      img.onerror = null;
    };
  }, [imageUrl]);

  useEffect(() => {
    if (!imageObj || !canvasRef.current || !containerRef.current) return;
    const canvas = canvasRef.current;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    // Set canvas resolution to match image native resolution for backend
    canvas.width = imageObj.naturalWidth;
    canvas.height = imageObj.naturalHeight;
    
    // Clear mask
    ctx.fillStyle = 'black';
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    onMaskChange(null);
  }, [imageObj]); // Only run on image load

  const getCoordinates = (e: React.MouseEvent | React.TouchEvent) => {
    if (!canvasRef.current) return null;
    const canvas = canvasRef.current;
    const rect = canvas.getBoundingClientRect();
    
    let clientX, clientY;
    if ('touches' in e) {
      clientX = e.touches[0].clientX;
      clientY = e.touches[0].clientY;
    } else {
      clientX = (e as React.MouseEvent).clientX;
      clientY = (e as React.MouseEvent).clientY;
    }

    // Map screen coordinates to internal canvas resolution
    const scaleX = canvas.width / rect.width;
    const scaleY = canvas.height / rect.height;
    return {
      x: (clientX - rect.left) * scaleX,
      y: (clientY - rect.top) * scaleY
    };
  };

  const startDrawing = (e: React.MouseEvent | React.TouchEvent) => {
    e.preventDefault();
    setIsDrawing(true);
    paintAt(e);
  };

  const stopDrawing = () => {
    if (!isDrawing) return;
    setIsDrawing(false);
    exportMask();
  };

  const draw = (e: React.MouseEvent | React.TouchEvent) => {
    if (!isDrawing || !canvasRef.current) return;
    e.preventDefault();
    paintAt(e);
  };

  const paintAt = (e: React.MouseEvent | React.TouchEvent) => {
    if (!canvasRef.current) return;
    const coords = getCoordinates(e);
    if (!coords) return;

    const ctx = canvasRef.current.getContext('2d');
    if (!ctx) return;

    // Draw white for mask
    ctx.fillStyle = 'white';
    ctx.beginPath();
    // Scale brush size by canvas scale to feel consistent
    const rect = canvasRef.current.getBoundingClientRect();
    const scale = canvasRef.current.width / rect.width;
    ctx.arc(coords.x, coords.y, (brushSize * scale) / 2, 0, Math.PI * 2);
    ctx.fill();
  };

  const exportMask = () => {
    if (!canvasRef.current) return;
    canvasRef.current.toBlob((blob) => {
      onMaskChange(blob);
    }, 'image/png');
  };

  const clearMask = () => {
    if (!canvasRef.current) return;
    const ctx = canvasRef.current.getContext('2d');
    if (!ctx) return;
    ctx.fillStyle = 'black';
    ctx.fillRect(0, 0, canvasRef.current.width, canvasRef.current.height);
    onMaskChange(null);
  };

  if (!imageObj) return <div className="w-full h-64 bg-bg-soft animate-pulse rounded-2xl"></div>;

  return (
    <div className="w-full flex flex-col gap-4">
      <div className="flex items-center justify-between">
        <label className="text-sm font-semibold text-text-main">{t.drawMask}</label>
        <button onClick={clearMask} className="text-sm text-primary font-medium hover:text-primary-hover">
          {t.clearMask}
        </button>
      </div>

      <div 
        ref={containerRef}
        className="relative w-full rounded-2xl overflow-hidden bg-bg-soft border border-border touch-none"
        style={{ aspectRatio: `${imageObj.naturalWidth} / ${imageObj.naturalHeight}` }}
      >
        <img 
          src={imageUrl ?? undefined}
          className="absolute inset-0 w-full h-full object-contain"
          alt={t.sourceImage}
          draggable="false"
        />
        <canvas
          ref={canvasRef}
          onMouseDown={startDrawing}
          onMouseMove={draw}
          onMouseUp={stopDrawing}
          onMouseLeave={stopDrawing}
          onTouchStart={startDrawing}
          onTouchMove={draw}
          onTouchEnd={stopDrawing}
          className="absolute inset-0 w-full h-full object-contain opacity-50 cursor-crosshair mix-blend-screen"
        />
      </div>

      <div className="flex items-center gap-4">
        <label className="text-sm text-text-muted whitespace-nowrap">{t.brushSize}</label>
        <input 
          type="range" 
          min="10" 
          max="100" 
          value={brushSize} 
          onChange={(e) => setBrushSize(parseInt(e.target.value))}
          className="w-full accent-primary"
        />
      </div>
    </div>
  );
}
