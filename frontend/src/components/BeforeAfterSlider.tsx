import { useState, useRef, useEffect } from 'react';
import { useI18n } from '../lib/i18n';

interface Props {
  beforeImage: string;
  afterImage: string;
}

export function BeforeAfterSlider({ beforeImage, afterImage }: Props) {
  const { t, dir } = useI18n();
  const [position, setPosition] = useState(50);
  const containerRef = useRef<HTMLDivElement>(null);
  const [isDragging, setIsDragging] = useState(false);

  const handleMove = (clientX: number) => {
    if (!containerRef.current) return;
    const rect = containerRef.current.getBoundingClientRect();
    const x = Math.max(0, Math.min(clientX - rect.left, rect.width));
    const percent = Math.max(0, Math.min((x / rect.width) * 100, 100));
    setPosition(dir === 'rtl' ? 100 - percent : percent);
  };

  const handleMouseMove = (e: MouseEvent) => {
    if (isDragging) handleMove(e.clientX);
  };
  
  const handleTouchMove = (e: TouchEvent) => {
    if (isDragging) handleMove(e.touches[0].clientX);
  };

  const handleMouseUp = () => setIsDragging(false);

  useEffect(() => {
    if (isDragging) {
      window.addEventListener('mousemove', handleMouseMove);
      window.addEventListener('mouseup', handleMouseUp);
      window.addEventListener('touchmove', handleTouchMove);
      window.addEventListener('touchend', handleMouseUp);
    }
    return () => {
      window.removeEventListener('mousemove', handleMouseMove);
      window.removeEventListener('mouseup', handleMouseUp);
      window.removeEventListener('touchmove', handleTouchMove);
      window.removeEventListener('touchend', handleMouseUp);
    };
  }, [isDragging]);

  const clipPathStyle = dir === 'rtl' 
    ? `polygon(${100 - position}% 0, 100% 0, 100% 100%, ${100 - position}% 100%)`
    : `polygon(0 0, ${position}% 0, ${position}% 100%, 0 100%)`;

  return (
    <div 
      className="relative w-full overflow-hidden rounded-2xl bg-bg-soft touch-none select-none"
      ref={containerRef}
      onMouseDown={(e) => {
        setIsDragging(true);
        handleMove(e.clientX);
      }}
      onTouchStart={(e) => {
        setIsDragging(true);
        handleMove(e.touches[0].clientX);
      }}
    >
      <img src={beforeImage} alt={t.before} className="w-full object-cover max-h-[70vh] block" />
      
      <div 
        className="absolute top-0 bottom-0 pointer-events-none"
        style={{
          left: dir === 'rtl' ? 'auto' : 0,
          right: dir === 'rtl' ? 0 : 'auto',
          width: '100%',
          clipPath: clipPathStyle,
        }}
      >
        <img src={afterImage} alt={t.after} className="absolute top-0 left-0 w-full h-full object-cover" />
      </div>

      <div 
        className="absolute top-0 bottom-0 w-1 bg-white cursor-ew-resize shadow-[0_0_10px_rgba(0,0,0,0.5)]"
        style={{ 
          left: dir === 'rtl' ? `${100 - position}%` : `${position}%`, 
          transform: 'translateX(-50%)' 
        }}
      >
        <div className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 w-8 h-8 bg-white rounded-full shadow-md flex items-center justify-center">
          <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" className="text-text-main">
            <polyline points="15 18 9 12 15 6"/>
          </svg>
          <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" className="text-text-main transform rotate-180">
            <polyline points="15 18 9 12 15 6"/>
          </svg>
        </div>
      </div>

      <div className="absolute top-4 left-4 bg-black/50 backdrop-blur-sm text-white px-3 py-1 rounded-full text-xs font-medium">
        {t.before}
      </div>
      <div className="absolute top-4 right-4 bg-primary/80 backdrop-blur-sm text-white px-3 py-1 rounded-full text-xs font-medium">
        {t.after}
      </div>
    </div>
  );
}
