import React from 'react';
import { useI18n } from '../lib/i18n';
import { IconUpload } from './Icons';

interface Props {
  file: File | null;
  onFileSelect: (file: File | null) => void;
  label?: string;
  description?: string;
}

export function ImageUploader({ file, onFileSelect, label, description }: Props) {
  const { t } = useI18n();
  const [previewUrl, setPreviewUrl] = React.useState<string | null>(null);
  const [isDragging, setIsDragging] = React.useState(false);
  const inputRef = React.useRef<HTMLInputElement>(null);

  React.useEffect(() => {
    if (file) {
      const url = URL.createObjectURL(file);
      setPreviewUrl(url);
      return () => URL.revokeObjectURL(url);
    } else {
      setPreviewUrl(null);
    }
  }, [file]);

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(false);
    if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
      onFileSelect(e.dataTransfer.files[0]);
    }
  };

  const handleDragOver = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(true);
  };

  const handleDragLeave = () => {
    setIsDragging(false);
  };

  return (
    <div className="w-full">
      {label && <label className="block text-sm font-semibold text-text-main mb-2">{label}</label>}
      {description && <p className="text-sm text-text-muted mb-4">{description}</p>}
      
      {!file ? (
        <div
          onClick={() => inputRef.current?.click()}
          onDrop={handleDrop}
          onDragOver={handleDragOver}
          onDragLeave={handleDragLeave}
          className={`relative flex flex-col items-center justify-center p-8 rounded-2xl border-2 border-dashed cursor-pointer transition-colors ${
            isDragging ? 'border-primary bg-surface-lavender' : 'border-border bg-surface hover:border-primary-light hover:bg-bg-soft'
          }`}
        >
          <input
            ref={inputRef}
            type="file"
            accept="image/jpeg,image/png,image/webp"
            className="hidden"
            onChange={(e) => {
              if (e.target.files && e.target.files.length > 0) {
                onFileSelect(e.target.files[0]);
              }
            }}
          />
          <div className="p-4 bg-bg-soft rounded-full mb-4">
            <IconUpload className="w-8 h-8 text-primary" />
          </div>
          <p className="font-semibold text-text-main text-center mb-1">{t.dropHere}</p>
          <p className="text-sm text-text-muted text-center mb-4">{t.orSelect}</p>
          <p className="text-xs text-text-muted text-center">{t.supportedFormats}</p>
        </div>
      ) : (
        <div className="relative rounded-2xl border border-border bg-surface overflow-hidden group">
          <img src={previewUrl!} alt={label ?? t.roomPhoto} className="w-full max-h-[400px] object-contain bg-bg-soft" />
          <div className="absolute inset-0 bg-black/40 opacity-0 group-hover:opacity-100 transition-opacity flex items-center justify-center gap-4">
            <button
              onClick={() => inputRef.current?.click()}
              className="px-4 py-2 bg-white text-text-main rounded-lg font-medium hover:bg-bg-soft transition-colors text-sm"
            >
              {t.changeImage}
            </button>
            <button
              onClick={() => onFileSelect(null)}
              className="px-4 py-2 bg-error text-white rounded-lg font-medium hover:bg-red-600 transition-colors text-sm"
            >
              {t.removeImage}
            </button>
          </div>
          <input
            ref={inputRef}
            type="file"
            accept="image/jpeg,image/png,image/webp"
            className="hidden"
            onChange={(e) => {
              if (e.target.files && e.target.files.length > 0) {
                onFileSelect(e.target.files[0]);
              }
            }}
          />
        </div>
      )}
    </div>
  );
}
