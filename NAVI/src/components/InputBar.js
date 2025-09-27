import React, { useRef, useState } from 'react';
import { Send, Mic, Camera, FileText } from 'lucide-react';

const InputBar = ({
  value,
  onChange,
  onSend,
  placeholder,
  showVoice = true,
  showCamera = true,
  showFile = true
}) => {
  const [isRecording, setIsRecording] = useState(false);
  const fileInputRef = useRef(null);

  const handleKeyPress = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      onSend();
    }
  };

  const toggleVoiceRecording = () => {
    setIsRecording(!isRecording);
    // 这里集成语音识别
  };

  const handleFileUpload = () => {
    fileInputRef.current?.click();
  };

  const handleFileSelect = (event) => {
    const file = event.target.files[0];
    if (file) {
      // 处理文件上传
      console.log('文件上传:', file.name);
    }
  };

  return (
    <div className="input-bar">
      {/* 多媒体按钮 */}
      <div className="media-buttons">
        {showVoice && (
          <button
            className={`media-button ${isRecording ? 'recording' : ''}`}
            onClick={toggleVoiceRecording}
          >
            <Mic size={18} />
          </button>
        )}

        {showCamera && (
          <button className="media-button" onClick={handleFileUpload}>
            <Camera size={18} />
          </button>
        )}

        {showFile && (
          <button className="media-button">
            <FileText size={18} />
          </button>
        )}
      </div>

      {/* 输入区域 */}
      <div className="input-container">
        <textarea
          value={value}
          onChange={(e) => onChange(e.target.value)}
          placeholder={placeholder}
          className="input-textarea"
          onKeyPress={handleKeyPress}
          rows={1}
        />
        <button
          onClick={onSend}
          className="send-button"
          disabled={!value.trim()}
        >
          <Send size={18} />
        </button>
      </div>

      <input
        ref={fileInputRef}
        type="file"
        accept="image/*"
        className="hidden"
        onChange={handleFileSelect}
      />
    </div>
  );
};

export default InputBar;