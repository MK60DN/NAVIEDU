class VoiceService {
  constructor() {
    this.recognition = null;
    this.synthesis = window.speechSynthesis;
    this.isListening = false;
    this.onResult = null;
    this.onError = null;
  }

  // 初始化语音识别
  initRecognition() {
    if ('webkitSpeechRecognition' in window) {
      this.recognition = new window.webkitSpeechRecognition();
    } else if ('SpeechRecognition' in window) {
      this.recognition = new window.SpeechRecognition();
    } else {
      console.warn('浏览器不支持语音识别');
      return false;
    }

    this.recognition.continuous = false;
    this.recognition.interimResults = false;
    this.recognition.lang = 'zh-CN';

    this.recognition.onstart = () => {
      this.isListening = true;
    };

    this.recognition.onresult = (event) => {
      const result = event.results[0][0].transcript;
      if (this.onResult) {
        this.onResult(result);
      }
    };

    this.recognition.onerror = (event) => {
      this.isListening = false;
      if (this.onError) {
        this.onError(event.error);
      }
    };

    this.recognition.onend = () => {
      this.isListening = false;
    };

    return true;
  }

  // 开始语音识别
  startListening(onResult, onError) {
    if (!this.recognition && !this.initRecognition()) {
      if (onError) onError('浏览器不支持语音识别');
      return false;
    }

    this.onResult = onResult;
    this.onError = onError;

    try {
      this.recognition.start();
      return true;
    } catch (error) {
      if (onError) onError('语音识别启动失败');
      return false;
    }
  }

  // 停止语音识别
  stopListening() {
    if (this.recognition && this.isListening) {
      this.recognition.stop();
    }
  }

  // 语音播放
  speak(text, options = {}) {
    if (!this.synthesis) {
      console.warn('浏览器不支持语音合成');
      return false;
    }

    // 停止当前播放
    this.synthesis.cancel();

    const utterance = new SpeechSynthesisUtterance(text);
    utterance.lang = options.lang || 'zh-CN';
    utterance.rate = options.rate || 1;
    utterance.pitch = options.pitch || 1;
    utterance.volume = options.volume || 1;

    this.synthesis.speak(utterance);
    return true;
  }

  // 停止语音播放
  stopSpeaking() {
    if (this.synthesis) {
      this.synthesis.cancel();
    }
  }

  // 获取可用语音列表
  getVoices() {
    if (!this.synthesis) return [];
    return this.synthesis.getVoices();
  }

  // 检查语音功能是否可用
  isSupported() {
    return {
      recognition: 'webkitSpeechRecognition' in window || 'SpeechRecognition' in window,
      synthesis: 'speechSynthesis' in window
    };
  }
}

export const voiceService = new VoiceService();