import { useCallback } from 'react';
import axios from 'axios';

export const useNaviAPI = () => {
  // 清理消息格式，确保符合后端要求
  const cleanMessages = (messages) => {
    if (!messages || !Array.isArray(messages)) return [];

    return messages.map(msg => ({
      content: msg.content || '',
      sender: msg.sender || 'user',
      timestamp: msg.timestamp ? msg.timestamp.toISOString() : new Date().toISOString()
    }));
  };

  const callLearningAPI = useCallback(async (message, context = {}) => {
    try {
      console.log('发送学习API请求:', message);

      const requestData = {
        message: String(message), // 确保是字符串
        context: cleanMessages(context.context || []),
        knowledge_graph: context.knowledgeGraph || null
      };

      console.log('请求数据:', requestData);

      const response = await axios.post('http://localhost:8000/api/learning', requestData);

      console.log('学习API响应:', response.data);
      return response.data;
    } catch (error) {
      console.error('Learning API error:', error);
      console.error('Error response:', error.response?.data);
      throw error;
    }
  }, []);

  const callQuestioningAPI = useCallback(async (message, context = {}) => {
    try {
      console.log('发送质疑API请求:', message);

      const requestData = {
        message: String(message),
        context: cleanMessages(context.context || []),
        learning_context: cleanMessages(context.learningContext || [])
      };

      const response = await axios.post('http://localhost:8000/api/questioning', requestData);

      console.log('质疑API响应:', response.data);
      return response.data;
    } catch (error) {
      console.error('Questioning API error:', error);
      console.error('Error response:', error.response?.data);
      throw error;
    }
  }, []);

  const callChatAPI = useCallback(async (message, context = {}) => {
    try {
      console.log('发送对话API请求:', message);

      const requestData = {
        message: String(message),
        context: cleanMessages(context.context || [])
      };

      const response = await axios.post('http://localhost:8000/api/chat', requestData);

      console.log('对话API响应:', response.data);
      return response.data;
    } catch (error) {
      console.error('Chat API error:', error);
      console.error('Error response:', error.response?.data);
      throw error;
    }
  }, []);

  return { callLearningAPI, callQuestioningAPI, callChatAPI };
};