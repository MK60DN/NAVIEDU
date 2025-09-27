import axios from 'axios';
import { API_CONFIG } from '../utils/constants';

class NaviAPI {
  constructor() {
    this.baseURL = API_CONFIG.BASE_URL;
    this.timeout = API_CONFIG.TIMEOUT;
  }

  async learning(message, context = {}) {
    try {
      const response = await axios.post(`${this.baseURL}/api/learning`, {
        message,
        context: context.context || [],
        knowledge_graph: context.knowledgeGraph
      }, { timeout: this.timeout });

      return response.data;
    } catch (error) {
      throw this.handleError(error);
    }
  }

  async questioning(message, context = {}) {
    try {
      const response = await axios.post(`${this.baseURL}/api/questioning`, {
        message,
        context: context.context || [],
        learning_context: context.learningContext
      }, { timeout: this.timeout });

      return response.data;
    } catch (error) {
      throw this.handleError(error);
    }
  }

  async chat(message, context = {}) {
    try {
      const response = await axios.post(`${this.baseURL}/api/chat`, {
        message,
        context: context.context || []
      }, { timeout: this.timeout });

      return response.data;
    } catch (error) {
      throw this.handleError(error);
    }
  }

  async updateKnowledgeGraph(knowledgeData) {
    try {
      const response = await axios.post(`${this.baseURL}/api/knowledge/update`, knowledgeData, {
        timeout: this.timeout
      });

      return response.data;
    } catch (error) {
      throw this.handleError(error);
    }
  }

  handleError(error) {
    if (error.response) {
      return new Error(`API Error: ${error.response.status} - ${error.response.data.detail || error.response.statusText}`);
    } else if (error.request) {
      return new Error('网络连接失败，请检查网络设置');
    } else {
      return new Error(`请求错误: ${error.message}`);
    }
  }
}

export const naviAPI = new NaviAPI();