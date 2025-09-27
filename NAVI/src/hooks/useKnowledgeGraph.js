import { useState, useCallback } from 'react';
import { useLocalStorage } from './useLocalStorage';

export const useKnowledgeGraph = () => {
  const [knowledgeGraph, setKnowledgeGraph] = useLocalStorage('navi_knowledge_graph', {
    id: 'root',
    title: '我的知识库',
    content: '个人知识图谱根节点',
    children: []
  });

  const addNode = useCallback((parentId, nodeData) => {
    const newNode = {
      id: `node_${Date.now()}`,
      title: nodeData.title || '新知识点',
      content: nodeData.content || '',
      type: nodeData.type || 'manual',
      children: []
    };

    const updateGraph = (node) => {
      if (node.id === parentId) {
        return { ...node, children: [...node.children, newNode] };
      }
      if (node.children) {
        return { ...node, children: node.children.map(updateGraph) };
      }
      return node;
    };

    setKnowledgeGraph(updateGraph);
    return newNode.id;
  }, [setKnowledgeGraph]);

  const updateNode = useCallback((nodeId, updates) => {
    const updateNodeData = (node) => {
      if (node.id === nodeId) {
        return { ...node, ...updates };
      }
      if (node.children) {
        return { ...node, children: node.children.map(updateNodeData) };
      }
      return node;
    };

    setKnowledgeGraph(updateNodeData);
  }, [setKnowledgeGraph]);

  const deleteNode = useCallback((nodeId) => {
    const removeNode = (node) => {
      if (node.children) {
        return { ...node, children: node.children.filter(child => child.id !== nodeId).map(removeNode) };
      }
      return node;
    };

    setKnowledgeGraph(removeNode);
  }, [setKnowledgeGraph]);

  const searchNodes = useCallback((query) => {
    const results = [];

    const search = (node) => {
      if (node.title.toLowerCase().includes(query.toLowerCase()) ||
          node.content.toLowerCase().includes(query.toLowerCase())) {
        results.push(node);
      }
      if (node.children) {
        node.children.forEach(search);
      }
    };

    search(knowledgeGraph);
    return results;
  }, [knowledgeGraph]);

  return {
    knowledgeGraph,
    setKnowledgeGraph,
    addNode,
    updateNode,
    deleteNode,
    searchNodes
  };
};