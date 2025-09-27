import React, { useState } from 'react';
import { GitBranch, Plus, Minus, Search, Edit, Trash2, BookOpen } from 'lucide-react';

const KnowledgeGraph = ({ knowledgeGraph, onUpdateKnowledge, onSwitchToLearning }) => {
  const [expandedNodes, setExpandedNodes] = useState(new Set(['root']));
  const [searchQuery, setSearchQuery] = useState('');
  const [selectedNode, setSelectedNode] = useState(null);

  const toggleNode = (nodeId) => {
    setExpandedNodes(prev => {
      const newSet = new Set(prev);
      if (newSet.has(nodeId)) {
        newSet.delete(nodeId);
      } else {
        newSet.add(nodeId);
      }
      return newSet;
    });
  };

  const addChildNode = (parentId) => {
    const newNode = {
      id: `node_${Date.now()}`,
      title: '新知识点',
      content: '点击编辑内容...',
      type: 'manual',
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

    onUpdateKnowledge(updateGraph);
    setExpandedNodes(prev => new Set([...prev, parentId]));
  };

  const deleteNode = (nodeId) => {
    const removeNode = (node) => {
      if (node.children) {
        return { ...node, children: node.children.filter(child => child.id !== nodeId).map(removeNode) };
      }
      return node;
    };

    onUpdateKnowledge(removeNode);
  };

  const editNode = (nodeId, newTitle, newContent) => {
    const updateNode = (node) => {
      if (node.id === nodeId) {
        return { ...node, title: newTitle, content: newContent };
      }
      if (node.children) {
        return { ...node, children: node.children.map(updateNode) };
      }
      return node;
    };

    onUpdateKnowledge(updateNode);
  };

  const getNodeColor = (type) => {
    switch (type) {
      case 'learning': return 'border-blue-500 bg-blue-500/10';
      case 'questioning': return 'border-red-500 bg-red-500/10';
      case 'manual': return 'border-green-500 bg-green-500/10';
      default: return 'border-yellow-500 bg-yellow-500/10';
    }
  };

  const renderKnowledgeNode = (node, level = 0) => {
    const isExpanded = expandedNodes.has(node.id);
    const hasChildren = node.children && node.children.length > 0;
    const isFiltered = searchQuery && !node.title.toLowerCase().includes(searchQuery.toLowerCase()) && !node.content.toLowerCase().includes(searchQuery.toLowerCase());

    if (isFiltered) return null;

    return (
      <div key={node.id} className="mb-2">
        <div className={`knowledge-node ${getNodeColor(node.type)} ${level > 0 ? `ml-${level * 4}` : ''}`}>
          <div className="node-header">
            <div className="node-controls">
              {hasChildren && (
                <button
                  className="expand-button"
                  onClick={() => toggleNode(node.id)}
                >
                  {isExpanded ? <Minus size={16} /> : <Plus size={16} />}
                </button>
              )}

              <div className="node-info" onClick={() => setSelectedNode(node)}>
                <div className="node-title">{node.title}</div>
                <div className="node-content">{node.content}</div>
              </div>
            </div>

            <div className="node-actions">
              <button
                className="action-button"
                onClick={() => onSwitchToLearning(node.title)}
                title="在学习模式中探索"
              >
                <BookOpen size={14} />
              </button>
              <button
                className="action-button"
                onClick={() => addChildNode(node.id)}
                title="添加子节点"
              >
                <Plus size={14} />
              </button>
              {node.id !== 'root' && (
                <button
                  className="action-button delete"
                  onClick={() => deleteNode(node.id)}
                  title="删除节点"
                >
                  <Trash2 size={14} />
                </button>
              )}
            </div>
          </div>

          {node.type && (
            <div className="node-type">{node.type}</div>
          )}
        </div>

        {isExpanded && hasChildren && (
          <div className="ml-4">
            {node.children.map(child => renderKnowledgeNode(child, level + 1))}
          </div>
        )}
      </div>
    );
  };

  return (
    <div className="knowledge-graph">
      {/* 搜索和控制栏 */}
      <div className="graph-header">
        <div className="search-container">
          <Search className="search-icon" size={18} />
          <input
            type="text"
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
            placeholder="搜索知识节点..."
            className="search-input"
          />
        </div>

        <div className="graph-stats">
          <span className="stat-item">
            <GitBranch size={16} />
            知识图谱
          </span>
        </div>
      </div>

      {/* 知识图谱内容 */}
      <div className="graph-content">
        {renderKnowledgeNode(knowledgeGraph)}
      </div>

      {/* 节点详情弹窗 */}
      {selectedNode && (
        <div className="node-modal" onClick={() => setSelectedNode(null)}>
          <div className="modal-content" onClick={(e) => e.stopPropagation()}>
            <div className="modal-header">
              <h3>编辑知识节点</h3>
              <button onClick={() => setSelectedNode(null)}>×</button>
            </div>

            <div className="modal-body">
              <div className="form-group">
                <label>标题</label>
                <input
                  type="text"
                  defaultValue={selectedNode.title}
                  onBlur={(e) => editNode(selectedNode.id, e.target.value, selectedNode.content)}
                />
              </div>

              <div className="form-group">
                <label>内容</label>
                <textarea
                  defaultValue={selectedNode.content}
                  onBlur={(e) => editNode(selectedNode.id, selectedNode.title, e.target.value)}
                  rows={4}
                />
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

export default KnowledgeGraph;