import React, { useState, useMemo } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { 
  ChevronLeftIcon, 
  ChevronRightIcon,
  MagnifyingGlassIcon,
  TagIcon,
  BrainIcon,
  CloudIcon,
  CogIcon,
  ChartBarIcon,
  DocumentTextIcon,
  ServerIcon
} from '@heroicons/react/24/outline';
import { Badge } from '../UI/Badge';
import { Tooltip } from '../UI/Tooltip';

interface NodeTemplate {
  id: string;
  type: string;
  label: string;
  description: string;
  category: string;
  icon: React.ComponentType<any>;
  color: string;
  tags: string[];
  isPopular?: boolean;
  isNew?: boolean;
}

interface NodeCategory {
  id: string;
  label: string;
  icon: React.ComponentType<any>;
  color: string;
}

const nodeCategories: NodeCategory[] = [
  { id: 'all', label: 'All Nodes', icon: TagIcon, color: '#6b7280' },
  { id: 'data', label: 'Data Sources', icon: ServerIcon, color: '#3b82f6' },
  { id: 'ml', label: 'Machine Learning', icon: BrainIcon, color: '#8b5cf6' },
  { id: 'transform', label: 'Transform', icon: CogIcon, color: '#10b981' },
  { id: 'deploy', label: 'Deploy', icon: CloudIcon, color: '#f59e0b' },
  { id: 'monitor', label: 'Monitor', icon: ChartBarIcon, color: '#ef4444' },
];

const nodeTemplates: NodeTemplate[] = [
  // Data Sources
  {
    id: 'csv-loader',
    type: 'data-source',
    label: 'CSV Loader',
    description: 'Load data from CSV files',
    category: 'data',
    icon: DocumentTextIcon,
    color: '#3b82f6',
    tags: ['data', 'import', 'csv'],
    isPopular: true,
  },
  {
    id: 'database-connector',
    type: 'database',
    label: 'Database',
    description: 'Connect to SQL databases',
    category: 'data',
    icon: ServerIcon,
    color: '#3b82f6',
    tags: ['data', 'sql', 'database'],
    isPopular: true,
  },
  
  // Machine Learning
  {
    id: 'ml-training',
    type: 'ml-training',
    label: 'ML Training',
    description: 'Train machine learning models',
    category: 'ml',
    icon: BrainIcon,
    color: '#8b5cf6',
    tags: ['ml', 'training', 'model'],
    isPopular: true,
  },
  {
    id: 'feature-engineering',
    type: 'feature-engineering',
    label: 'Feature Engineering',
    description: 'Transform and engineer features',
    category: 'transform',
    icon: CogIcon,
    color: '#10b981',
    tags: ['features', 'transform', 'preprocessing'],
  },
  {
    id: 'model-validation',
    type: 'validation',
    label: 'Model Validation',
    description: 'Validate model performance',
    category: 'ml',
    icon: ChartBarIcon,
    color: '#8b5cf6',
    tags: ['validation', 'metrics', 'evaluation'],
  },
  
  // Deployment
  {
    id: 'model-deployment',
    type: 'deployment',
    label: 'Model Deploy',
    description: 'Deploy models to production',
    category: 'deploy',
    icon: CloudIcon,
    color: '#f59e0b',
    tags: ['deploy', 'production', 'serving'],
    isNew: true,
  },
  
  // Monitoring
  {
    id: 'drift-detection',
    type: 'monitoring',
    label: 'Drift Detection',
    description: 'Monitor model drift',
    category: 'monitor',
    icon: ChartBarIcon,
    color: '#ef4444',
    tags: ['monitoring', 'drift', 'performance'],
    isNew: true,
  },
];

interface NodePaletteProps {
  isCollapsed: boolean;
  onNodeDragStart: (event: React.DragEvent, nodeType: string) => void;
  onToggleCollapse?: () => void;
}

export const NodePalette: React.FC<NodePaletteProps> = ({
  isCollapsed,
  onNodeDragStart,
  onToggleCollapse,
}) => {
  const [selectedCategory, setSelectedCategory] = useState('all');
  const [searchQuery, setSearchQuery] = useState('');

  const filteredNodes = useMemo(() => {
    let filtered = nodeTemplates;

    // Filter by category
    if (selectedCategory !== 'all') {
      filtered = filtered.filter(node => node.category === selectedCategory);
    }

    // Filter by search query
    if (searchQuery) {
      const query = searchQuery.toLowerCase();
      filtered = filtered.filter(node =>
        node.label.toLowerCase().includes(query) ||
        node.description.toLowerCase().includes(query) ||
        node.tags.some(tag => tag.toLowerCase().includes(query))
      );
    }

    return filtered;
  }, [selectedCategory, searchQuery]);

  const handleDragStart = (event: React.DragEvent, nodeTemplate: NodeTemplate) => {
    event.dataTransfer.setData('application/reactflow', nodeTemplate.type);
    event.dataTransfer.setData('node-template', JSON.stringify(nodeTemplate));
    event.dataTransfer.effectAllowed = 'move';
    onNodeDragStart(event, nodeTemplate.type);
  };

  return (
    <motion.div
      className={`node-palette ${isCollapsed ? 'collapsed' : 'expanded'}`}
      animate={{ width: isCollapsed ? 60 : 320 }}
      transition={{ duration: 0.3, ease: 'easeInOut' }}
    >
      {/* Header */}
      <div className="palette-header">
        <AnimatePresence mode="wait">
          {!isCollapsed && (
            <motion.div
              className="header-content"
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              exit={{ opacity: 0 }}
              transition={{ duration: 0.2 }}
            >
              <h2 className="palette-title">Node Library</h2>
              <Badge variant="secondary" size="sm">
                {filteredNodes.length} nodes
              </Badge>
            </motion.div>
          )}
        </AnimatePresence>

        <button
          className="collapse-toggle"
          onClick={onToggleCollapse}
          aria-label={isCollapsed ? 'Expand palette' : 'Collapse palette'}
        >
          {isCollapsed ? (
            <ChevronRightIcon className="w-5 h-5" />
          ) : (
            <ChevronLeftIcon className="w-5 h-5" />
          )}
        </button>
      </div>

      <AnimatePresence>
        {!isCollapsed && (
          <motion.div
            className="palette-content"
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            transition={{ duration: 0.2 }}
          >
            {/* Search */}
            <div className="search-section">
              <div className="search-input">
                <MagnifyingGlassIcon className="search-icon" />
                <input
                  type="text"
                  placeholder="Search nodes..."
                  value={searchQuery}
                  onChange={(e) => setSearchQuery(e.target.value)}
                  className="search-field"
                />
                {searchQuery && (
                  <button
                    onClick={() => setSearchQuery('')}
                    className="search-clear"
                  >
                    ×
                  </button>
                )}
              </div>
            </div>

            {/* Categories */}
            <div className="categories-section">
              <h3 className="section-title">Categories</h3>
              <div className="categories-list">
                {nodeCategories.map((category) => {
                  const Icon = category.icon;
                  return (
                    <button
                      key={category.id}
                      className={`category-item ${selectedCategory === category.id ? 'active' : ''}`}
                      onClick={() => setSelectedCategory(category.id)}
                    >
                      <Icon
                        className="category-icon"
                        style={{ color: category.color }}
                      />
                      <span className="category-label">{category.label}</span>
                      <Badge variant="outline" size="xs">
                        {category.id === 'all'
                          ? nodeTemplates.length
                          : nodeTemplates.filter(n => n.category === category.id).length
                        }
                      </Badge>
                    </button>
                  );
                })}
              </div>
            </div>

            {/* Nodes */}
            <div className="nodes-section">
              <h3 className="section-title">
                {selectedCategory === 'all' ? 'All Nodes' : 
                 nodeCategories.find(c => c.id === selectedCategory)?.label}
              </h3>
              
              <div className="nodes-list">
                <AnimatePresence>
                  {filteredNodes.map((node, index) => {
                    const Icon = node.icon;
                    return (
                      <motion.div
                        key={node.id}
                        className="node-item"
                        initial={{ opacity: 0, y: 20 }}
                        animate={{ opacity: 1, y: 0 }}
                        exit={{ opacity: 0, y: -20 }}
                        transition={{ duration: 0.2, delay: index * 0.05 }}
                        draggable
                        onDragStart={(e) => handleDragStart(e, node)}
                        whileHover={{ scale: 1.02 }}
                        whileTap={{ scale: 0.98 }}
                      >
                        <Tooltip content={node.description} placement="right">
                          <div className="node-content">
                            <div
                              className="node-icon"
                              style={{ backgroundColor: `${node.color}20` }}
                            >
                              <Icon
                                className="w-5 h-5"
                                style={{ color: node.color }}
                              />
                            </div>
                            
                            <div className="node-info">
                              <div className="node-header">
                                <span className="node-label">{node.label}</span>
                                <div className="node-badges">
                                  {node.isPopular && (
                                    <Badge variant="success" size="xs">
                                      Popular
                                    </Badge>
                                  )}
                                  {node.isNew && (
                                    <Badge variant="warning" size="xs">
                                      New
                                    </Badge>
                                  )}
                                </div>
                              </div>
                              <p className="node-description">{node.description}</p>
                              <div className="node-tags">
                                {node.tags.slice(0, 3).map((tag) => (
                                  <span key={tag} className="node-tag">
                                    {tag}
                                  </span>
                                ))}
                              </div>
                            </div>
                          </div>
                        </Tooltip>
                      </motion.div>
                    );
                  })}
                </AnimatePresence>
              </div>

              {filteredNodes.length === 0 && (
                <div className="empty-state">
                  <p className="empty-message">No nodes found</p>
                  <p className="empty-hint">Try adjusting your search or category filter</p>
                </div>
              )}
            </div>
          </motion.div>
        )}
      </AnimatePresence>

      <style jsx>{`
        .node-palette {
          height: 100vh;
          background: white;
          border-right: 1px solid #e5e7eb;
          display: flex;
          flex-direction: column;
          overflow: hidden;
          box-shadow: 2px 0 4px -1px rgba(0, 0, 0, 0.1);
        }

        .palette-header {
          display: flex;
          align-items: center;
          justify-content: space-between;
          padding: 1rem;
          border-bottom: 1px solid #e5e7eb;
          background: #f9fafb;
          min-height: 60px;
        }

        .header-content {
          display: flex;
          align-items: center;
          gap: 0.75rem;
        }

        .palette-title {
          font-size: 1.125rem;
          font-weight: 600;
          color: #111827;
          margin: 0;
        }

        .collapse-toggle {
          width: 32px;
          height: 32px;
          border-radius: 6px;
          border: 1px solid #d1d5db;
          background: white;
          display: flex;
          align-items: center;
          justify-content: center;
          cursor: pointer;
          transition: all 0.2s ease;
          color: #6b7280;
        }

        .collapse-toggle:hover {
          background: #f3f4f6;
          border-color: #9ca3af;
          color: #374151;
        }

        .palette-content {
          flex: 1;
          overflow: hidden;
          display: flex;
          flex-direction: column;
        }

        .search-section {
          padding: 1rem;
          border-bottom: 1px solid #e5e7eb;
        }

        .search-input {
          position: relative;
          display: flex;
          align-items: center;
        }

        .search-icon {
          position: absolute;
          left: 12px;
          width: 18px;
          height: 18px;
          color: #6b7280;
          z-index: 1;
        }

        .search-field {
          width: 100%;
          padding: 0.75rem 0.75rem 0.75rem 2.5rem;
          border: 1px solid #d1d5db;
          border-radius: 8px;
          font-size: 0.875rem;
          background: white;
          transition: all 0.2s ease;
        }

        .search-field:focus {
          outline: none;
          border-color: #3b82f6;
          box-shadow: 0 0 0 3px rgba(59, 130, 246, 0.1);
        }

        .search-clear {
          position: absolute;
          right: 12px;
          width: 20px;
          height: 20px;
          border: none;
          background: #6b7280;
          color: white;
          border-radius: 50%;
          cursor: pointer;
          font-size: 14px;
          display: flex;
          align-items: center;
          justify-content: center;
        }

        .categories-section {
          padding: 1rem;
          border-bottom: 1px solid #e5e7eb;
        }

        .section-title {
          font-size: 0.875rem;
          font-weight: 600;
          color: #374151;
          margin: 0 0 0.75rem 0;
          text-transform: uppercase;
          letter-spacing: 0.05em;
        }

        .categories-list {
          display: flex;
          flex-direction: column;
          gap: 2px;
        }

        .category-item {
          display: flex;
          align-items: center;
          gap: 0.75rem;
          padding: 0.625rem 0.75rem;
          border: none;
          background: transparent;
          border-radius: 6px;
          cursor: pointer;
          transition: all 0.2s ease;
          text-align: left;
          width: 100%;
        }

        .category-item:hover {
          background: #f3f4f6;
        }

        .category-item.active {
          background: #dbeafe;
          color: #1d4ed8;
        }

        .category-icon {
          width: 18px;
          height: 18px;
          flex-shrink: 0;
        }

        .category-label {
          flex: 1;
          font-size: 0.875rem;
          font-weight: 500;
        }

        .nodes-section {
          flex: 1;
          padding: 1rem;
          overflow: hidden;
          display: flex;
          flex-direction: column;
        }

        .nodes-list {
          flex: 1;
          overflow-y: auto;
          display: flex;
          flex-direction: column;
          gap: 0.5rem;
        }

        .node-item {
          border: 1px solid #e5e7eb;
          border-radius: 8px;
          background: white;
          cursor: grab;
          transition: all 0.2s ease;
        }

        .node-item:hover {
          border-color: #3b82f6;
          box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
        }

        .node-item:active {
          cursor: grabbing;
        }

        .node-content {
          padding: 0.75rem;
          display: flex;
          gap: 0.75rem;
        }

        .node-icon {
          width: 40px;
          height: 40px;
          border-radius: 8px;
          display: flex;
          align-items: center;
          justify-content: center;
          flex-shrink: 0;
        }

        .node-info {
          flex: 1;
          min-width: 0;
        }

        .node-header {
          display: flex;
          align-items: center;
          justify-content: space-between;
          margin-bottom: 0.25rem;
        }

        .node-label {
          font-size: 0.875rem;
          font-weight: 600;
          color: #111827;
        }

        .node-badges {
          display: flex;
          gap: 0.25rem;
        }

        .node-description {
          font-size: 0.75rem;
          color: #6b7280;
          margin: 0 0 0.5rem 0;
          line-height: 1.4;
        }

        .node-tags {
          display: flex;
          gap: 0.25rem;
          flex-wrap: wrap;
        }

        .node-tag {
          font-size: 0.625rem;
          padding: 0.125rem 0.375rem;
          background: #f3f4f6;
          color: #6b7280;
          border-radius: 12px;
          font-weight: 500;
        }

        .empty-state {
          text-align: center;
          padding: 2rem 1rem;
        }

        .empty-message {
          font-size: 0.875rem;
          font-weight: 500;
          color: #6b7280;
          margin: 0 0 0.25rem 0;
        }

        .empty-hint {
          font-size: 0.75rem;
          color: #9ca3af;
          margin: 0;
        }

        /* Scrollbar styling */
        .nodes-list::-webkit-scrollbar {
          width: 6px;
        }

        .nodes-list::-webkit-scrollbar-track {
          background: #f1f5f9;
          border-radius: 3px;
        }

        .nodes-list::-webkit-scrollbar-thumb {
          background: #cbd5e1;
          border-radius: 3px;
        }

        .nodes-list::-webkit-scrollbar-thumb:hover {
          background: #94a3b8;
        }
      `}</style>
    </motion.div>
  );
};

export default NodePalette;