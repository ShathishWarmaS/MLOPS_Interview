import React, { memo, useState } from 'react';
import { Handle, Position, NodeProps } from 'reactflow';
import { motion } from 'framer-motion';
import { 
  BrainIcon, 
  PlayIcon, 
  PauseIcon, 
  CheckCircleIcon, 
  ExclamationTriangleIcon,
  CogIcon,
  ChartBarIcon 
} from '@heroicons/react/24/outline';
import { Tooltip } from '../../UI/Tooltip';
import { Badge } from '../../UI/Badge';
import { ProgressBar } from '../../UI/ProgressBar';

interface MLTrainingNodeData {
  label: string;
  algorithm: string;
  status: 'idle' | 'running' | 'completed' | 'error';
  progress?: number;
  metrics?: {
    accuracy: number;
    loss: number;
    epoch: number;
    totalEpochs: number;
  };
  config: {
    hyperparameters: Record<string, any>;
    dataset: string;
    outputPath: string;
  };
  isSelected?: boolean;
  isExecuting?: boolean;
}

const statusColors = {
  idle: 'bg-gray-100 border-gray-300 text-gray-700',
  running: 'bg-blue-50 border-blue-300 text-blue-700',
  completed: 'bg-green-50 border-green-300 text-green-700',
  error: 'bg-red-50 border-red-300 text-red-700',
};

const statusIcons = {
  idle: CogIcon,
  running: PlayIcon,
  completed: CheckCircleIcon,
  error: ExclamationTriangleIcon,
};

export const MLTrainingNode: React.FC<NodeProps<MLTrainingNodeData>> = memo(({ data, selected, id }) => {
  const [isExpanded, setIsExpanded] = useState(false);
  const StatusIcon = statusIcons[data.status];

  return (
    <motion.div
      className={`ml-training-node ${statusColors[data.status]} ${selected ? 'ring-2 ring-blue-500' : ''}`}
      initial={{ scale: 0.9, opacity: 0 }}
      animate={{ scale: 1, opacity: 1 }}
      transition={{ duration: 0.2 }}
      whileHover={{ scale: 1.02 }}
      layout
    >
      {/* Input Handle */}
      <Handle
        type="target"
        position={Position.Left}
        className="ml-node-handle handle-input"
        isConnectable={true}
      />

      {/* Node Header */}
      <div className="node-header">
        <div className="node-icon">
          <BrainIcon className="w-5 h-5" />
        </div>
        
        <div className="node-title">
          <h3 className="font-semibold text-sm">{data.label}</h3>
          <p className="text-xs opacity-70">{data.algorithm}</p>
        </div>

        <div className="node-status">
          <Tooltip content={`Status: ${data.status}`}>
            <StatusIcon className="w-4 h-4" />
          </Tooltip>
        </div>
      </div>

      {/* Training Progress */}
      {data.status === 'running' && data.progress !== undefined && (
        <div className="training-progress">
          <ProgressBar 
            progress={data.progress} 
            className="mb-2"
            showLabel={true}
          />
          {data.metrics && (
            <div className="metrics-mini">
              <span className="metric">
                Epoch: {data.metrics.epoch}/{data.metrics.totalEpochs}
              </span>
              <span className="metric">
                Acc: {(data.metrics.accuracy * 100).toFixed(1)}%
              </span>
            </div>
          )}
        </div>
      )}

      {/* Status Badges */}
      <div className="node-badges">
        <Badge variant={data.status === 'completed' ? 'success' : 'default'} size="sm">
          {data.algorithm}
        </Badge>
        
        {data.status === 'running' && (
          <Badge variant="warning" size="sm" animated>
            Training
          </Badge>
        )}
        
        {data.status === 'completed' && data.metrics && (
          <Badge variant="success" size="sm">
            {(data.metrics.accuracy * 100).toFixed(1)}% Acc
          </Badge>
        )}
      </div>

      {/* Expandable Details */}
      {isExpanded && (
        <motion.div
          className="node-details"
          initial={{ height: 0, opacity: 0 }}
          animate={{ height: 'auto', opacity: 1 }}
          exit={{ height: 0, opacity: 0 }}
          transition={{ duration: 0.2 }}
        >
          <div className="detail-section">
            <h4 className="detail-title">Configuration</h4>
            <div className="detail-content">
              <div className="config-item">
                <span className="config-label">Dataset:</span>
                <span className="config-value">{data.config.dataset}</span>
              </div>
              <div className="config-item">
                <span className="config-label">Output:</span>
                <span className="config-value">{data.config.outputPath}</span>
              </div>
            </div>
          </div>

          {data.config.hyperparameters && (
            <div className="detail-section">
              <h4 className="detail-title">Hyperparameters</h4>
              <div className="hyperparameters">
                {Object.entries(data.config.hyperparameters).map(([key, value]) => (
                  <div key={key} className="param-item">
                    <span className="param-key">{key}:</span>
                    <span className="param-value">{String(value)}</span>
                  </div>
                ))}
              </div>
            </div>
          )}

          {data.metrics && (
            <div className="detail-section">
              <h4 className="detail-title">Metrics</h4>
              <div className="metrics-grid">
                <div className="metric-item">
                  <ChartBarIcon className="w-4 h-4" />
                  <span>Accuracy: {(data.metrics.accuracy * 100).toFixed(2)}%</span>
                </div>
                <div className="metric-item">
                  <ChartBarIcon className="w-4 h-4" />
                  <span>Loss: {data.metrics.loss.toFixed(4)}</span>
                </div>
              </div>
            </div>
          )}
        </motion.div>
      )}

      {/* Expand Toggle */}
      <button
        className="expand-toggle"
        onClick={() => setIsExpanded(!isExpanded)}
        aria-label={isExpanded ? 'Collapse details' : 'Expand details'}
      >
        <motion.div
          animate={{ rotate: isExpanded ? 180 : 0 }}
          transition={{ duration: 0.2 }}
        >
          ▼
        </motion.div>
      </button>

      {/* Execution Indicator */}
      {data.isExecuting && (
        <div className="execution-indicator">
          <motion.div
            className="execution-pulse"
            animate={{ scale: [1, 1.2, 1], opacity: [0.7, 1, 0.7] }}
            transition={{ duration: 1.5, repeat: Infinity }}
          />
        </div>
      )}

      {/* Output Handle */}
      <Handle
        type="source"
        position={Position.Right}
        className="ml-node-handle handle-output"
        isConnectable={true}
      />

      <style jsx>{`
        .ml-training-node {
          min-width: 280px;
          max-width: 350px;
          padding: 16px;
          border-radius: 12px;
          border: 2px solid;
          background: white;
          box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
          position: relative;
          transition: all 0.2s ease;
        }

        .ml-training-node:hover {
          box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.1), 0 4px 6px -2px rgba(0, 0, 0, 0.05);
        }

        .node-header {
          display: flex;
          align-items: center;
          gap: 12px;
          margin-bottom: 12px;
        }

        .node-icon {
          width: 32px;
          height: 32px;
          border-radius: 8px;
          background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
          display: flex;
          align-items: center;
          justify-content: center;
          color: white;
        }

        .node-title {
          flex: 1;
        }

        .node-title h3 {
          margin: 0;
          font-size: 14px;
          font-weight: 600;
        }

        .node-title p {
          margin: 0;
          font-size: 12px;
          opacity: 0.7;
        }

        .node-status {
          width: 20px;
          height: 20px;
          display: flex;
          align-items: center;
          justify-content: center;
        }

        .training-progress {
          margin-bottom: 12px;
        }

        .metrics-mini {
          display: flex;
          justify-content: space-between;
          font-size: 11px;
          opacity: 0.8;
        }

        .node-badges {
          display: flex;
          gap: 6px;
          flex-wrap: wrap;
          margin-bottom: 8px;
        }

        .node-details {
          border-top: 1px solid rgba(0, 0, 0, 0.1);
          padding-top: 12px;
          margin-top: 12px;
        }

        .detail-section {
          margin-bottom: 12px;
        }

        .detail-section:last-child {
          margin-bottom: 0;
        }

        .detail-title {
          font-size: 12px;
          font-weight: 600;
          margin-bottom: 6px;
          color: #374151;
        }

        .detail-content {
          display: flex;
          flex-direction: column;
          gap: 4px;
        }

        .config-item {
          display: flex;
          justify-content: space-between;
          font-size: 11px;
        }

        .config-label {
          color: #6b7280;
        }

        .config-value {
          font-weight: 500;
          color: #374151;
        }

        .hyperparameters {
          display: grid;
          grid-template-columns: 1fr 1fr;
          gap: 4px;
        }

        .param-item {
          display: flex;
          flex-direction: column;
          font-size: 11px;
        }

        .param-key {
          color: #6b7280;
          font-weight: 500;
        }

        .param-value {
          color: #374151;
          font-weight: 600;
        }

        .metrics-grid {
          display: flex;
          flex-direction: column;
          gap: 6px;
        }

        .metric-item {
          display: flex;
          align-items: center;
          gap: 6px;
          font-size: 11px;
          color: #374151;
        }

        .expand-toggle {
          position: absolute;
          bottom: -8px;
          left: 50%;
          transform: translateX(-50%);
          width: 20px;
          height: 20px;
          border-radius: 50%;
          background: white;
          border: 2px solid #e5e7eb;
          display: flex;
          align-items: center;
          justify-content: center;
          cursor: pointer;
          font-size: 10px;
          color: #6b7280;
          transition: all 0.2s ease;
        }

        .expand-toggle:hover {
          border-color: #3b82f6;
          color: #3b82f6;
        }

        .execution-indicator {
          position: absolute;
          top: -4px;
          right: -4px;
          width: 12px;
          height: 12px;
        }

        .execution-pulse {
          width: 100%;
          height: 100%;
          border-radius: 50%;
          background: #10b981;
        }

        .ml-node-handle {
          width: 12px;
          height: 12px;
          border: 2px solid #374151;
          background: white;
        }

        .handle-input {
          left: -6px;
        }

        .handle-output {
          right: -6px;
        }

        .ml-node-handle:hover {
          border-color: #3b82f6;
        }
      `}</style>
    </motion.div>
  );
});

MLTrainingNode.displayName = 'MLTrainingNode';

export default MLTrainingNode;