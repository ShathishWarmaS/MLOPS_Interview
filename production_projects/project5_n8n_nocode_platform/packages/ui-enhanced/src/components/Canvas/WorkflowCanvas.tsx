import React, { useCallback, useRef, useState, useMemo } from 'react';
import ReactFlow, {
  Node,
  Edge,
  addEdge,
  Background,
  Controls,
  MiniMap,
  useNodesState,
  useEdgesState,
  ConnectionMode,
  Connection,
  ReactFlowProvider,
  Panel,
  useReactFlow,
  MarkerType,
  Position,
} from 'reactflow';
import { motion, AnimatePresence } from 'framer-motion';
import { useHotkeys } from 'react-hotkeys-hook';
import { toast } from 'react-hot-toast';

import { CustomNodeTypes } from './nodes';
import { NodePalette } from '../NodePalette/NodePalette';
import { PropertyPanel } from '../PropertyPanel/PropertyPanel';
import { Toolbar } from '../Toolbar/Toolbar';
import { CollaborationCursors } from '../Collaboration/CollaborationCursors';
import { ExecutionIndicator } from '../Execution/ExecutionIndicator';
import { ContextMenu } from '../ContextMenu/ContextMenu';
import { useWorkflowStore } from '../../stores/workflowStore';
import { useCollaboration } from '../../hooks/useCollaboration';
import { useWorkflowExecution } from '../../hooks/useWorkflowExecution';
import { useAutoSave } from '../../hooks/useAutoSave';

import 'reactflow/dist/style.css';
import './WorkflowCanvas.css';

interface WorkflowCanvasProps {
  workflowId: string;
  isReadOnly?: boolean;
  showMinimap?: boolean;
  showControls?: boolean;
  theme?: 'light' | 'dark';
}

const connectionLineStyle = {
  strokeWidth: 2,
  stroke: '#4f46e5',
};

const defaultEdgeOptions = {
  style: connectionLineStyle,
  type: 'smoothstep',
  markerEnd: {
    type: MarkerType.ArrowClosed,
    color: '#4f46e5',
  },
};

export const WorkflowCanvas: React.FC<WorkflowCanvasProps> = ({
  workflowId,
  isReadOnly = false,
  showMinimap = true,
  showControls = true,
  theme = 'light',
}) => {
  const reactFlowWrapper = useRef<HTMLDivElement>(null);
  const [reactFlowInstance, setReactFlowInstance] = useState<any>(null);
  const [contextMenu, setContextMenu] = useState<{
    visible: boolean;
    x: number;
    y: number;
    nodeId?: string;
  }>({ visible: false, x: 0, y: 0 });

  // Store hooks
  const {
    nodes: storeNodes,
    edges: storeEdges,
    selectedNodes,
    selectedEdges,
    addNode,
    updateNode,
    deleteNode,
    addEdge: storeAddEdge,
    deleteEdge,
    setSelectedNodes,
    setSelectedEdges,
    clearSelection,
  } = useWorkflowStore();

  // React Flow hooks
  const [nodes, setNodes, onNodesChange] = useNodesState(storeNodes);
  const [edges, setEdges, onEdgesChange] = useEdgesState(storeEdges);

  // Custom hooks
  const { collaborators, broadcastCursor } = useCollaboration(workflowId);
  const { executeWorkflow, isExecuting, executionResults } = useWorkflowExecution();
  const { saveStatus } = useAutoSave(workflowId, { nodes, edges });

  // Hotkeys
  useHotkeys('ctrl+s, cmd+s', (e) => {
    e.preventDefault();
    handleSave();
  });

  useHotkeys('delete', () => {
    handleDeleteSelected();
  });

  useHotkeys('ctrl+c, cmd+c', () => {
    handleCopy();
  });

  useHotkeys('ctrl+v, cmd+v', () => {
    handlePaste();
  });

  useHotkeys('ctrl+z, cmd+z', () => {
    handleUndo();
  });

  useHotkeys('ctrl+y, cmd+y', () => {
    handleRedo();
  });

  // Event handlers
  const onConnect = useCallback(
    (params: Connection | Edge) => {
      const newEdge = {
        ...params,
        id: `e${params.source}-${params.target}`,
        ...defaultEdgeOptions,
      };
      setEdges((eds) => addEdge(newEdge, eds));
      storeAddEdge(newEdge);
    },
    [setEdges, storeAddEdge]
  );

  const onDragOver = useCallback((event: React.DragEvent) => {
    event.preventDefault();
    event.dataTransfer.dropEffect = 'move';
  }, []);

  const onDrop = useCallback(
    (event: React.DragEvent) => {
      event.preventDefault();

      const reactFlowBounds = reactFlowWrapper.current?.getBoundingClientRect();
      const type = event.dataTransfer.getData('application/reactflow');

      if (typeof type === 'undefined' || !type || !reactFlowInstance) {
        return;
      }

      const position = reactFlowInstance.project({
        x: event.clientX - (reactFlowBounds?.left ?? 0),
        y: event.clientY - (reactFlowBounds?.top ?? 0),
      });

      const newNode = {
        id: `node_${Date.now()}`,
        type,
        position,
        data: {
          label: `${type} node`,
          config: {},
        },
        sourcePosition: Position.Right,
        targetPosition: Position.Left,
      };

      setNodes((nds) => nds.concat(newNode));
      addNode(newNode);
      
      toast.success(`Added ${type} node`);
    },
    [reactFlowInstance, setNodes, addNode]
  );

  const onNodeContextMenu = useCallback(
    (event: React.MouseEvent, node: Node) => {
      event.preventDefault();
      setContextMenu({
        visible: true,
        x: event.clientX,
        y: event.clientY,
        nodeId: node.id,
      });
    },
    []
  );

  const onPaneContextMenu = useCallback((event: React.MouseEvent) => {
    event.preventDefault();
    setContextMenu({
      visible: true,
      x: event.clientX,
      y: event.clientY,
    });
  }, []);

  const onSelectionChange = useCallback(
    ({ nodes: selectedNodes, edges: selectedEdges }: { nodes: Node[]; edges: Edge[] }) => {
      setSelectedNodes(selectedNodes.map((n) => n.id));
      setSelectedEdges(selectedEdges.map((e) => e.id));
    },
    [setSelectedNodes, setSelectedEdges]
  );

  const onMouseMove = useCallback(
    (event: React.MouseEvent) => {
      if (reactFlowWrapper.current) {
        const bounds = reactFlowWrapper.current.getBoundingClientRect();
        broadcastCursor({
          x: event.clientX - bounds.left,
          y: event.clientY - bounds.top,
        });
      }
    },
    [broadcastCursor]
  );

  // Action handlers
  const handleSave = useCallback(async () => {
    try {
      // Save workflow logic here
      toast.success('Workflow saved successfully');
    } catch (error) {
      toast.error('Failed to save workflow');
    }
  }, []);

  const handleExecute = useCallback(async () => {
    try {
      await executeWorkflow({ nodes, edges });
      toast.success('Workflow execution started');
    } catch (error) {
      toast.error('Failed to execute workflow');
    }
  }, [executeWorkflow, nodes, edges]);

  const handleDeleteSelected = useCallback(() => {
    selectedNodes.forEach((nodeId) => {
      setNodes((nds) => nds.filter((n) => n.id !== nodeId));
      deleteNode(nodeId);
    });
    
    selectedEdges.forEach((edgeId) => {
      setEdges((eds) => eds.filter((e) => e.id !== edgeId));
      deleteEdge(edgeId);
    });
    
    clearSelection();
    toast.success('Deleted selected items');
  }, [selectedNodes, selectedEdges, setNodes, setEdges, deleteNode, deleteEdge, clearSelection]);

  const handleCopy = useCallback(() => {
    // Copy logic here
    toast.success('Copied to clipboard');
  }, []);

  const handlePaste = useCallback(() => {
    // Paste logic here
    toast.success('Pasted from clipboard');
  }, []);

  const handleUndo = useCallback(() => {
    // Undo logic here
    toast.success('Undone');
  }, []);

  const handleRedo = useCallback(() => {
    // Redo logic here
    toast.success('Redone');
  }, []);

  const closeContextMenu = useCallback(() => {
    setContextMenu({ visible: false, x: 0, y: 0 });
  }, []);

  // Memoized node types
  const nodeTypes = useMemo(() => CustomNodeTypes, []);

  return (
    <div className="workflow-canvas-container">
      <ReactFlowProvider>
        <div className="workflow-canvas" ref={reactFlowWrapper}>
          {/* Main Canvas */}
          <ReactFlow
            nodes={nodes}
            edges={edges}
            onNodesChange={onNodesChange}
            onEdgesChange={onEdgesChange}
            onConnect={onConnect}
            onInit={setReactFlowInstance}
            onDrop={onDrop}
            onDragOver={onDragOver}
            onNodeContextMenu={onNodeContextMenu}
            onPaneContextMenu={onPaneContextMenu}
            onSelectionChange={onSelectionChange}
            onMouseMove={onMouseMove}
            onClick={closeContextMenu}
            connectionMode={ConnectionMode.Loose}
            defaultEdgeOptions={defaultEdgeOptions}
            nodeTypes={nodeTypes}
            className={`workflow-canvas-main theme-${theme}`}
            proOptions={{ hideAttribution: true }}
            fitView
            attributionPosition="bottom-left"
          >
            {/* Background */}
            <Background
              variant="dots"
              gap={20}
              size={1}
              className="workflow-background"
            />

            {/* Controls */}
            {showControls && (
              <Controls
                className="workflow-controls"
                showZoom={true}
                showFitView={true}
                showInteractive={true}
              />
            )}

            {/* Minimap */}
            {showMinimap && (
              <MiniMap
                className="workflow-minimap"
                nodeStrokeColor="#374151"
                nodeColor="#e5e7eb"
                nodeBorderRadius={8}
                maskColor="rgba(0, 0, 0, 0.1)"
              />
            )}

            {/* Panels */}
            <Panel position="top-left">
              <Toolbar
                onSave={handleSave}
                onExecute={handleExecute}
                onUndo={handleUndo}
                onRedo={handleRedo}
                isExecuting={isExecuting}
                saveStatus={saveStatus}
                isReadOnly={isReadOnly}
              />
            </Panel>

            <Panel position="top-right">
              <ExecutionIndicator
                isExecuting={isExecuting}
                results={executionResults}
              />
            </Panel>

            <Panel position="bottom-right">
              <div className="workflow-stats">
                <span className="stat-item">
                  Nodes: {nodes.length}
                </span>
                <span className="stat-item">
                  Connections: {edges.length}
                </span>
                <span className="stat-item">
                  Selected: {selectedNodes.length + selectedEdges.length}
                </span>
              </div>
            </Panel>
          </ReactFlow>

          {/* Collaboration Cursors */}
          <CollaborationCursors collaborators={collaborators} />

          {/* Context Menu */}
          <AnimatePresence>
            {contextMenu.visible && (
              <ContextMenu
                x={contextMenu.x}
                y={contextMenu.y}
                nodeId={contextMenu.nodeId}
                onClose={closeContextMenu}
              />
            )}
          </AnimatePresence>
        </div>

        {/* Side Panels */}
        <div className="workflow-sidepanels">
          <NodePalette
            isCollapsed={false}
            onNodeDragStart={() => {}}
          />
          
          <PropertyPanel
            selectedNodes={selectedNodes}
            selectedEdges={selectedEdges}
            onUpdateNode={updateNode}
          />
        </div>
      </ReactFlowProvider>

      <style jsx>{`
        .workflow-canvas-container {
          display: flex;
          width: 100vw;
          height: 100vh;
          background: ${theme === 'dark' ? '#1f2937' : '#f9fafb'};
        }

        .workflow-canvas {
          flex: 1;
          position: relative;
        }

        .workflow-canvas-main {
          width: 100%;
          height: 100%;
        }

        .workflow-canvas-main.theme-dark {
          background: #1f2937;
        }

        .workflow-canvas-main.theme-light {
          background: #ffffff;
        }

        .workflow-sidepanels {
          display: flex;
          flex-direction: row;
        }

        .workflow-stats {
          display: flex;
          gap: 1rem;
          padding: 0.5rem 1rem;
          background: rgba(255, 255, 255, 0.9);
          border-radius: 0.5rem;
          box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
          font-size: 0.875rem;
          color: #374151;
        }

        .stat-item {
          white-space: nowrap;
        }

        .workflow-background {
          opacity: 0.5;
        }

        .workflow-controls {
          box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
        }

        .workflow-minimap {
          box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
          border-radius: 0.5rem;
        }

        /* Custom scrollbars */
        ::-webkit-scrollbar {
          width: 8px;
          height: 8px;
        }

        ::-webkit-scrollbar-track {
          background: #f1f5f9;
          border-radius: 4px;
        }

        ::-webkit-scrollbar-thumb {
          background: #cbd5e1;
          border-radius: 4px;
        }

        ::-webkit-scrollbar-thumb:hover {
          background: #94a3b8;
        }
      `}</style>
    </div>
  );
};

export default WorkflowCanvas;