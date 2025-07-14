<template>
  <div class="workflow-canvas-container">
    <!-- Canvas Header -->
    <div class="canvas-header">
      <div class="workflow-info">
        <h2 class="workflow-title" v-if="workflow">
          {{ workflow.name || 'Untitled Workflow' }}
        </h2>
        <div class="workflow-status">
          <span :class="['status-indicator', workflow?.status || 'draft']">
            {{ formatStatus(workflow?.status || 'draft') }}
          </span>
        </div>
      </div>
      
      <div class="canvas-controls">
        <button @click="saveWorkflow" class="btn btn-primary" :disabled="!hasChanges">
          <i class="fas fa-save"></i> Save
        </button>
        <button @click="executeWorkflow" class="btn btn-success" :disabled="!canExecute">
          <i class="fas fa-play"></i> Execute
        </button>
        <button @click="stopExecution" class="btn btn-danger" v-if="isExecuting">
          <i class="fas fa-stop"></i> Stop
        </button>
        <button @click="showSettings" class="btn btn-secondary">
          <i class="fas fa-cog"></i> Settings
        </button>
      </div>
    </div>

    <!-- Main Canvas Area -->
    <div class="canvas-main" ref="canvasContainer">
      <!-- Canvas Background -->
      <div 
        class="canvas-background"
        :style="{ 
          transform: `translate(${canvasTransform.x}px, ${canvasTransform.y}px) scale(${canvasTransform.scale})`,
          width: `${canvasSize.width}px`,
          height: `${canvasSize.height}px`
        }"
        @mousedown="onCanvasMouseDown"
        @mousemove="onCanvasMouseMove"
        @mouseup="onCanvasMouseUp"
        @wheel="onCanvasWheel"
      >
        <!-- Grid Pattern -->
        <div class="canvas-grid" v-if="showGrid">
          <svg width="100%" height="100%">
            <defs>
              <pattern id="grid" width="20" height="20" patternUnits="userSpaceOnUse">
                <path d="M 20 0 L 0 0 0 20" fill="none" stroke="#e0e0e0" stroke-width="1"/>
              </pattern>
            </defs>
            <rect width="100%" height="100%" fill="url(#grid)" />
          </svg>
        </div>

        <!-- Connection Lines -->
        <svg class="connections-layer" width="100%" height="100%">
          <g v-for="connection in connections" :key="connection.id">
            <ConnectionRenderer
              :connection="connection"
              :nodes="nodes"
              :is-selected="selectedConnections.includes(connection.id)"
              @select="selectConnection"
              @delete="deleteConnection"
            />
          </g>
          
          <!-- Temporary connection line while dragging -->
          <g v-if="dragConnection.active">
            <path
              :d="createConnectionPath(dragConnection.start, dragConnection.current)"
              stroke="#4a90e2"
              stroke-width="2"
              fill="none"
              stroke-dasharray="5,5"
            />
          </g>
        </svg>

        <!-- Workflow Nodes -->
        <div class="nodes-layer">
          <NodeRenderer
            v-for="node in nodes"
            :key="node.id"
            :node="node"
            :is-selected="selectedNodes.includes(node.id)"
            :is-executing="executingNodes.includes(node.id)"
            :execution-data="getNodeExecutionData(node.id)"
            :scale="canvasTransform.scale"
            @select="selectNode"
            @move="moveNode"
            @delete="deleteNode"
            @duplicate="duplicateNode"
            @connection-start="startConnection"
            @connection-end="endConnection"
            @edit="editNode"
          />
        </div>

        <!-- Selection Box -->
        <div
          v-if="selectionBox.active"
          class="selection-box"
          :style="{
            left: `${Math.min(selectionBox.start.x, selectionBox.end.x)}px`,
            top: `${Math.min(selectionBox.start.y, selectionBox.end.y)}px`,
            width: `${Math.abs(selectionBox.end.x - selectionBox.start.x)}px`,
            height: `${Math.abs(selectionBox.end.y - selectionBox.start.y)}px`
          }"
        />
      </div>
    </div>

    <!-- Canvas Footer -->
    <div class="canvas-footer">
      <div class="canvas-info">
        <span>Nodes: {{ nodes.length }}</span>
        <span>Connections: {{ connections.length }}</span>
        <span>Zoom: {{ Math.round(canvasTransform.scale * 100) }}%</span>
      </div>
      
      <div class="canvas-zoom-controls">
        <button @click="zoomOut" class="btn btn-sm">
          <i class="fas fa-minus"></i>
        </button>
        <button @click="resetZoom" class="btn btn-sm">
          <i class="fas fa-expand-arrows-alt"></i>
        </button>
        <button @click="zoomIn" class="btn btn-sm">
          <i class="fas fa-plus"></i>
        </button>
        <button @click="fitToView" class="btn btn-sm">
          <i class="fas fa-compress-arrows-alt"></i>
        </button>
      </div>
    </div>

    <!-- Context Menu -->
    <div
      v-if="contextMenu.visible"
      class="context-menu"
      :style="{ left: `${contextMenu.x}px`, top: `${contextMenu.y}px` }"
      @click.stop
    >
      <div class="context-menu-item" @click="addNode">
        <i class="fas fa-plus"></i> Add Node
      </div>
      <div class="context-menu-item" @click="pasteNodes" v-if="clipboardHasNodes">
        <i class="fas fa-paste"></i> Paste
      </div>
      <div class="context-menu-separator" v-if="hasSelection"></div>
      <div class="context-menu-item" @click="copySelection" v-if="hasSelection">
        <i class="fas fa-copy"></i> Copy
      </div>
      <div class="context-menu-item" @click="deleteSelection" v-if="hasSelection">
        <i class="fas fa-trash"></i> Delete
      </div>
    </div>

    <!-- Minimap -->
    <div class="minimap" v-if="showMinimap">
      <div class="minimap-viewport" :style="minimapViewportStyle"></div>
      <svg class="minimap-content" viewBox="0 0 1000 600">
        <rect
          v-for="node in nodes"
          :key="node.id"
          :x="node.position[0] / 2"
          :y="node.position[1] / 2"
          width="40"
          height="20"
          :fill="getNodeColor(node.type)"
          opacity="0.7"
        />
      </svg>
    </div>
  </div>
</template>

<script setup lang="ts">
import { ref, reactive, computed, onMounted, onUnmounted, watch } from 'vue';
import { useStore } from 'vuex';
import { useRoute, useRouter } from 'vue-router';
import NodeRenderer from './NodeRenderer.vue';
import ConnectionRenderer from './ConnectionRenderer.vue';

// Types
interface Node {
  id: string;
  name: string;
  type: string;
  position: [number, number];
  parameters: Record<string, any>;
  disabled?: boolean;
}

interface Connection {
  id: string;
  sourceNodeId: string;
  sourceOutput: number;
  targetNodeId: string;
  targetInput: number;
}

interface CanvasTransform {
  x: number;
  y: number;
  scale: number;
}

// Store and routing
const store = useStore();
const route = useRoute();
const router = useRouter();

// Refs
const canvasContainer = ref<HTMLElement>();

// Reactive data
const workflow = ref(null);
const nodes = ref<Node[]>([]);
const connections = ref<Connection[]>([]);
const selectedNodes = ref<string[]>([]);
const selectedConnections = ref<string[]>([]);
const executingNodes = ref<string[]>([]);
const hasChanges = ref(false);
const isExecuting = ref(false);
const showGrid = ref(true);
const showMinimap = ref(true);

const canvasTransform = reactive<CanvasTransform>({
  x: 0,
  y: 0,
  scale: 1
});

const canvasSize = reactive({
  width: 5000,
  height: 3000
});

const dragConnection = reactive({
  active: false,
  start: { x: 0, y: 0 },
  current: { x: 0, y: 0 },
  sourceNodeId: '',
  sourceOutput: 0
});

const selectionBox = reactive({
  active: false,
  start: { x: 0, y: 0 },
  end: { x: 0, y: 0 }
});

const contextMenu = reactive({
  visible: false,
  x: 0,
  y: 0
});

// Computed properties
const canExecute = computed(() => {
  return nodes.value.length > 0 && !isExecuting.value;
});

const hasSelection = computed(() => {
  return selectedNodes.value.length > 0 || selectedConnections.value.length > 0;
});

const clipboardHasNodes = computed(() => {
  // Check if clipboard has nodes
  return false; // Placeholder
});

const minimapViewportStyle = computed(() => {
  const containerRect = canvasContainer.value?.getBoundingClientRect();
  if (!containerRect) return {};
  
  return {
    left: `${-canvasTransform.x / 10}px`,
    top: `${-canvasTransform.y / 10}px`,
    width: `${containerRect.width / 10}px`,
    height: `${containerRect.height / 10}px`
  };
});

// Methods
const formatStatus = (status: string): string => {
  return status.charAt(0).toUpperCase() + status.slice(1);
};

const getNodeColor = (nodeType: string): string => {
  const colors: Record<string, string> = {
    'modelTraining': '#4a90e2',
    'modelDeployment': '#7ed321',
    'dataValidation': '#f5a623',
    'featureEngineering': '#bd10e0',
    'modelMonitoring': '#b8e986'
  };
  return colors[nodeType] || '#50e3c2';
};

const saveWorkflow = async () => {
  try {
    await store.dispatch('workflows/saveWorkflow', {
      id: workflow.value?.id,
      nodes: nodes.value,
      connections: connections.value
    });
    hasChanges.value = false;
  } catch (error) {
    console.error('Failed to save workflow:', error);
  }
};

const executeWorkflow = async () => {
  try {
    isExecuting.value = true;
    executingNodes.value = [];
    
    await store.dispatch('workflows/executeWorkflow', {
      id: workflow.value?.id,
      nodes: nodes.value,
      connections: connections.value
    });
  } catch (error) {
    console.error('Failed to execute workflow:', error);
  } finally {
    isExecuting.value = false;
  }
};

const stopExecution = async () => {
  try {
    await store.dispatch('workflows/stopExecution', workflow.value?.id);
    isExecuting.value = false;
    executingNodes.value = [];
  } catch (error) {
    console.error('Failed to stop execution:', error);
  }
};

const showSettings = () => {
  // Open workflow settings dialog
};

const selectNode = (nodeId: string, multi = false) => {
  if (multi) {
    if (selectedNodes.value.includes(nodeId)) {
      selectedNodes.value = selectedNodes.value.filter(id => id !== nodeId);
    } else {
      selectedNodes.value.push(nodeId);
    }
  } else {
    selectedNodes.value = [nodeId];
    selectedConnections.value = [];
  }
};

const selectConnection = (connectionId: string, multi = false) => {
  if (multi) {
    if (selectedConnections.value.includes(connectionId)) {
      selectedConnections.value = selectedConnections.value.filter(id => id !== connectionId);
    } else {
      selectedConnections.value.push(connectionId);
    }
  } else {
    selectedConnections.value = [connectionId];
    selectedNodes.value = [];
  }
};

const moveNode = (nodeId: string, position: [number, number]) => {
  const node = nodes.value.find(n => n.id === nodeId);
  if (node) {
    node.position = position;
    hasChanges.value = true;
  }
};

const deleteNode = (nodeId: string) => {
  nodes.value = nodes.value.filter(n => n.id !== nodeId);
  connections.value = connections.value.filter(
    c => c.sourceNodeId !== nodeId && c.targetNodeId !== nodeId
  );
  selectedNodes.value = selectedNodes.value.filter(id => id !== nodeId);
  hasChanges.value = true;
};

const duplicateNode = (nodeId: string) => {
  const node = nodes.value.find(n => n.id === nodeId);
  if (node) {
    const newNode = {
      ...node,
      id: `${node.type}_${Date.now()}`,
      name: `${node.name} Copy`,
      position: [node.position[0] + 100, node.position[1] + 100] as [number, number]
    };
    nodes.value.push(newNode);
    hasChanges.value = true;
  }
};

const deleteConnection = (connectionId: string) => {
  connections.value = connections.value.filter(c => c.id !== connectionId);
  selectedConnections.value = selectedConnections.value.filter(id => id !== connectionId);
  hasChanges.value = true;
};

const editNode = (nodeId: string) => {
  // Open node properties panel
  store.commit('ui/setSelectedNode', nodeId);
  store.commit('ui/setPropertyPanelVisible', true);
};

const startConnection = (nodeId: string, outputIndex: number, event: MouseEvent) => {
  dragConnection.active = true;
  dragConnection.sourceNodeId = nodeId;
  dragConnection.sourceOutput = outputIndex;
  
  const rect = canvasContainer.value?.getBoundingClientRect();
  if (rect) {
    dragConnection.start.x = event.clientX - rect.left;
    dragConnection.start.y = event.clientY - rect.top;
    dragConnection.current = { ...dragConnection.start };
  }
};

const endConnection = (nodeId: string, inputIndex: number) => {
  if (dragConnection.active && dragConnection.sourceNodeId !== nodeId) {
    const connection: Connection = {
      id: `${dragConnection.sourceNodeId}_${dragConnection.sourceOutput}_${nodeId}_${inputIndex}`,
      sourceNodeId: dragConnection.sourceNodeId,
      sourceOutput: dragConnection.sourceOutput,
      targetNodeId: nodeId,
      targetInput: inputIndex
    };
    
    // Check if connection already exists
    const existingConnection = connections.value.find(
      c => c.sourceNodeId === connection.sourceNodeId &&
           c.sourceOutput === connection.sourceOutput &&
           c.targetNodeId === connection.targetNodeId &&
           c.targetInput === connection.targetInput
    );
    
    if (!existingConnection) {
      connections.value.push(connection);
      hasChanges.value = true;
    }
  }
  
  dragConnection.active = false;
};

const createConnectionPath = (start: { x: number, y: number }, end: { x: number, y: number }): string => {
  const dx = end.x - start.x;
  const dy = end.y - start.y;
  const controlPointOffset = Math.min(Math.abs(dx) * 0.5, 100);
  
  return `M ${start.x} ${start.y} C ${start.x + controlPointOffset} ${start.y} ${end.x - controlPointOffset} ${end.y} ${end.x} ${end.y}`;
};

const getNodeExecutionData = (nodeId: string) => {
  // Return execution data for node
  return null;
};

// Canvas interaction methods
const onCanvasMouseDown = (event: MouseEvent) => {
  if (event.button === 0) { // Left click
    const rect = canvasContainer.value?.getBoundingClientRect();
    if (rect) {
      selectionBox.active = true;
      selectionBox.start.x = event.clientX - rect.left - canvasTransform.x;
      selectionBox.start.y = event.clientY - rect.top - canvasTransform.y;
      selectionBox.end = { ...selectionBox.start };
    }
  } else if (event.button === 2) { // Right click
    event.preventDefault();
    showContextMenu(event);
  }
};

const onCanvasMouseMove = (event: MouseEvent) => {
  if (dragConnection.active) {
    const rect = canvasContainer.value?.getBoundingClientRect();
    if (rect) {
      dragConnection.current.x = event.clientX - rect.left;
      dragConnection.current.y = event.clientY - rect.top;
    }
  } else if (selectionBox.active) {
    const rect = canvasContainer.value?.getBoundingClientRect();
    if (rect) {
      selectionBox.end.x = event.clientX - rect.left - canvasTransform.x;
      selectionBox.end.y = event.clientY - rect.top - canvasTransform.y;
    }
  }
};

const onCanvasMouseUp = (event: MouseEvent) => {
  if (selectionBox.active) {
    // Select nodes within selection box
    const boxLeft = Math.min(selectionBox.start.x, selectionBox.end.x);
    const boxTop = Math.min(selectionBox.start.y, selectionBox.end.y);
    const boxRight = Math.max(selectionBox.start.x, selectionBox.end.x);
    const boxBottom = Math.max(selectionBox.start.y, selectionBox.end.y);
    
    const selectedNodeIds = nodes.value.filter(node => {
      const [x, y] = node.position;
      return x >= boxLeft && x <= boxRight && y >= boxTop && y <= boxBottom;
    }).map(node => node.id);
    
    selectedNodes.value = selectedNodeIds;
    selectionBox.active = false;
  }
  
  if (dragConnection.active) {
    dragConnection.active = false;
  }
};

const onCanvasWheel = (event: WheelEvent) => {
  event.preventDefault();
  
  const zoomFactor = event.deltaY > 0 ? 0.9 : 1.1;
  const newScale = Math.max(0.1, Math.min(3, canvasTransform.scale * zoomFactor));
  
  const rect = canvasContainer.value?.getBoundingClientRect();
  if (rect) {
    const mouseX = event.clientX - rect.left;
    const mouseY = event.clientY - rect.top;
    
    const oldX = (mouseX - canvasTransform.x) / canvasTransform.scale;
    const oldY = (mouseY - canvasTransform.y) / canvasTransform.scale;
    
    canvasTransform.scale = newScale;
    canvasTransform.x = mouseX - oldX * newScale;
    canvasTransform.y = mouseY - oldY * newScale;
  }
};

const showContextMenu = (event: MouseEvent) => {
  contextMenu.visible = true;
  contextMenu.x = event.clientX;
  contextMenu.y = event.clientY;
};

const hideContextMenu = () => {
  contextMenu.visible = false;
};

// Zoom controls
const zoomIn = () => {
  canvasTransform.scale = Math.min(3, canvasTransform.scale * 1.2);
};

const zoomOut = () => {
  canvasTransform.scale = Math.max(0.1, canvasTransform.scale * 0.8);
};

const resetZoom = () => {
  canvasTransform.scale = 1;
  canvasTransform.x = 0;
  canvasTransform.y = 0;
};

const fitToView = () => {
  if (nodes.value.length === 0) return;
  
  const positions = nodes.value.map(n => n.position);
  const minX = Math.min(...positions.map(p => p[0]));
  const maxX = Math.max(...positions.map(p => p[0]));
  const minY = Math.min(...positions.map(p => p[1]));
  const maxY = Math.max(...positions.map(p => p[1]));
  
  const rect = canvasContainer.value?.getBoundingClientRect();
  if (rect) {
    const padding = 100;
    const contentWidth = maxX - minX + padding * 2;
    const contentHeight = maxY - minY + padding * 2;
    
    const scaleX = rect.width / contentWidth;
    const scaleY = rect.height / contentHeight;
    const scale = Math.min(scaleX, scaleY, 1);
    
    canvasTransform.scale = scale;
    canvasTransform.x = (rect.width - contentWidth * scale) / 2 - minX * scale + padding * scale;
    canvasTransform.y = (rect.height - contentHeight * scale) / 2 - minY * scale + padding * scale;
  }
};

// Context menu actions
const addNode = () => {
  store.commit('ui/setNodePaletteVisible', true);
  hideContextMenu();
};

const copySelection = () => {
  // Copy selected nodes to clipboard
  hideContextMenu();
};

const pasteNodes = () => {
  // Paste nodes from clipboard
  hideContextMenu();
};

const deleteSelection = () => {
  selectedNodes.value.forEach(nodeId => deleteNode(nodeId));
  selectedConnections.value.forEach(connectionId => deleteConnection(connectionId));
  hideContextMenu();
};

// Lifecycle
onMounted(async () => {
  // Load workflow data
  const workflowId = route.params.id as string;
  if (workflowId) {
    try {
      const workflowData = await store.dispatch('workflows/loadWorkflow', workflowId);
      workflow.value = workflowData;
      nodes.value = workflowData.nodes || [];
      connections.value = workflowData.connections || [];
    } catch (error) {
      console.error('Failed to load workflow:', error);
    }
  }
  
  // Add event listeners
  document.addEventListener('click', hideContextMenu);
  document.addEventListener('contextmenu', (e) => e.preventDefault());
});

onUnmounted(() => {
  document.removeEventListener('click', hideContextMenu);
  document.removeEventListener('contextmenu', (e) => e.preventDefault());
});

// Watch for changes
watch([nodes, connections], () => {
  hasChanges.value = true;
}, { deep: true });
</script>

<style scoped>
.workflow-canvas-container {
  display: flex;
  flex-direction: column;
  height: 100vh;
  background: #f8f9fa;
}

.canvas-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 1rem;
  background: white;
  border-bottom: 1px solid #e0e0e0;
  box-shadow: 0 2px 4px rgba(0,0,0,0.1);
}

.workflow-info {
  display: flex;
  align-items: center;
  gap: 1rem;
}

.workflow-title {
  margin: 0;
  color: #333;
  font-size: 1.25rem;
}

.status-indicator {
  padding: 0.25rem 0.75rem;
  border-radius: 1rem;
  font-size: 0.75rem;
  font-weight: 600;
  text-transform: uppercase;
}

.status-indicator.draft {
  background: #f0f0f0;
  color: #666;
}

.status-indicator.running {
  background: #e3f2fd;
  color: #1976d2;
}

.status-indicator.success {
  background: #e8f5e8;
  color: #4caf50;
}

.status-indicator.error {
  background: #ffebee;
  color: #f44336;
}

.canvas-controls {
  display: flex;
  gap: 0.5rem;
}

.btn {
  padding: 0.5rem 1rem;
  border: none;
  border-radius: 0.25rem;
  cursor: pointer;
  font-size: 0.875rem;
  transition: background-color 0.2s;
}

.btn:disabled {
  opacity: 0.6;
  cursor: not-allowed;
}

.btn-primary { background: #4a90e2; color: white; }
.btn-success { background: #7ed321; color: white; }
.btn-danger { background: #d0021b; color: white; }
.btn-secondary { background: #f5f5f5; color: #333; }
.btn-sm { padding: 0.25rem 0.5rem; }

.canvas-main {
  flex: 1;
  position: relative;
  overflow: hidden;
  background: #fafbfc;
}

.canvas-background {
  position: absolute;
  cursor: grab;
  transform-origin: 0 0;
}

.canvas-background:active {
  cursor: grabbing;
}

.canvas-grid {
  position: absolute;
  top: 0;
  left: 0;
  width: 100%;
  height: 100%;
  opacity: 0.5;
}

.connections-layer {
  position: absolute;
  top: 0;
  left: 0;
  width: 100%;
  height: 100%;
  pointer-events: none;
}

.nodes-layer {
  position: relative;
  width: 100%;
  height: 100%;
}

.selection-box {
  position: absolute;
  border: 2px dashed #4a90e2;
  background: rgba(74, 144, 226, 0.1);
  pointer-events: none;
}

.canvas-footer {
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 0.5rem 1rem;
  background: white;
  border-top: 1px solid #e0e0e0;
  font-size: 0.75rem;
  color: #666;
}

.canvas-info {
  display: flex;
  gap: 1rem;
}

.canvas-zoom-controls {
  display: flex;
  gap: 0.25rem;
}

.context-menu {
  position: fixed;
  background: white;
  border: 1px solid #e0e0e0;
  border-radius: 0.25rem;
  box-shadow: 0 4px 12px rgba(0,0,0,0.15);
  z-index: 1000;
  min-width: 150px;
}

.context-menu-item {
  padding: 0.5rem 1rem;
  cursor: pointer;
  display: flex;
  align-items: center;
  gap: 0.5rem;
  font-size: 0.875rem;
}

.context-menu-item:hover {
  background: #f5f5f5;
}

.context-menu-separator {
  height: 1px;
  background: #e0e0e0;
  margin: 0.25rem 0;
}

.minimap {
  position: absolute;
  bottom: 20px;
  right: 20px;
  width: 200px;
  height: 120px;
  background: rgba(255,255,255,0.9);
  border: 1px solid #e0e0e0;
  border-radius: 0.25rem;
  overflow: hidden;
}

.minimap-viewport {
  position: absolute;
  border: 2px solid #4a90e2;
  background: rgba(74, 144, 226, 0.1);
}

.minimap-content {
  width: 100%;
  height: 100%;
}
</style>