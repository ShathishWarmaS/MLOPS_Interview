/**
 * Core Workflow Engine for n8n MLOps Platform
 * Handles workflow execution, node management, and data flow
 */

import { EventEmitter } from 'events';
import { INodeType, IWorkflowBase, IExecuteWorkflowInfo, IRunExecutionData } from 'n8n-workflow';

export interface IWorkflowEngine {
  executeWorkflow(workflowData: IWorkflowBase, executionData?: IRunExecutionData): Promise<IExecuteWorkflowInfo>;
  validateWorkflow(workflowData: IWorkflowBase): Promise<boolean>;
  getAvailableNodes(): Promise<INodeType[]>;
  registerNode(nodeType: INodeType): void;
  unregisterNode(nodeName: string): void;
}

export interface IWorkflowExecution {
  id: string;
  workflowId: string;
  status: 'running' | 'success' | 'error' | 'waiting' | 'canceled';
  startTime: Date;
  endTime?: Date;
  data: IRunExecutionData;
  error?: string;
  metrics: {
    totalNodes: number;
    executedNodes: number;
    duration: number;
    memoryUsage: number;
    cpuUsage: number;
  };
}

export class WorkflowEngine extends EventEmitter implements IWorkflowEngine {
  private registeredNodes: Map<string, INodeType> = new Map();
  private activeExecutions: Map<string, IWorkflowExecution> = new Map();
  private executionQueue: string[] = [];
  private maxConcurrentExecutions: number = 10;

  constructor() {
    super();
    this.loadDefaultNodes();
  }

  /**
   * Execute a workflow with the given data
   */
  async executeWorkflow(
    workflowData: IWorkflowBase, 
    executionData?: IRunExecutionData
  ): Promise<IExecuteWorkflowInfo> {
    const executionId = this.generateExecutionId();
    
    try {
      // Validate workflow before execution
      const isValid = await this.validateWorkflow(workflowData);
      if (!isValid) {
        throw new Error('Workflow validation failed');
      }

      // Create execution record
      const execution: IWorkflowExecution = {
        id: executionId,
        workflowId: workflowData.id || 'unknown',
        status: 'running',
        startTime: new Date(),
        data: executionData || { resultData: { runData: {} } },
        metrics: {
          totalNodes: Object.keys(workflowData.nodes).length,
          executedNodes: 0,
          duration: 0,
          memoryUsage: 0,
          cpuUsage: 0
        }
      };

      this.activeExecutions.set(executionId, execution);
      this.emit('executionStarted', execution);

      // Execute workflow nodes
      const result = await this.executeWorkflowNodes(workflowData, execution);

      // Update execution status
      execution.status = 'success';
      execution.endTime = new Date();
      execution.metrics.duration = execution.endTime.getTime() - execution.startTime.getTime();

      this.emit('executionCompleted', execution);

      return result;

    } catch (error) {
      // Handle execution error
      const execution = this.activeExecutions.get(executionId);
      if (execution) {
        execution.status = 'error';
        execution.endTime = new Date();
        execution.error = error instanceof Error ? error.message : String(error);
        execution.metrics.duration = execution.endTime.getTime() - execution.startTime.getTime();
      }

      this.emit('executionFailed', execution, error);
      throw error;

    } finally {
      // Cleanup
      setTimeout(() => {
        this.activeExecutions.delete(executionId);
      }, 300000); // Keep execution data for 5 minutes
    }
  }

  /**
   * Validate workflow structure and node configurations
   */
  async validateWorkflow(workflowData: IWorkflowBase): Promise<boolean> {
    try {
      // Check if workflow has nodes
      if (!workflowData.nodes || Object.keys(workflowData.nodes).length === 0) {
        throw new Error('Workflow must contain at least one node');
      }

      // Validate each node
      for (const [nodeId, node] of Object.entries(workflowData.nodes)) {
        const nodeType = this.registeredNodes.get(node.type);
        if (!nodeType) {
          throw new Error(`Unknown node type: ${node.type}`);
        }

        // Validate node parameters
        if (nodeType.description.properties) {
          for (const property of nodeType.description.properties) {
            if (property.required && !node.parameters?.[property.name]) {
              throw new Error(`Required parameter '${property.name}' missing in node '${nodeId}'`);
            }
          }
        }
      }

      // Validate connections
      if (workflowData.connections) {
        for (const [sourceNode, connections] of Object.entries(workflowData.connections)) {
          if (!workflowData.nodes[sourceNode]) {
            throw new Error(`Source node '${sourceNode}' not found in workflow`);
          }

          for (const [outputIndex, outputs] of Object.entries(connections)) {
            for (const output of outputs || []) {
              if (!workflowData.nodes[output.node]) {
                throw new Error(`Target node '${output.node}' not found in workflow`);
              }
            }
          }
        }
      }

      return true;

    } catch (error) {
      this.emit('validationError', { workflowData, error });
      return false;
    }
  }

  /**
   * Get all available node types
   */
  async getAvailableNodes(): Promise<INodeType[]> {
    return Array.from(this.registeredNodes.values());
  }

  /**
   * Register a new node type
   */
  registerNode(nodeType: INodeType): void {
    this.registeredNodes.set(nodeType.description.name, nodeType);
    this.emit('nodeRegistered', nodeType);
  }

  /**
   * Unregister a node type
   */
  unregisterNode(nodeName: string): void {
    const nodeType = this.registeredNodes.get(nodeName);
    if (nodeType) {
      this.registeredNodes.delete(nodeName);
      this.emit('nodeUnregistered', nodeType);
    }
  }

  /**
   * Execute workflow nodes in proper order
   */
  private async executeWorkflowNodes(
    workflowData: IWorkflowBase, 
    execution: IWorkflowExecution
  ): Promise<IExecuteWorkflowInfo> {
    const executionOrder = this.calculateExecutionOrder(workflowData);
    const nodeResults: { [key: string]: any } = {};

    for (const nodeId of executionOrder) {
      const node = workflowData.nodes[nodeId];
      const nodeType = this.registeredNodes.get(node.type);

      if (!nodeType) {
        throw new Error(`Node type '${node.type}' not found`);
      }

      try {
        // Prepare node execution context
        const executionContext = this.createExecutionContext(
          node, 
          nodeResults, 
          workflowData, 
          execution
        );

        // Execute node
        const startTime = process.hrtime.bigint();
        const result = await this.executeNode(nodeType, executionContext);
        const endTime = process.hrtime.bigint();

        // Store result
        nodeResults[nodeId] = result;
        execution.metrics.executedNodes++;

        // Track performance
        const executionTime = Number(endTime - startTime) / 1e6; // Convert to milliseconds
        this.emit('nodeExecuted', {
          nodeId,
          nodeType: node.type,
          executionTime,
          result
        });

      } catch (error) {
        this.emit('nodeExecutionFailed', {
          nodeId,
          nodeType: node.type,
          error
        });
        throw new Error(`Node '${nodeId}' execution failed: ${error}`);
      }
    }

    return {
      finished: true,
      data: { resultData: { runData: nodeResults } }
    };
  }

  /**
   * Calculate the execution order of nodes based on dependencies
   */
  private calculateExecutionOrder(workflowData: IWorkflowBase): string[] {
    const nodes = Object.keys(workflowData.nodes);
    const visited = new Set<string>();
    const order: string[] = [];

    const visit = (nodeId: string) => {
      if (visited.has(nodeId)) return;
      visited.add(nodeId);

      // Visit dependencies first
      const connections = workflowData.connections?.[nodeId];
      if (connections) {
        for (const outputConnections of Object.values(connections)) {
          for (const connection of outputConnections || []) {
            visit(connection.node);
          }
        }
      }

      order.push(nodeId);
    };

    // Find start nodes (nodes with no incoming connections)
    const startNodes = nodes.filter(nodeId => {
      return !Object.values(workflowData.connections || {}).some(outputs =>
        Object.values(outputs).some(connections =>
          connections?.some(conn => conn.node === nodeId)
        )
      );
    });

    // If no start nodes found, start with the first node
    if (startNodes.length === 0 && nodes.length > 0) {
      startNodes.push(nodes[0]);
    }

    // Visit all start nodes
    for (const startNode of startNodes) {
      visit(startNode);
    }

    return order.reverse();
  }

  /**
   * Create execution context for a node
   */
  private createExecutionContext(
    node: any, 
    nodeResults: any, 
    workflowData: IWorkflowBase, 
    execution: IWorkflowExecution
  ): any {
    return {
      getInputData: () => {
        // Get input data from previous nodes
        const inputData = [];
        // Implementation would gather data from connected nodes
        return inputData;
      },
      getNodeParameter: (parameterName: string) => {
        return node.parameters?.[parameterName];
      },
      getCredentials: (credentialType: string) => {
        // Return credentials for the node
        return {};
      },
      helpers: {
        httpRequest: async (options: any) => {
          // HTTP request helper
          const fetch = (await import('node-fetch')).default;
          const response = await fetch(options.url, options);
          return response.json();
        }
      },
      executionId: execution.id,
      workflowId: execution.workflowId
    };
  }

  /**
   * Execute individual node
   */
  private async executeNode(nodeType: INodeType, context: any): Promise<any> {
    if (typeof nodeType.execute === 'function') {
      return await nodeType.execute.call(context);
    }
    throw new Error('Node type does not have execute method');
  }

  /**
   * Load default built-in nodes
   */
  private loadDefaultNodes(): void {
    // Register built-in nodes
    const defaultNodes = [
      // Data nodes
      require('./nodes/DataValidation.node'),
      require('./nodes/FeatureEngineering.node'),
      
      // ML nodes
      require('./nodes/ModelTraining.node'),
      require('./nodes/ModelDeployment.node'),
      require('./nodes/ModelMonitoring.node'),
      
      // Utility nodes
      require('./nodes/HttpRequest.node'),
      require('./nodes/DatabaseQuery.node'),
      require('./nodes/FileOperation.node')
    ];

    for (const NodeClass of defaultNodes) {
      try {
        const nodeInstance = new NodeClass();
        this.registerNode(nodeInstance);
      } catch (error) {
        console.warn(`Failed to load default node: ${error}`);
      }
    }
  }

  /**
   * Generate unique execution ID
   */
  private generateExecutionId(): string {
    return `exec_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
  }

  /**
   * Get execution status
   */
  getExecution(executionId: string): IWorkflowExecution | undefined {
    return this.activeExecutions.get(executionId);
  }

  /**
   * Get all active executions
   */
  getActiveExecutions(): IWorkflowExecution[] {
    return Array.from(this.activeExecutions.values());
  }

  /**
   * Cancel workflow execution
   */
  async cancelExecution(executionId: string): Promise<boolean> {
    const execution = this.activeExecutions.get(executionId);
    if (execution && execution.status === 'running') {
      execution.status = 'canceled';
      execution.endTime = new Date();
      this.emit('executionCanceled', execution);
      return true;
    }
    return false;
  }

  /**
   * Get workflow engine statistics
   */
  getStatistics(): {
    totalExecutions: number;
    activeExecutions: number;
    registeredNodes: number;
    averageExecutionTime: number;
  } {
    const executions = Array.from(this.activeExecutions.values());
    const completedExecutions = executions.filter(e => e.status === 'success' || e.status === 'error');
    
    return {
      totalExecutions: executions.length,
      activeExecutions: executions.filter(e => e.status === 'running').length,
      registeredNodes: this.registeredNodes.size,
      averageExecutionTime: completedExecutions.length > 0 
        ? completedExecutions.reduce((sum, e) => sum + e.metrics.duration, 0) / completedExecutions.length 
        : 0
    };
  }
}

export default WorkflowEngine;