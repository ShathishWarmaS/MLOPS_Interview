const express = require('express');
const cors = require('cors');
const path = require('path');
const WebSocket = require('ws');

const app = express();
const port = 8080;

// Enable CORS
app.use(cors());
app.use(express.json());

// Serve static files
app.use(express.static(__dirname));

// API endpoints for demo
app.get('/api/nodes', (req, res) => {
  res.json({
    nodes: [
      {
        type: 'dataSource',
        name: 'Data Source',
        description: 'Load data from files, databases, APIs',
        icon: '📊',
        category: 'Data'
      },
      {
        type: 'dataValidation',
        name: 'Data Validation',
        description: 'Validate data quality and schema',
        icon: '✅',
        category: 'Data'
      },
      {
        type: 'featureEngineering',
        name: 'Feature Engineering',
        description: 'Transform and engineer features',
        icon: '🔧',
        category: 'ML'
      },
      {
        type: 'modelTraining',
        name: 'Model Training',
        description: 'Train ML models with various algorithms',
        icon: '🧠',
        category: 'ML'
      },
      {
        type: 'modelEvaluation',
        name: 'Model Evaluation',
        description: 'Evaluate model performance',
        icon: '📈',
        category: 'ML'
      },
      {
        type: 'modelDeployment',
        name: 'Model Deployment',
        description: 'Deploy models to production',
        icon: '🚀',
        category: 'Deployment'
      }
    ]
  });
});

app.post('/api/workflows/execute', (req, res) => {
  const { nodes, connections } = req.body;
  
  // Simulate workflow execution
  setTimeout(() => {
    res.json({
      executionId: `exec_${Date.now()}`,
      status: 'completed',
      results: {
        totalNodes: nodes.length,
        successfulNodes: nodes.length,
        executionTime: Math.random() * 30000 + 5000,
        metrics: {
          accuracy: 0.92 + Math.random() * 0.05,
          precision: 0.89 + Math.random() * 0.08,
          recall: 0.88 + Math.random() * 0.09
        }
      }
    });
  }, 2000);
});

app.get('/api/workflows/templates', (req, res) => {
  res.json({
    templates: [
      {
        id: 'ml-training-pipeline',
        name: 'ML Training Pipeline',
        description: 'Complete machine learning training workflow',
        nodes: 8,
        category: 'Machine Learning'
      },
      {
        id: 'data-validation-pipeline',
        name: 'Data Validation Pipeline',
        description: 'Automated data quality validation',
        nodes: 5,
        category: 'Data Quality'
      },
      {
        id: 'model-deployment-pipeline',
        name: 'Model Deployment Pipeline',
        description: 'Automated model deployment and monitoring',
        nodes: 6,
        category: 'Deployment'
      }
    ]
  });
});

// WebSocket for real-time updates
const server = app.listen(port, () => {
  console.log(`🚀 n8n MLOps Demo Server running at http://localhost:${port}`);
  console.log(`📱 Open http://localhost:${port}/simple-n8n-demo.html to see the demo`);
});

const wss = new WebSocket.Server({ server });

wss.on('connection', (ws) => {
  console.log('🔗 Client connected');
  
  ws.on('message', (message) => {
    try {
      const data = JSON.parse(message);
      
      if (data.type === 'workflow_execute') {
        // Simulate real-time execution updates
        const executionSteps = [
          'Initializing workflow...',
          'Loading data source...',
          'Validating data quality...',
          'Engineering features...',
          'Training model...',
          'Evaluating performance...',
          'Deploying model...',
          'Execution completed!'
        ];
        
        executionSteps.forEach((step, index) => {
          setTimeout(() => {
            ws.send(JSON.stringify({
              type: 'execution_update',
              step: index + 1,
              total: executionSteps.length,
              message: step,
              progress: ((index + 1) / executionSteps.length) * 100
            }));
          }, index * 1000);
        });
      }
    } catch (error) {
      console.error('WebSocket message error:', error);
    }
  });
  
  ws.on('close', () => {
    console.log('❌ Client disconnected');
  });
});

// Graceful shutdown
process.on('SIGTERM', () => {
  console.log('🛑 Server shutting down...');
  server.close(() => {
    console.log('✅ Server closed');
  });
});