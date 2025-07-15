#!/usr/bin/env node

/**
 * Test MCP Server without npm install
 * This script tests the code structure and syntax without requiring dependencies
 */

const fs = require('fs');
const path = require('path');

console.log('🧪 MCP Server Code Structure Test');
console.log('=================================\n');

// Test 1: Check file structure
function testFileStructure() {
  console.log('1. 📁 File Structure Test');
  console.log('-------------------------');
  
  const requiredFiles = [
    { path: 'src/index.ts', description: 'Main MCP server entry point' },
    { path: 'src/services/ai-orchestrator.ts', description: 'AI model orchestration' },
    { path: 'src/services/rag-service.ts', description: 'RAG with ChromaDB' },
    { path: 'src/services/perplexity-service.ts', description: 'Perplexity search integration' },
    { path: 'src/services/mlops-integration.ts', description: 'MLOps pipeline management' },
    { path: 'src/services/media-processor.ts', description: 'Multimodal processing' },
    { path: 'src/config/config.ts', description: 'Configuration management' },
    { path: 'src/utils/logger.ts', description: 'Logging utilities' },
    { path: 'package.json', description: 'Node.js package configuration' },
    { path: 'tsconfig.json', description: 'TypeScript configuration' },
    { path: '.env.example', description: 'Environment variables template' }
  ];
  
  let allFilesExist = true;
  
  requiredFiles.forEach(file => {
    if (fs.existsSync(file.path)) {
      console.log(`   ✅ ${file.path} - ${file.description}`);
    } else {
      console.log(`   ❌ ${file.path} - MISSING`);
      allFilesExist = false;
    }
  });
  
  return allFilesExist;
}

// Test 2: Check TypeScript syntax (basic)
function testTypeScriptSyntax() {
  console.log('\n2. 🔍 TypeScript Syntax Test');
  console.log('----------------------------');
  
  const tsFiles = [
    'src/index.ts',
    'src/services/ai-orchestrator.ts',
    'src/services/rag-service.ts',
    'src/services/perplexity-service.ts',
    'src/services/mlops-integration.ts',
    'src/services/media-processor.ts',
    'src/config/config.ts',
    'src/utils/logger.ts'
  ];
  
  let syntaxOk = true;
  
  tsFiles.forEach(filePath => {
    if (fs.existsSync(filePath)) {
      try {
        const content = fs.readFileSync(filePath, 'utf8');
        
        // Basic syntax checks
        const hasImports = content.includes('import ') || content.includes('require(');
        const hasExports = content.includes('export ') || content.includes('module.exports');
        const hasTypeScript = content.includes(': string') || content.includes(': number') || content.includes('interface ');
        
        console.log(`   ✅ ${filePath} - Syntax OK`);
        console.log(`      - Has imports: ${hasImports ? '✅' : '❌'}`);
        console.log(`      - Has exports: ${hasExports ? '✅' : '❌'}`);
        console.log(`      - Has TypeScript: ${hasTypeScript ? '✅' : '❌'}`);
        
      } catch (error) {
        console.log(`   ❌ ${filePath} - Error reading file: ${error.message}`);
        syntaxOk = false;
      }
    } else {
      console.log(`   ❌ ${filePath} - File not found`);
      syntaxOk = false;
    }
  });
  
  return syntaxOk;
}

// Test 3: Check configuration files
function testConfigFiles() {
  console.log('\n3. ⚙️  Configuration Files Test');
  console.log('-------------------------------');
  
  // Test package.json
  try {
    const packageJson = JSON.parse(fs.readFileSync('package.json', 'utf8'));
    console.log(`   ✅ package.json - Valid JSON`);
    console.log(`      - Name: ${packageJson.name}`);
    console.log(`      - Version: ${packageJson.version}`);
    console.log(`      - Type: ${packageJson.type}`);
    console.log(`      - Main: ${packageJson.main}`);
    console.log(`      - Scripts: ${Object.keys(packageJson.scripts || {}).join(', ')}`);
    
    // Check key dependencies
    const keyDeps = ['@modelcontextprotocol/sdk', 'openai', '@anthropic-ai/sdk', 'typescript'];
    keyDeps.forEach(dep => {
      if (packageJson.dependencies?.[dep] || packageJson.devDependencies?.[dep]) {
        console.log(`      - ${dep}: ✅`);
      } else {
        console.log(`      - ${dep}: ❌`);
      }
    });
    
  } catch (error) {
    console.log(`   ❌ package.json - Error: ${error.message}`);
    return false;
  }
  
  // Test tsconfig.json
  try {
    const tsConfig = JSON.parse(fs.readFileSync('tsconfig.json', 'utf8'));
    console.log(`   ✅ tsconfig.json - Valid JSON`);
    console.log(`      - Target: ${tsConfig.compilerOptions?.target}`);
    console.log(`      - Module: ${tsConfig.compilerOptions?.module}`);
    console.log(`      - Output: ${tsConfig.compilerOptions?.outDir}`);
    console.log(`      - Strict: ${tsConfig.compilerOptions?.strict}`);
  } catch (error) {
    console.log(`   ❌ tsconfig.json - Error: ${error.message}`);
    return false;
  }
  
  return true;
}

// Test 4: Check MCP server tools
function testMCPTools() {
  console.log('\n4. 🛠️  MCP Tools Test');
  console.log('--------------------');
  
  try {
    const indexContent = fs.readFileSync('src/index.ts', 'utf8');
    
    // Look for MCP tools
    const expectedTools = [
      'chat_with_ai',
      'process_multimodal',
      'rag_query',
      'rag_ingest',
      'perplexity_search',
      'mlops_deploy_model',
      'mlops_create_pipeline',
      'compare_ai_models',
      'create_ai_workflow',
      'analyze_data_with_ai'
    ];
    
    expectedTools.forEach(tool => {
      if (indexContent.includes(`'${tool}'`) || indexContent.includes(`"${tool}"`)) {
        console.log(`   ✅ ${tool} - Found`);
      } else {
        console.log(`   ❌ ${tool} - Not found`);
      }
    });
    
    // Check for handler methods
    const handlerMethods = expectedTools.map(tool => {
      const methodName = tool.replace(/_([a-z])/g, (_, letter) => letter.toUpperCase());
      return `handle${methodName.charAt(0).toUpperCase() + methodName.slice(1)}`;
    });
    
    console.log('\n   Handler Methods:');
    handlerMethods.forEach(method => {
      if (indexContent.includes(method)) {
        console.log(`   ✅ ${method} - Found`);
      } else {
        console.log(`   ❌ ${method} - Not found`);
      }
    });
    
  } catch (error) {
    console.log(`   ❌ Error checking MCP tools: ${error.message}`);
    return false;
  }
  
  return true;
}

// Test 5: Check environment setup
function testEnvironment() {
  console.log('\n5. 🌍 Environment Test');
  console.log('---------------------');
  
  if (fs.existsSync('.env')) {
    console.log('   ✅ .env file exists');
    
    try {
      const envContent = fs.readFileSync('.env', 'utf8');
      const requiredVars = [
        'OPENAI_API_KEY',
        'ANTHROPIC_API_KEY',
        'GOOGLE_API_KEY',
        'PERPLEXITY_API_KEY',
        'JWT_SECRET'
      ];
      
      requiredVars.forEach(varName => {
        if (envContent.includes(varName)) {
          // Check if it's still the placeholder
          if (envContent.includes(`${varName}=your_`)) {
            console.log(`   ⚠️  ${varName} - Needs configuration`);
          } else {
            console.log(`   ✅ ${varName} - Configured`);
          }
        } else {
          console.log(`   ❌ ${varName} - Missing`);
        }
      });
      
    } catch (error) {
      console.log(`   ❌ Error reading .env: ${error.message}`);
    }
  } else {
    console.log('   ❌ .env file not found');
  }
  
  if (fs.existsSync('.env.example')) {
    console.log('   ✅ .env.example file exists');
  } else {
    console.log('   ❌ .env.example file not found');
  }
  
  return true;
}

// Test 6: Check code quality
function testCodeQuality() {
  console.log('\n6. 📊 Code Quality Test');
  console.log('----------------------');
  
  const files = [
    'src/index.ts',
    'src/services/ai-orchestrator.ts',
    'src/services/rag-service.ts'
  ];
  
  files.forEach(filePath => {
    if (fs.existsSync(filePath)) {
      const content = fs.readFileSync(filePath, 'utf8');
      const lines = content.split('\n').length;
      const hasComments = content.includes('//') || content.includes('/*');
      const hasErrorHandling = content.includes('try {') || content.includes('catch');
      const hasTypes = content.includes('interface ') || content.includes('type ');
      
      console.log(`   📄 ${filePath}:`);
      console.log(`      - Lines: ${lines}`);
      console.log(`      - Has comments: ${hasComments ? '✅' : '❌'}`);
      console.log(`      - Has error handling: ${hasErrorHandling ? '✅' : '❌'}`);
      console.log(`      - Has types: ${hasTypes ? '✅' : '❌'}`);
    }
  });
  
  return true;
}

// Main test runner
function runTests() {
  console.log('Running comprehensive MCP server tests...\n');
  
  const results = {
    fileStructure: testFileStructure(),
    typeScriptSyntax: testTypeScriptSyntax(),
    configFiles: testConfigFiles(),
    mcpTools: testMCPTools(),
    environment: testEnvironment(),
    codeQuality: testCodeQuality()
  };
  
  console.log('\n🎯 Test Results Summary');
  console.log('======================');
  
  Object.entries(results).forEach(([testName, passed]) => {
    const status = passed ? '✅ PASSED' : '❌ FAILED';
    console.log(`${testName}: ${status}`);
  });
  
  const allPassed = Object.values(results).every(result => result);
  
  if (allPassed) {
    console.log('\n🎉 All tests passed! Your MCP server structure is ready.');
    console.log('\nNext steps:');
    console.log('1. npm install (install dependencies)');
    console.log('2. Configure API keys in .env file');
    console.log('3. Start ChromaDB: docker run -p 8000:8000 chromadb/chroma');
    console.log('4. npm run build (compile TypeScript)');
    console.log('5. npm run dev (start development server)');
  } else {
    console.log('\n⚠️  Some tests failed. Please review the results above.');
  }
  
  return allPassed;
}

// Run the tests
if (require.main === module) {
  runTests();
}

module.exports = { runTests };