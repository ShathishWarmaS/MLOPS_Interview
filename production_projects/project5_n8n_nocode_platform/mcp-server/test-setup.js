#!/usr/bin/env node

/**
 * MCP Server Test Setup
 * This script helps you test the MCP server functionality step by step
 */

console.log('🚀 MCP Server Test Setup');
console.log('========================');

// Test 1: Check Node.js version
console.log('\n1. Node.js Version Check:');
console.log(`   Node.js: ${process.version}`);
console.log(`   Required: >= 18.0.0`);

const nodeVersion = process.version.replace('v', '').split('.').map(Number);
if (nodeVersion[0] >= 18) {
    console.log('   ✅ Node.js version is compatible');
} else {
    console.log('   ❌ Node.js version is too old. Please upgrade to Node.js 18+');
    process.exit(1);
}

// Test 2: Check if required files exist
console.log('\n2. File Structure Check:');
const fs = require('fs');
const path = require('path');

const requiredFiles = [
    'package.json',
    'tsconfig.json',
    '.env.example',
    'src/index.ts',
    'src/services/ai-orchestrator.ts',
    'src/services/rag-service.ts',
    'src/services/perplexity-service.ts',
    'src/services/mlops-integration.ts',
    'src/services/media-processor.ts',
    'src/config/config.ts',
    'src/utils/logger.ts'
];

let allFilesExist = true;
requiredFiles.forEach(file => {
    if (fs.existsSync(file)) {
        console.log(`   ✅ ${file}`);
    } else {
        console.log(`   ❌ ${file} - MISSING`);
        allFilesExist = false;
    }
});

if (!allFilesExist) {
    console.log('\n❌ Some required files are missing. Please check the project structure.');
    process.exit(1);
}

// Test 3: Check package.json
console.log('\n3. Package.json Check:');
try {
    const packageJson = JSON.parse(fs.readFileSync('package.json', 'utf8'));
    console.log(`   ✅ Package name: ${packageJson.name}`);
    console.log(`   ✅ Version: ${packageJson.version}`);
    console.log(`   ✅ Main entry: ${packageJson.main}`);
    console.log(`   ✅ Scripts available: ${Object.keys(packageJson.scripts).join(', ')}`);
} catch (error) {
    console.log('   ❌ Error reading package.json:', error.message);
    process.exit(1);
}

// Test 4: Environment setup
console.log('\n4. Environment Setup:');
if (fs.existsSync('.env')) {
    console.log('   ✅ .env file exists');
} else {
    console.log('   ⚠️  .env file not found. Creating from .env.example...');
    try {
        fs.copyFileSync('.env.example', '.env');
        console.log('   ✅ .env file created from .env.example');
        console.log('   ⚠️  Please edit .env file with your actual API keys');
    } catch (error) {
        console.log('   ❌ Error creating .env file:', error.message);
    }
}

// Test 5: TypeScript check
console.log('\n5. TypeScript Configuration:');
try {
    const tsConfig = JSON.parse(fs.readFileSync('tsconfig.json', 'utf8'));
    console.log(`   ✅ TypeScript target: ${tsConfig.compilerOptions.target}`);
    console.log(`   ✅ Module system: ${tsConfig.compilerOptions.module}`);
    console.log(`   ✅ Output directory: ${tsConfig.compilerOptions.outDir}`);
} catch (error) {
    console.log('   ❌ Error reading tsconfig.json:', error.message);
}

console.log('\n🎯 Next Steps:');
console.log('==============');
console.log('1. Install dependencies:');
console.log('   npm install');
console.log('');
console.log('2. Edit .env file with your API keys:');
console.log('   - OPENAI_API_KEY=your_openai_key');
console.log('   - ANTHROPIC_API_KEY=your_anthropic_key');
console.log('   - GOOGLE_API_KEY=your_google_key');
console.log('   - PERPLEXITY_API_KEY=your_perplexity_key');
console.log('   - JWT_SECRET=your_jwt_secret');
console.log('');
console.log('3. Start external services:');
console.log('   docker run -p 8000:8000 chromadb/chroma');
console.log('');
console.log('4. Test TypeScript compilation:');
console.log('   npm run build');
console.log('');
console.log('5. Run the server:');
console.log('   npm run dev');
console.log('');
console.log('6. Test with MCP client or tools');

console.log('\n📋 Testing Checklist:');
console.log('=====================');
console.log('□ Install dependencies (npm install)');
console.log('□ Configure API keys in .env file');
console.log('□ Start ChromaDB container');
console.log('□ Build TypeScript code (npm run build)');
console.log('□ Run development server (npm run dev)');
console.log('□ Test MCP tools functionality');
console.log('□ Check server logs for errors');