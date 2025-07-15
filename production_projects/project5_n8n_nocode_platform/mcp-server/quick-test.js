// Quick test to verify the MCP server structure
import { spawn } from 'child_process';
import { promises as fs } from 'fs';
import path from 'path';

async function testTypeScriptCompilation() {
  console.log('🔍 Testing TypeScript compilation...');
  
  return new Promise((resolve, reject) => {
    const tsc = spawn('npx', ['tsc', '--noEmit', '--skipLibCheck'], {
      stdio: 'pipe',
      shell: true
    });
    
    let stdout = '';
    let stderr = '';
    
    tsc.stdout.on('data', (data) => {
      stdout += data.toString();
    });
    
    tsc.stderr.on('data', (data) => {
      stderr += data.toString();
    });
    
    tsc.on('close', (code) => {
      if (code === 0) {
        console.log('✅ TypeScript compilation successful');
        resolve(true);
      } else {
        console.log('❌ TypeScript compilation failed:');
        console.log(stderr);
        reject(new Error('TypeScript compilation failed'));
      }
    });
  });
}

async function testPackageInstall() {
  console.log('📦 Testing package installation...');
  
  try {
    // Check if node_modules exists
    const nodeModulesPath = path.join(process.cwd(), 'node_modules');
    await fs.access(nodeModulesPath);
    console.log('✅ node_modules directory exists');
    
    // Check if key packages are installed
    const keyPackages = [
      '@modelcontextprotocol/sdk',
      'openai',
      '@anthropic-ai/sdk',
      'typescript',
      'tsx'
    ];
    
    for (const pkg of keyPackages) {
      try {
        const pkgPath = path.join(nodeModulesPath, pkg);
        await fs.access(pkgPath);
        console.log(`✅ ${pkg} is installed`);
      } catch (error) {
        console.log(`❌ ${pkg} is NOT installed`);
      }
    }
    
    return true;
  } catch (error) {
    console.log('❌ node_modules directory not found');
    console.log('Please run: npm install');
    return false;
  }
}

async function testConfigFile() {
  console.log('⚙️  Testing configuration...');
  
  try {
    // Check if .env file exists
    await fs.access('.env');
    console.log('✅ .env file exists');
    
    // Read .env file
    const envContent = await fs.readFile('.env', 'utf8');
    const requiredVars = [
      'OPENAI_API_KEY',
      'ANTHROPIC_API_KEY',
      'GOOGLE_API_KEY',
      'PERPLEXITY_API_KEY',
      'JWT_SECRET'
    ];
    
    for (const varName of requiredVars) {
      if (envContent.includes(`${varName}=your_`) || !envContent.includes(varName)) {
        console.log(`⚠️  ${varName} needs to be configured`);
      } else {
        console.log(`✅ ${varName} is configured`);
      }
    }
    
    return true;
  } catch (error) {
    console.log('❌ .env file not found');
    return false;
  }
}

async function main() {
  console.log('🚀 MCP Server Quick Test');
  console.log('========================\n');
  
  try {
    // Test 1: Check packages
    const packagesOk = await testPackageInstall();
    console.log();
    
    // Test 2: Check configuration
    const configOk = await testConfigFile();
    console.log();
    
    // Test 3: TypeScript compilation (only if packages are installed)
    if (packagesOk) {
      await testTypeScriptCompilation();
    } else {
      console.log('⏭️  Skipping TypeScript test (packages not installed)');
    }
    
    console.log('\n🎉 Quick test completed!');
    
    if (!packagesOk) {
      console.log('\n⚠️  Next step: npm install');
    } else if (!configOk) {
      console.log('\n⚠️  Next step: Configure API keys in .env file');
    } else {
      console.log('\n✅ Ready to start the server with: npm run dev');
    }
    
  } catch (error) {
    console.error('❌ Test failed:', error.message);
    process.exit(1);
  }
}

main();