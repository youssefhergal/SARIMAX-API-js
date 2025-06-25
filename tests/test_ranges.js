import * as math from 'mathjs';
import fs from 'fs';
import bvh from 'bvh-parser';

// StandardScaler class
class StandardScaler {
  constructor() {
    this.mean = null;
    this.std = null;
  }

  fit(data) {
    const matrix = math.matrix(data);
    this.mean = math.mean(matrix, 0)._data;
    
    const numCols = data[0].length;
    this.std = new Array(numCols);
    
    for (let col = 0; col < numCols; col++) {
      const columnData = data.map(row => row[col]);
      const meanVal = this.mean[col];
      const variance = columnData.reduce((sum, val) => sum + Math.pow(val - meanVal, 2), 0) / columnData.length;
      this.std[col] = Math.sqrt(variance);
      if (this.std[col] === 0) this.std[col] = 1;
    }
    
    return this;
  }

  transform(data) {
    if (!this.mean || !this.std) throw new Error("Scaler not fitted");
    return data.map(row => 
      row.map((val, i) => (val - this.mean[i]) / this.std[i])
    );
  }

  fitTransform(data) {
    return this.fit(data).transform(data);
  }

  inverseTransform(data) {
    if (!this.mean || !this.std) throw new Error("Scaler not fitted");
    return data.map(row => 
      row.map((val, i) => val * this.std[i] + this.mean[i])
    );
  }
}

// Parse BVH header
function parseBVHHeader(bvhContent) {
  const lines = bvhContent.split('\n');
  const channels = [];
  
  let currentJoint = null;
  let inMotionSection = false;
  
  for (let i = 0; i < lines.length; i++) {
    const line = lines[i].trim();
    
    if (line === 'MOTION') {
      inMotionSection = true;
      break;
    }
    
    if (line.startsWith('ROOT ') || line.startsWith('JOINT ')) {
      const parts = line.split(/\s+/);
      currentJoint = parts[1];
    } else if (line.startsWith('CHANNELS ')) {
      const parts = line.split(/\s+/);
      const numChannels = parseInt(parts[1]);
      const channelTypes = parts.slice(2, 2 + numChannels);
      
      channelTypes.forEach(channelType => {
        channels.push(`${currentJoint}_${channelType}`);
      });
    }
  }
  
  return { channels };
}

// Extract data from BVH
function extractDataFromBVH(path, targetJoint, exogJoints) {
  try {
    console.log(`📂 Loading BVH file: ${path}`);
    const bvhContent = fs.readFileSync(path, 'utf-8');
    
    const headerInfo = parseBVHHeader(bvhContent);
    console.log(`✅ Parsed header: ${headerInfo.channels.length} channels found`);
    
    const result = bvh(bvhContent);
    const frames = result?.frames;
    
    if (!frames || frames.length === 0) {
      throw new Error('No frames found in BVH file');
    }
    
    console.log(`📊 Found ${frames.length} frames`);
    
    const targetIndex = headerInfo.channels.findIndex(ch => ch === targetJoint);
    if (targetIndex === -1) {
      throw new Error(`Target joint "${targetJoint}" not found`);
    }
    
    const exogIndices = exogJoints.map(joint => {
      const index = headerInfo.channels.findIndex(ch => ch === joint);
      return index === -1 ? -1 : index;
    });
    
    const endog = frames.map(frame => frame[targetIndex]);
    const exog = frames.map(frame => 
      exogIndices.map(idx => idx === -1 ? 0 : frame[idx])
    );
    
    return { endog, exog };
    
  } catch (error) {
    console.error('❌ Error loading BVH file:', error.message);
    throw error;
  }
}

// Test the ranges
console.log("🧪 Testing y-axis ranges with StandardScaler...\n");

try {
  // Load glassblowing data
  const bvhTestG = extractDataFromBVH(
    './BVH/Glassblowing/Test_Glassblowing.bvh',
    'LeftForeArm_Yrotation',
    ['Hips_Xrotation', 'Spine_Yrotation', 'Spine_Zrotation', 'RightForeArm_Yrotation']
  );

  // Create test data
  const testDataG = bvhTestG.endog.map((endogValue, i) => [
    bvhTestG.exog[i][0], // Hips_Xrotation
    bvhTestG.exog[i][1], // Spine_Yrotation 
    bvhTestG.exog[i][2], // Spine_Zrotation
    endogValue,           // LeftForeArm_Yrotation
    bvhTestG.exog[i][3]   // RightForeArm_Yrotation
  ]);

  console.log(`\n📊 Original LeftForeArm_Yrotation data range:`);
  const originalValues = testDataG.map(row => row[3]); // LeftForeArm_Yrotation is at index 3
  const minOrig = Math.min(...originalValues);
  const maxOrig = Math.max(...originalValues);
  console.log(`   Min: ${minOrig.toFixed(3)}, Max: ${maxOrig.toFixed(3)}, Range: ${(maxOrig - minOrig).toFixed(3)}`);
  console.log(`   Sample values: [${originalValues.slice(0, 10).map(v => v.toFixed(3)).join(', ')}...]`);

  // Apply StandardScaler
  const scaler = new StandardScaler();
  const normalizedData = scaler.fitTransform(testDataG);
  
  console.log(`\n📏 StandardScaler normalized LeftForeArm_Yrotation data range:`);
  const normalizedValues = normalizedData.map(row => row[3]); // LeftForeArm_Yrotation is at index 3
  const minNorm = Math.min(...normalizedValues);
  const maxNorm = Math.max(...normalizedValues);
  console.log(`   Min: ${minNorm.toFixed(3)}, Max: ${maxNorm.toFixed(3)}, Range: ${(maxNorm - minNorm).toFixed(3)}`);
  console.log(`   Sample values: [${normalizedValues.slice(0, 10).map(v => v.toFixed(3)).join(', ')}...]`);
  
  // Apply inverse transform
  const denormalizedData = scaler.inverseTransform(normalizedData);
  const denormalizedValues = denormalizedData.map(row => row[3]);
  
  console.log(`\n🔄 Inverse transformed LeftForeArm_Yrotation data range:`);
  const minDenorm = Math.min(...denormalizedValues);
  const maxDenorm = Math.max(...denormalizedValues);
  console.log(`   Min: ${minDenorm.toFixed(3)}, Max: ${maxDenorm.toFixed(3)}, Range: ${(maxDenorm - minDenorm).toFixed(3)}`);
  console.log(`   Sample values: [${denormalizedValues.slice(0, 10).map(v => v.toFixed(3)).join(', ')}...]`);

  // Check if inverse transform worked correctly
  const maxDiff = Math.max(...originalValues.map((orig, i) => Math.abs(orig - denormalizedValues[i])));
  console.log(`\n✅ Maximum difference between original and inverse-transformed: ${maxDiff.toFixed(6)}`);
  
  console.log(`\n📋 Summary:`);
  console.log(`   🔹 Original data: Python-like joint angle values in degrees`);
  console.log(`   🔹 StandardScaler: Normalized around 0 with std=1 (typically -3 to +3 range)`);
  console.log(`   🔹 This should match Python sklearn.StandardScaler behavior!`);
  
} catch (error) {
  console.error('❌ Test failed:', error.message);
} 