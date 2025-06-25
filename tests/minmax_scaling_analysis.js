import * as math from 'mathjs';
import fs from 'fs';
import bvh from 'bvh-parser';

// MinMaxScaler class
class MinMaxScaler {
  constructor() {
    this.min = null;
    this.max = null;
    this.scale = null;
    this.minScale = null;
  }

  fit(data) {
    const matrix = math.matrix(data);
    this.min = math.min(matrix, 0)._data;
    this.max = math.max(matrix, 0)._data;
    this.scale = this.max.map((max, i) => max - this.min[i]);
    this.minScale = this.min;
    return this;
  }

  transform(data) {
    if (!this.scale) throw new Error("Scaler not fitted");
    return data.map(row => 
      row.map((val, i) => (val - this.minScale[i]) / this.scale[i])
    );
  }

  fitTransform(data) {
    return this.fit(data).transform(data);
  }

  inverseTransform(data) {
    if (!this.scale) throw new Error("Scaler not fitted");
    return data.map(row => 
      row.map((val, i) => val * this.scale[i] + this.minScale[i])
    );
  }

  getScalingInfo() {
    return {
      min: this.min,
      max: this.max,
      scale: this.scale
    };
  }
}

// Parse BVH header function
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

// Pandas-like describe function
function describe(data, columnName = 'Column') {
  const sortedData = [...data].sort((a, b) => a - b);
  const n = data.length;
  
  const count = n;
  const mean = data.reduce((sum, val) => sum + val, 0) / n;
  const variance = data.reduce((sum, val) => sum + Math.pow(val - mean, 2), 0) / (n - 1);
  const std = Math.sqrt(variance);
  const min = Math.min(...data);
  const max = Math.max(...data);
  
  // Percentiles
  const q25Index = Math.floor(0.25 * (n - 1));
  const q50Index = Math.floor(0.50 * (n - 1));
  const q75Index = Math.floor(0.75 * (n - 1));
  
  const q25 = sortedData[q25Index];
  const q50 = sortedData[q50Index];
  const q75 = sortedData[q75Index];
  
  console.log(`\n📊 ${columnName}.describe()`);
  console.log('='.repeat(40));
  console.log(`count    ${count.toFixed(6)}`);
  console.log(`mean     ${mean.toFixed(6)}`);
  console.log(`std      ${std.toFixed(6)}`);
  console.log(`min      ${min.toFixed(6)}`);
  console.log(`25%      ${q25.toFixed(6)}`);
  console.log(`50%      ${q50.toFixed(6)}`);
  console.log(`75%      ${q75.toFixed(6)}`);
  console.log(`max      ${max.toFixed(6)}`);
  
  return { count, mean, std, min, '25%': q25, '50%': q50, '75%': q75, max };
}

// Main analysis with MinMaxScaler
console.log("🔧 MinMaxScaler Transformation Analysis");
console.log("=" .repeat(50));

try {
  console.log("\n🧪 ANALYZING GLASSBLOWING TEST DATA WITH MINMAX SCALING");
  console.log("=" .repeat(55));
  
  // Load the TEST glassblowing data
  const bvhTestG = extractDataFromBVH(
    './BVH/Glassblowing/Test_Glassblowing.bvh',
    'LeftForeArm_Yrotation',
    ['Hips_Xrotation', 'Spine_Yrotation', 'Spine_Zrotation', 'RightForeArm_Yrotation']
  );

  // Create test data matrix
  const testDataG = bvhTestG.endog.map((endogValue, i) => [
    bvhTestG.exog[i][0], // Hips_Xrotation
    bvhTestG.exog[i][1], // Spine_Yrotation 
    bvhTestG.exog[i][2], // Spine_Zrotation
    endogValue,          // LeftForeArm_Yrotation
    bvhTestG.exog[i][3]  // RightForeArm_Yrotation
  ]);

  const columnNames = ['Hips_Xrotation', 'Spine_Yrotation', 'Spine_Zrotation', 'LeftForeArm_Yrotation', 'RightForeArm_Yrotation'];

  console.log(`\n📋 Original Test Data Shape: [${testDataG.length}, ${testDataG[0].length}]`);
  console.log(`📋 Columns: [${columnNames.map(col => `'${col}'`).join(', ')}]`);

  // 1. Show ORIGINAL data statistics (focus on LeftForeArm_Yrotation)
  console.log("\n" + "🔍 ORIGINAL DATA (Before MinMaxScaler)".padEnd(50, "="));
  const originalLeftForeArm = testDataG.map(row => row[3]); // LeftForeArm_Yrotation at index 3
  const originalStats = describe(originalLeftForeArm, "LeftForeArm_Yrotation [ORIGINAL]");

  // Show sample original values
  console.log(`\n🔍 Sample original values:`);
  console.log(`   First 10: [${originalLeftForeArm.slice(0, 10).map(v => v.toFixed(3)).join(', ')}]`);
  console.log(`   Elements [800-810]: [${originalLeftForeArm.slice(800, 810).map(v => v.toFixed(3)).join(', ')}]`);

  // 2. Apply MinMaxScaler
  console.log("\n" + "⚙️ APPLYING MINMAX SCALER".padEnd(50, "="));
  const scaler = new MinMaxScaler();
  const scaledData = scaler.fitTransform(testDataG);
  
  // Show scaler parameters
  const scalingInfo = scaler.getScalingInfo();
  console.log(`\n📐 MinMaxScaler Parameters:`);
  columnNames.forEach((name, i) => {
    console.log(`   ${name}:`);
    console.log(`      Min: ${scalingInfo.min[i].toFixed(6)}`);
    console.log(`      Max: ${scalingInfo.max[i].toFixed(6)}`);
    console.log(`      Scale: ${scalingInfo.scale[i].toFixed(6)}`);
  });

  // 3. Show SCALED data statistics
  console.log("\n" + "🔍 SCALED DATA (After MinMaxScaler)".padEnd(50, "="));
  const scaledLeftForeArm = scaledData.map(row => row[3]); // LeftForeArm_Yrotation at index 3
  const scaledStats = describe(scaledLeftForeArm, "LeftForeArm_Yrotation [SCALED 0-1]");

  // Show sample scaled values
  console.log(`\n🔍 Sample scaled values:`);
  console.log(`   First 10: [${scaledLeftForeArm.slice(0, 10).map(v => v.toFixed(6)).join(', ')}]`);
  console.log(`   Elements [800-810]: [${scaledLeftForeArm.slice(800, 810).map(v => v.toFixed(6)).join(', ')}]`);

  // 4. Verify inverse transformation
  console.log("\n" + "🔄 INVERSE TRANSFORMATION VERIFICATION".padEnd(50, "="));
  const inverseData = scaler.inverseTransform(scaledData);
  const inverseLeftForeArm = inverseData.map(row => row[3]);
  
  // Check if inverse transform worked correctly
  const maxDiff = Math.max(...originalLeftForeArm.map((orig, i) => Math.abs(orig - inverseLeftForeArm[i])));
  console.log(`✅ Maximum difference between original and inverse-transformed: ${maxDiff.toFixed(10)}`);
  
  if (maxDiff < 1e-10) {
    console.log(`✅ Inverse transformation is PERFECT!`);
  } else {
    console.log(`⚠️  Some precision lost in inverse transformation`);
  }

  // 5. Comparison summary
  console.log("\n" + "📊 BEFORE vs AFTER COMPARISON".padEnd(50, "="));
  console.log(`🔹 LeftForeArm_Yrotation ORIGINAL:`);
  console.log(`   Range: [${originalStats.min.toFixed(3)}, ${originalStats.max.toFixed(3)}] (${(originalStats.max - originalStats.min).toFixed(3)} degrees)`);
  console.log(`   Mean: ${originalStats.mean.toFixed(3)}, Std: ${originalStats.std.toFixed(3)}`);
  
  console.log(`🔹 LeftForeArm_Yrotation SCALED:`);
  console.log(`   Range: [${scaledStats.min.toFixed(6)}, ${scaledStats.max.toFixed(6)}] (normalized to 0-1)`);
  console.log(`   Mean: ${scaledStats.mean.toFixed(6)}, Std: ${scaledStats.std.toFixed(6)}`);

  console.log(`\n💡 MinMaxScaler transforms data to [0, 1] range!`);
  console.log(`💡 This is different from StandardScaler which centers around 0 with std=1`);

} catch (error) {
  console.error('❌ Analysis failed:', error.message);
} 