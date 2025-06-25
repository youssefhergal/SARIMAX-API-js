import * as math from 'mathjs';
import fs from 'fs';
import bvh from 'bvh-parser';

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

// Pandas-like describe function for an array
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
  const q50 = sortedData[q50Index]; // median
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
  
  return {
    count, mean, std, min, 
    '25%': q25, '50%': q50, '75%': q75, 
    max
  };
}

// Main analysis for TEST data
console.log("🐼 Pandas-like Analysis for TEST Data");
console.log("=" .repeat(50));

try {
  console.log("\n🧪 ANALYZING BENDING TEST DATA");
  console.log("=" .repeat(35));
  
  // Load the TEST bending data
  const bvhTestB = extractDataFromBVH(
    './BVH/Bending/Test_Bending.bvh',
    'Hips_Xrotation',
    ['Spine_Yrotation', 'Spine_Zrotation', 'LeftArm_Xrotation', 'RightArm_Xrotation']
  );

  // Create DataFrame-like structure for TEST bending
  const dfTestB = {
    'Hips_Xrotation': bvhTestB.endog,
    'Spine_Yrotation': bvhTestB.exog.map(row => row[0]),
    'Spine_Zrotation': bvhTestB.exog.map(row => row[1]),
    'LeftArm_Xrotation': bvhTestB.exog.map(row => row[2]),
    'RightArm_Xrotation': bvhTestB.exog.map(row => row[3])
  };

  console.log(`\n📋 Test Bending Data Shape: [${dfTestB['Hips_Xrotation'].length}, ${Object.keys(dfTestB).length}]`);
  console.log(`📋 Columns: [${Object.keys(dfTestB).map(col => `'${col}'`).join(', ')}]`);

  // Equivalent to: dfTestB['Hips_Xrotation'].describe()
  const statsTestB = describe(dfTestB['Hips_Xrotation'], "dfTestB['Hips_Xrotation']");

  // Sample element access
  const indexSample = 500;
  if (indexSample < dfTestB['Hips_Xrotation'].length) {
    console.log(`\n🎯 dfTestB['Hips_Xrotation'][${indexSample}] = ${dfTestB['Hips_Xrotation'][indexSample].toFixed(6)}`);
  } else {
    console.log(`\n⚠️  Index ${indexSample} is out of bounds. Data length: ${dfTestB['Hips_Xrotation'].length}`);
  }

  // Show additional info for TEST bending
  console.log(`\n📈 Additional Info (Test Bending):`);
  console.log(`   📊 Data type: Float64 (equivalent to Python float64)`);
  console.log(`   📊 Total frames: ${dfTestB['Hips_Xrotation'].length}`);
  console.log(`   📊 Missing values: ${dfTestB['Hips_Xrotation'].filter(val => isNaN(val) || !isFinite(val)).length}`);
  console.log(`🔍 First 5 values: [${dfTestB['Hips_Xrotation'].slice(0, 5).map(v => v.toFixed(3)).join(', ')}]`);
  console.log(`🔍 Last 5 values:  [${dfTestB['Hips_Xrotation'].slice(-5).map(v => v.toFixed(3)).join(', ')}]`);

  console.log("\n" + "=".repeat(50));
  console.log("🧪 ANALYZING GLASSBLOWING TEST DATA");
  console.log("=" .repeat(35));
  
  // Load the TEST glassblowing data
  const bvhTestG = extractDataFromBVH(
    './BVH/Glassblowing/Test_Glassblowing.bvh',
    'LeftForeArm_Yrotation',
    ['Hips_Xrotation', 'Spine_Yrotation', 'Spine_Zrotation', 'RightForeArm_Yrotation']
  );

  // Create DataFrame-like structure for TEST glassblowing
  const dfTestG = {
    'Hips_Xrotation': bvhTestG.exog.map(row => row[0]),
    'Spine_Yrotation': bvhTestG.exog.map(row => row[1]),
    'Spine_Zrotation': bvhTestG.exog.map(row => row[2]),
    'LeftForeArm_Yrotation': bvhTestG.endog,  // Target variable
    'RightForeArm_Yrotation': bvhTestG.exog.map(row => row[3])
  };

  console.log(`\n📋 Test Glassblowing Data Shape: [${dfTestG['LeftForeArm_Yrotation'].length}, ${Object.keys(dfTestG).length}]`);
  console.log(`📋 Columns: [${Object.keys(dfTestG).map(col => `'${col}'`).join(', ')}]`);

  // Equivalent to: dfTestG['LeftForeArm_Yrotation'].describe()
  const statsTestG = describe(dfTestG['LeftForeArm_Yrotation'], "dfTestG['LeftForeArm_Yrotation']");

  // Sample element access for glassblowing
  const indexSampleG = 800;
  if (indexSampleG < dfTestG['LeftForeArm_Yrotation'].length) {
    console.log(`\n🎯 dfTestG['LeftForeArm_Yrotation'][${indexSampleG}] = ${dfTestG['LeftForeArm_Yrotation'][indexSampleG].toFixed(6)}`);
  } else {
    console.log(`\n⚠️  Index ${indexSampleG} is out of bounds. Data length: ${dfTestG['LeftForeArm_Yrotation'].length}`);
  }

  // Show additional info for TEST glassblowing
  console.log(`\n📈 Additional Info (Test Glassblowing):`);
  console.log(`   📊 Data type: Float64 (equivalent to Python float64)`);
  console.log(`   📊 Total frames: ${dfTestG['LeftForeArm_Yrotation'].length}`);
  console.log(`   📊 Missing values: ${dfTestG['LeftForeArm_Yrotation'].filter(val => isNaN(val) || !isFinite(val)).length}`);
  console.log(`🔍 First 5 values: [${dfTestG['LeftForeArm_Yrotation'].slice(0, 5).map(v => v.toFixed(3)).join(', ')}]`);
  console.log(`🔍 Last 5 values:  [${dfTestG['LeftForeArm_Yrotation'].slice(-5).map(v => v.toFixed(3)).join(', ')}]`);

  // Comparison summary
  console.log("\n" + "=".repeat(50));
  console.log("📊 COMPARISON SUMMARY");
  console.log("=" .repeat(20));
  console.log(`🔹 Bending Test (Hips_Xrotation):`);
  console.log(`   Frames: ${statsTestB.count}, Mean: ${statsTestB.mean.toFixed(3)}, Std: ${statsTestB.std.toFixed(3)}`);
  console.log(`   Range: [${statsTestB.min.toFixed(3)}, ${statsTestB.max.toFixed(3)}]`);
  
  console.log(`🔹 Glassblowing Test (LeftForeArm_Yrotation):`);
  console.log(`   Frames: ${statsTestG.count}, Mean: ${statsTestG.mean.toFixed(3)}, Std: ${statsTestG.std.toFixed(3)}`);
  console.log(`   Range: [${statsTestG.min.toFixed(3)}, ${statsTestG.max.toFixed(3)}]`);

  console.log(`\n💡 These are the TEST datasets used for forecasting plots!`);

} catch (error) {
  console.error('❌ Analysis failed:', error.message);
} 