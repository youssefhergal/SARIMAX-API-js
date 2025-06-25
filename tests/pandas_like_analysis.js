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

// Main analysis
console.log("🐼 Pandas-like Analysis for Training Bending Data");
console.log("=" .repeat(50));

try {
  // Load the training bending data
  const bvhTrainB = extractDataFromBVH(
    './BVH/Bending/Test_Bending.bvh',
    'Hips_Xrotation',
    ['Spine_Yrotation', 'Spine_Zrotation', 'LeftArm_Xrotation', 'RightArm_Xrotation']
  );

  // Create DataFrame-like structure
  const dfTrainB = {
    'Hips_Xrotation': bvhTrainB.endog,
    'Spine_Yrotation': bvhTrainB.exog.map(row => row[0]),
    'Spine_Zrotation': bvhTrainB.exog.map(row => row[1]),
    'LeftArm_Xrotation': bvhTrainB.exog.map(row => row[2]),
    'RightArm_Xrotation': bvhTrainB.exog.map(row => row[3])
  };

  console.log(`\n📋 Training Bending Data Shape: [${dfTrainB['Hips_Xrotation'].length}, ${Object.keys(dfTrainB).length}]`);
  console.log(`📋 Columns: [${Object.keys(dfTrainB).map(col => `'${col}'`).join(', ')}]`);

  // Equivalent to: dfTrainB['Hips_Xrotation'].describe()
  const stats = describe(dfTrainB['Hips_Xrotation'], "dfTrainB['Hips_Xrotation']");

  // Equivalent to: dfTrainB['Hips_Xrotation'][870]
  const index870 = 870;
  if (index870 < dfTrainB['Hips_Xrotation'].length) {
    console.log(`\n🎯 dfTrainB['Hips_Xrotation'][${index870}] = ${dfTrainB['Hips_Xrotation'][index870].toFixed(6)}`);
  } else {
    console.log(`\n⚠️  Index ${index870} is out of bounds. Data length: ${dfTrainB['Hips_Xrotation'].length}`);
    console.log(`🎯 dfTrainB['Hips_Xrotation'][${dfTrainB['Hips_Xrotation'].length - 1}] = ${dfTrainB['Hips_Xrotation'][dfTrainB['Hips_Xrotation'].length - 1].toFixed(6)} (last element)`);
  }

  // Show some additional info
  console.log(`\n📈 Additional Info:`);
  console.log(`   📊 Data type: Float64 (equivalent to Python float64)`);
  console.log(`   📊 Total frames: ${dfTrainB['Hips_Xrotation'].length}`);
  console.log(`   📊 Missing values: ${dfTrainB['Hips_Xrotation'].filter(val => isNaN(val) || !isFinite(val)).length}`);
  
  // Show first and last few values
  console.log(`\n🔍 First 5 values: [${dfTrainB['Hips_Xrotation'].slice(0, 5).map(v => v.toFixed(3)).join(', ')}]`);
  console.log(`🔍 Last 5 values:  [${dfTrainB['Hips_Xrotation'].slice(-5).map(v => v.toFixed(3)).join(', ')}]`);

  // Show sample around index 870 if it exists
  if (index870 < dfTrainB['Hips_Xrotation'].length) {
    const start = Math.max(0, index870 - 2);
    const end = Math.min(dfTrainB['Hips_Xrotation'].length, index870 + 3);
    console.log(`\n🔍 Values around index ${index870}:`);
    for (let i = start; i < end; i++) {
      const marker = i === index870 ? ' ← TARGET' : '';
      console.log(`   [${i}]: ${dfTrainB['Hips_Xrotation'][i].toFixed(6)}${marker}`);
    }
  }

} catch (error) {
  console.error('❌ Analysis failed:', error.message);
} 