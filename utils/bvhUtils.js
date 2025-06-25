import fs from 'fs';
import bvh from 'bvh-parser';

// Parse BVH header to extract channel names
export function parseBVHHeader(bvhContent) {
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

// BVH data extraction function with proper header parsing
export function extractDataFromBVH(path, targetJoint, exogJoints) {
  try {
    console.log(`📂 Loading BVH file: ${path}`);
    const bvhContent = fs.readFileSync(path, 'utf-8');
    
    // Parse BVH header to get proper channel names
    const headerInfo = parseBVHHeader(bvhContent);
    console.log(`✅ Parsed header: ${headerInfo.channels.length} channels found`);
    
    // Parse BVH data
    const result = bvh(bvhContent);
    const frames = result?.frames;
    
    if (!frames || frames.length === 0) {
      throw new Error('No frames found in BVH file');
    }
    
    console.log(`📊 Found ${frames.length} frames`);
    console.log(`🎯 Looking for target joint: ${targetJoint}`);
    console.log(`🔗 Looking for exogenous joints: ${exogJoints.join(', ')}`);
    
    // Find target joint index
    const targetIndex = headerInfo.channels.findIndex(ch => ch === targetJoint);
    if (targetIndex === -1) {
      console.log('❌ Available channels:', headerInfo.channels.slice(0, 20));
      throw new Error(`Target joint "${targetJoint}" not found`);
    }
    
    // Find exogenous joint indices
    const exogIndices = exogJoints.map(joint => {
      const index = headerInfo.channels.findIndex(ch => ch === joint);
      if (index === -1) {
        console.warn(`⚠️ Exogenous joint "${joint}" not found, using zeros`);
        return -1;
      }
      return index;
    });
    
    console.log(`✅ Target "${targetJoint}" found at index: ${targetIndex}`);
    console.log(`✅ Exogenous joints indices:`, exogIndices);
    
    // Extract data
    const endog = frames.map(frame => frame[targetIndex]);
    const exog = frames.map(frame => 
      exogIndices.map(idx => idx === -1 ? 0 : frame[idx])
    );
    
    // Verify data quality
    const validEndog = endog.filter(val => !isNaN(val) && isFinite(val));
    const validExog = exog.filter(row => row.every(val => !isNaN(val) && isFinite(val)));
    
    if (validEndog.length !== frames.length || validExog.length !== frames.length) {
      console.warn(`⚠️ Some invalid data found. Valid endog: ${validEndog.length}/${frames.length}`);
    }
    
    console.log(`✅ Successfully extracted:
      📈 Endogenous values: ${endog.length} frames
      📊 Exogenous values: ${exog.length} frames x ${exog[0].length} variables
      📋 Sample endogenous values: [${endog.slice(0, 5).map(v => v.toFixed(3)).join(', ')}...]
      📋 Sample exogenous values: [${exog[0].map(v => v.toFixed(3)).join(', ')}]`);
    
    return { endog, exog };
    
  } catch (error) {
    console.error('❌ Error loading BVH file:', error.message);
    console.log('⚠️ THIS SHOULD NOT HAPPEN - BVH parsing failed!');
    throw error;
  }
}

// Get all rotation columns from BVH data
export function getRotationColumns(bvhData) {
  const result = bvh(bvhData);
  const jointNames = result?.motionChannels?.map(ch => ch.name) || [];
  return jointNames.filter(name => name.includes('rotation'));
}

// Extract Euler angles from BVH
export function extractEulerAngles(bvhPath) {
  const bvhContent = fs.readFileSync(bvhPath, 'utf-8');
  const result = bvh(bvhContent);
  const frames = result?.frames;
  const jointNames = result?.motionChannels?.map(ch => ch.name);
  
  const rotationCols = jointNames.filter(name => name.includes('rotation'));
  const jointMap = {};
  jointNames.forEach((name, i) => { jointMap[name] = i; });

  const eulerAngles = {};
  rotationCols.forEach(col => {
    eulerAngles[col] = frames.map(frame => frame[jointMap[col]]);
  });

  return eulerAngles;
} 