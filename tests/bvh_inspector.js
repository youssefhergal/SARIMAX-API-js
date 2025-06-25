// BVH File Inspector and Visualizer
// Analyzes BVH motion capture files and creates plots of the raw data

import fs from 'fs';
import bvh from 'bvh-parser';

// Function to manually parse BVH header for joint structure
function parseBVHHeader(bvhContent) {
  const lines = bvhContent.split('\n').map(line => line.trim());
  const joints = [];
  const channels = [];
  let currentJoint = null;
  let hierarchyLevel = 0;
  
  let inMotionSection = false;
  
  for (let i = 0; i < lines.length; i++) {
    const line = lines[i];
    
    if (line.startsWith('MOTION')) {
      inMotionSection = true;
      break;
    }
    
    if (line.startsWith('ROOT') || line.startsWith('JOINT')) {
      const parts = line.split(/\s+/);
      const jointName = parts[1];
      currentJoint = {
        name: jointName,
        type: line.startsWith('ROOT') ? 'ROOT' : 'JOINT',
        level: hierarchyLevel,
        channels: []
      };
      joints.push(currentJoint);
    } else if (line.startsWith('CHANNELS')) {
      if (currentJoint) {
        const parts = line.split(/\s+/);
        const channelCount = parseInt(parts[1]);
        const channelTypes = parts.slice(2, 2 + channelCount);
        
        currentJoint.channels = channelTypes;
        
        // Add to global channels list
        channelTypes.forEach(channelType => {
          channels.push(`${currentJoint.name}_${channelType}`);
        });
      }
    } else if (line === '{') {
      hierarchyLevel++;
    } else if (line === '}') {
      hierarchyLevel--;
    }
  }
  
  return { joints, channels };
}

// Function to inspect BVH file structure and frames
function inspectBVH(filePath) {
  console.log(`\n🔍 INSPECTING BVH FILE: ${filePath}`);
  console.log('='.repeat(80));

  try {
    // Read and parse BVH file
    const bvhContent = fs.readFileSync(filePath, 'utf-8');
    const result = bvh(bvhContent);
    
    // Manual header parsing
    const headerInfo = parseBVHHeader(bvhContent);
    
    console.log('\n📁 BVH FILE STRUCTURE:');
    console.log(`Keys available: ${Object.keys(result).join(', ')}`);
    
    // Inspect frames
    if (result.frames) {
      console.log(`\n📊 FRAME INFORMATION:`);
      console.log(`Total frames: ${result.frames.length}`);
      console.log(`Frame time: ${result.frameTime || 'N/A'} seconds`);
      console.log(`Total duration: ${(result.frames.length * (result.frameTime || 0)).toFixed(2)} seconds`);
      console.log(`Channels per frame: ${result.frames[0]?.length || 'N/A'}`);
      
      // Show first few frame values
      console.log(`\n📋 FIRST 5 FRAMES DATA:`);
      for (let i = 0; i < Math.min(5, result.frames.length); i++) {
        const frame = result.frames[i];
        console.log(`Frame ${i}: [${frame.slice(0, 10).map(v => v.toFixed(3)).join(', ')}${frame.length > 10 ? ', ...' : ''}] (${frame.length} values)`);
      }
      
      // Statistics about frame data
      console.log(`\n📈 FRAME DATA STATISTICS:`);
      if (result.frames.length > 0) {
        const firstFrame = result.frames[0];
        const lastFrame = result.frames[result.frames.length - 1];
        const midFrame = result.frames[Math.floor(result.frames.length / 2)];
        
        console.log(`First frame range: ${Math.min(...firstFrame).toFixed(3)} to ${Math.max(...firstFrame).toFixed(3)}`);
        console.log(`Middle frame range: ${Math.min(...midFrame).toFixed(3)} to ${Math.max(...midFrame).toFixed(3)}`);
        console.log(`Last frame range: ${Math.min(...lastFrame).toFixed(3)} to ${Math.max(...lastFrame).toFixed(3)}`);
      }
    }
    
    // Display parsed joint hierarchy
    if (headerInfo.joints.length > 0) {
      console.log(`\n🦴 JOINT HIERARCHY (${headerInfo.joints.length} joints):`);
      headerInfo.joints.forEach(joint => {
        const indent = '  '.repeat(joint.level);
        console.log(`${indent}${joint.type === 'ROOT' ? '🏠' : '🔗'} ${joint.name}`);
        if (joint.channels.length > 0) {
          console.log(`${indent}   📍 Channels: ${joint.channels.join(', ')}`);
        }
      });
      
      console.log(`\n🔧 CHANNEL INFORMATION (${headerInfo.channels.length} channels):`);
      console.log(`Total channels found: ${headerInfo.channels.length}`);
      
      if (headerInfo.channels.length > 0) {
        console.log(`\nFirst 30 channels:`);
        headerInfo.channels.slice(0, 30).forEach((channel, i) => {
          console.log(`  ${i.toString().padStart(3)}: ${channel}`);
        });
        
        if (headerInfo.channels.length > 30) {
          console.log(`  ... and ${headerInfo.channels.length - 30} more channels`);
        }
      }
      
      // Show channel breakdown by type
      const channelTypes = {};
      headerInfo.channels.forEach(channel => {
        const type = channel.split('_')[1];
        channelTypes[type] = (channelTypes[type] || 0) + 1;
      });
      
      console.log(`\n📈 CHANNEL TYPE BREAKDOWN:`);
      Object.entries(channelTypes).forEach(([type, count]) => {
        console.log(`  ${type}: ${count} channels`);
      });
      
      return { ...result, parsedChannels: headerInfo.channels, parsedJoints: headerInfo.joints };
    } else {
      console.log(`\n⚠️ Could not parse joint hierarchy from BVH header`);
      return result;
    }
    
  } catch (error) {
    console.error(`❌ Error inspecting BVH file: ${error.message}`);
    return null;
  }
}

// Function to print joint hierarchy recursively
function printJointHierarchy(joint, depth = 0, prefix = '') {
  const indent = '  '.repeat(depth);
  const connector = depth === 0 ? '' : '├─ ';
  
  console.log(`${indent}${connector}${joint.name}`);
  
  if (joint.channels && joint.channels.length > 0) {
    console.log(`${indent}   📍 Channels: ${joint.channels.join(', ')}`);
  }
  
  if (joint.offset) {
    console.log(`${indent}   📐 Offset: [${joint.offset.map(v => v.toFixed(3)).join(', ')}]`);
  }
  
  if (joint.joints && joint.joints.length > 0) {
    joint.joints.forEach((childJoint, index) => {
      const isLast = index === joint.joints.length - 1;
      printJointHierarchy(childJoint, depth + 1, isLast ? '└─ ' : '├─ ');
    });
  }
}

// Function to extract all channel names
function extractChannelInfo(joint, channels = []) {
  if (joint.channels) {
    joint.channels.forEach(channel => {
      channels.push(`${joint.name}_${channel}`);
    });
  }
  
  if (joint.joints) {
    joint.joints.forEach(childJoint => {
      extractChannelInfo(childJoint, channels);
    });
  }
  
  return channels;
}

// Function to create CSV-like data preview
function createDataPreview(bvhResult, channelNames, numRows = 10) {
  if (!bvhResult || !bvhResult.frames || !channelNames) {
    console.log('❌ No frame data or channel names available for preview');
    return;
  }
  
  const frames = bvhResult.frames;
  const previewRows = Math.min(numRows, frames.length);
  
  console.log('\n📊 DATA PREVIEW (CSV-like format):');
  console.log('='.repeat(120));
  
  // Header row
  const headerColumns = channelNames.slice(0, 10);
  const headerRow = ['Frame', ...headerColumns, '...'].join('\t');
  console.log(headerRow);
  console.log('-'.repeat(120));
  
  // Data rows
  for (let i = 0; i < previewRows; i++) {
    const frame = frames[i];
    const dataColumns = frame.slice(0, 10).map(val => val.toFixed(6));
    const dataRow = [i.toString(), ...dataColumns, '...'].join('\t');
    console.log(dataRow);
  }
  
  console.log('\n📋 SAMPLE OF ACTUAL EXPECTED FORMAT:');
  console.log('Index\t' + channelNames.slice(0, 6).join('\t') + '\t...');
  for (let i = 0; i < Math.min(5, frames.length); i++) {
    const frame = frames[i];
    const values = frame.slice(0, 6).map(val => val.toFixed(6));
    console.log(`${i}\t${values.join('\t')}\t...`);
  }
}

// Function to create HTML plot of BVH motion data
function createBVHPlot(bvhResult, channelIndices, channelNames, filename) {
  if (!bvhResult || !bvhResult.frames) {
    console.log('❌ No frame data to plot');
    return;
  }
  
  const frames = bvhResult.frames;
  const timeFrames = Array.from({length: frames.length}, (_, i) => i * (bvhResult.frameTime || 1));
  
  // Extract data for selected channels
  const plotData = channelIndices.map((channelIndex, i) => {
    const values = frames.map(frame => frame[channelIndex] || 0);
    return {
      name: channelNames[i],
      values: values,
      index: channelIndex
    };
  });
  
  // Calculate statistics
  const stats = plotData.map(channel => {
    const values = channel.values;
    const min = Math.min(...values);
    const max = Math.max(...values);
    const mean = values.reduce((sum, v) => sum + v, 0) / values.length;
    const std = Math.sqrt(values.reduce((sum, v) => sum + Math.pow(v - mean, 2), 0) / values.length);
    
    return { ...channel, min, max, mean, std };
  });
  
  const htmlContent = `
<!DOCTYPE html>
<html>
<head>
    <title>BVH Motion Data Visualization - Parsed Channels</title>
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; }
        .plot-container { width: 100%; height: 600px; margin-bottom: 30px; }
        .stats-container {
            background: #f5f5f5;
            padding: 20px;
            border-radius: 8px;
            margin-bottom: 20px;
        }
        .stats-table {
            width: 100%;
            border-collapse: collapse;
            margin-top: 15px;
        }
        .stats-table th, .stats-table td {
            padding: 8px 12px;
            border: 1px solid #ddd;
            text-align: left;
        }
        .stats-table th {
            background-color: #4CAF50;
            color: white;
        }
        .stats-table tr:nth-child(even) {
            background-color: #f9f9f9;
        }
        .channel-name {
            font-weight: bold;
            color: #333;
        }
        .number {
            font-family: monospace;
            text-align: right;
        }
        .position { background-color: #e3f2fd; }
        .rotation { background-color: #f3e5f5; }
        h1 { color: #333; text-align: center; }
        h2 { color: #666; }
    </style>
</head>
<body>
    <h1>🎭 BVH Motion Data Visualization</h1>
    <h2>📊 Parsed Motion Capture Data with Named Channels</h2>
    
    <div class="stats-container">
        <h3>📈 Motion Statistics</h3>
        <p><strong>Total Frames:</strong> ${frames.length}</p>
        <p><strong>Frame Time:</strong> ${bvhResult.frameTime || 'N/A'} seconds</p>
        <p><strong>Total Duration:</strong> ${(frames.length * (bvhResult.frameTime || 0)).toFixed(2)} seconds</p>
        <p><strong>Channels per Frame:</strong> ${frames[0]?.length || 'N/A'}</p>
        
        <table class="stats-table">
            <thead>
                <tr>
                    <th>Channel Name</th>
                    <th>Index</th>
                    <th>Type</th>
                    <th>Min</th>
                    <th>Max</th>
                    <th>Mean</th>
                    <th>Std Dev</th>
                    <th>Range</th>
                </tr>
            </thead>
            <tbody>
                ${stats.map(stat => {
                  const channelType = stat.name.includes('position') ? 'position' : 'rotation';
                  return `
                    <tr class="${channelType}">
                        <td class="channel-name">${stat.name}</td>
                        <td class="number">${stat.index}</td>
                        <td>${channelType}</td>
                        <td class="number">${stat.min.toFixed(3)}</td>
                        <td class="number">${stat.max.toFixed(3)}</td>
                        <td class="number">${stat.mean.toFixed(3)}</td>
                        <td class="number">${stat.std.toFixed(3)}</td>
                        <td class="number">${(stat.max - stat.min).toFixed(3)}</td>
                    </tr>`;
                }).join('')}
            </tbody>
        </table>
        
        <div style="margin-top: 15px;">
            <p><strong>Legend:</strong></p>
            <span style="background: #e3f2fd; padding: 3px 8px; margin-right: 10px;">Position Channels</span>
            <span style="background: #f3e5f5; padding: 3px 8px;">Rotation Channels</span>
        </div>
    </div>
    
    <div id="plot" class="plot-container"></div>
    
    <script>
        const traces = [
            ${plotData.map((channel, i) => `
            {
                x: [${timeFrames.join(', ')}],
                y: [${channel.values.join(', ')}],
                type: 'scatter',
                mode: 'lines',
                name: '${channel.name}',
                line: { width: 2 }
            }`).join(',\n            ')}
        ];
        
        const layout = {
            title: 'BVH Motion Capture Data - Named Channels',
            xaxis: { 
                title: 'Time (seconds)',
                showgrid: true,
                gridcolor: '#eee'
            },
            yaxis: { 
                title: 'Channel Values',
                showgrid: true,
                gridcolor: '#eee'
            },
            hovermode: 'x unified',
            legend: {
                x: 1.02,
                y: 1
            },
            margin: { t: 50, l: 60, r: 150, b: 60 }
        };
        
        Plotly.newPlot('plot', traces, layout, {responsive: true});
    </script>
</body>
</html>`;

  fs.writeFileSync(filename, htmlContent);
  console.log(`\n📊 Enhanced BVH plot saved: ${filename}`);
}

// Function to extract and analyze Hips X rotation specifically
function analyzeHipsXRotation(filePath) {
  console.log(`\n🔍 ANALYZING HIPS X ROTATION: ${filePath}`);
  console.log('='.repeat(80));

  try {
    // Read and parse BVH file
    const bvhContent = fs.readFileSync(filePath, 'utf-8');
    const result = bvh(bvhContent);
    
    // Manual header parsing to get proper channel names
    const headerInfo = parseBVHHeader(bvhContent);
    
    console.log('\n📊 BVH FILE INFO:');
    console.log(`Total frames: ${result.frames?.length || 'N/A'}`);
    console.log(`Frame time: ${result.frameTime || 'N/A'} seconds`);
    console.log(`Duration: ${(result.frames.length * (result.frameTime || 0)).toFixed(2)} seconds`);
    console.log(`Total channels: ${headerInfo.channels.length}`);
    
    // Find Hips_Xrotation channel
    const hipsXChannelIndex = headerInfo.channels.findIndex(ch => 
      ch === 'Hips_Xrotation' || 
      ch.toLowerCase().includes('hips') && ch.toLowerCase().includes('xrotation')
    );
    
    if (hipsXChannelIndex === -1) {
      console.log('\n❌ Hips_Xrotation channel not found!');
      console.log('\n📋 Available Hips channels:');
      const hipsChannels = headerInfo.channels.filter(ch => ch.toLowerCase().includes('hips'));
      hipsChannels.forEach((ch, i) => {
        console.log(`  ${i}: ${ch}`);
      });
      
      console.log('\n📋 All available channels (first 30):');
      headerInfo.channels.slice(0, 30).forEach((ch, i) => {
        console.log(`  ${i}: ${ch}`);
      });
      return null;
    }
    
    const channelName = headerInfo.channels[hipsXChannelIndex];
    console.log(`\n✅ Found: ${channelName} at index ${hipsXChannelIndex}`);
    
    // Extract Hips X rotation data
    if (result.frames && result.frames.length > 0) {
      const hipsXData = result.frames.map(frame => frame[hipsXChannelIndex]);
      
      // Calculate statistics
      const min = Math.min(...hipsXData);
      const max = Math.max(...hipsXData);
      const mean = hipsXData.reduce((sum, val) => sum + val, 0) / hipsXData.length;
      const std = Math.sqrt(hipsXData.reduce((sum, val) => sum + Math.pow(val - mean, 2), 0) / hipsXData.length);
      
      console.log(`\n📈 HIPS X ROTATION STATISTICS:`);
      console.log(`Channel: ${channelName}`);
      console.log(`Index: ${hipsXChannelIndex}`);
      console.log(`Frames: ${hipsXData.length}`);
      console.log(`Min angle: ${min.toFixed(6)}°`);
      console.log(`Max angle: ${max.toFixed(6)}°`);
      console.log(`Mean: ${mean.toFixed(6)}°`);
      console.log(`Std Dev: ${std.toFixed(6)}°`);
      console.log(`Range: ${(max - min).toFixed(6)}°`);
      
             // Show detailed frame data
       console.log(`\n📊 DETAILED HIPS X ROTATION DATA:`);
       console.log('Frame | Angle (degrees)');
       console.log('------|----------------');
       
       // Show first 30 frames
       for (let i = 0; i < Math.min(30, hipsXData.length); i++) {
         console.log(`${i.toString().padStart(5)} | ${hipsXData[i].toFixed(8)}`);
       }
       
       if (hipsXData.length > 30) {
         console.log(`...   | ... (${hipsXData.length - 60} frames omitted)`);
         
         // Show last 30 frames
         console.log(`\n📊 LAST 30 FRAMES:`);
         console.log('Frame | Angle (degrees)');
         console.log('------|----------------');
         for (let i = Math.max(0, hipsXData.length - 30); i < hipsXData.length; i++) {
           console.log(`${i.toString().padStart(5)} | ${hipsXData[i].toFixed(8)}`);
         }
       }
      
      const frameTime = result.frameTime || (1/30); // Default to 30fps if not specified
      
      return {
        channelName,
        channelIndex: hipsXChannelIndex,
        data: hipsXData,
        frameTime,
        stats: { min, max, mean, std }
      };
    }
    
  } catch (error) {
    console.error(`❌ Error analyzing BVH file: ${error.message}`);
    return null;
  }
}

// Function to create focused Hips X rotation plot
function createHipsXPlot(hipsData, filename, title) {
  if (!hipsData) {
    console.log('❌ No Hips X rotation data to plot');
    return;
  }
  
  const frameNumbers = Array.from({length: hipsData.data.length}, (_, i) => i);
  
  const htmlContent = `
<!DOCTYPE html>
<html>
<head>
    <title>${title} - Hips X Rotation Analysis</title>
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; }
        .plot-container { width: 100%; height: 700px; margin-bottom: 30px; }
        .stats-container {
            background: #f0f8ff;
            padding: 20px;
            border-radius: 8px;
            margin-bottom: 20px;
            border: 2px solid #4CAF50;
        }
        .stats-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 15px;
            margin-top: 15px;
        }
        .stat-box {
            background: white;
            padding: 15px;
            border-radius: 5px;
            border: 1px solid #ddd;
            text-align: center;
        }
        .stat-value {
            font-size: 24px;
            font-weight: bold;
            color: #2196F3;
        }
        .stat-label {
            font-size: 14px;
            color: #666;
            margin-top: 5px;
        }
        h1 { color: #333; text-align: center; }
        h2 { color: #666; text-align: center; }
        h3 { color: #444; }
    </style>
</head>
<body>
    <h1>🎯 Hips X Rotation Analysis</h1>
    <h2>${title}</h2>
    
    <div class="stats-container">
        <h3>📊 Motion Statistics - ${hipsData.channelName}</h3>
        <div class="stats-grid">
            <div class="stat-box">
                <div class="stat-value">${hipsData.data.length}</div>
                <div class="stat-label">Total Frames</div>
            </div>
            <div class="stat-box">
                <div class="stat-value">${(hipsData.data.length * hipsData.frameTime).toFixed(2)}s</div>
                <div class="stat-label">Duration</div>
            </div>
            <div class="stat-box">
                <div class="stat-value">${hipsData.stats.min.toFixed(2)}°</div>
                <div class="stat-label">Minimum Angle</div>
            </div>
            <div class="stat-box">
                <div class="stat-value">${hipsData.stats.max.toFixed(2)}°</div>
                <div class="stat-label">Maximum Angle</div>
            </div>
            <div class="stat-box">
                <div class="stat-value">${hipsData.stats.mean.toFixed(2)}°</div>
                <div class="stat-label">Mean Angle</div>
            </div>
            <div class="stat-box">
                <div class="stat-value">${hipsData.stats.std.toFixed(2)}°</div>
                <div class="stat-label">Standard Deviation</div>
            </div>
            <div class="stat-box">
                <div class="stat-value">${(hipsData.stats.max - hipsData.stats.min).toFixed(2)}°</div>
                <div class="stat-label">Range</div>
            </div>
            <div class="stat-box">
                <div class="stat-value">${hipsData.channelIndex}</div>
                <div class="stat-label">Channel Index</div>
            </div>
        </div>
    </div>
    
    <div id="plot" class="plot-container"></div>
    
    <script>
                 const trace = {
             x: [${frameNumbers.join(', ')}],
             y: [${hipsData.data.join(', ')}],
             type: 'scatter',
             mode: 'lines+markers',
             name: '${hipsData.channelName}',
             line: { 
                 color: '#2196F3', 
                 width: 2 
             },
             marker: {
                 size: 3,
                 color: '#1976D2'
             }
         };
         
         // Add horizontal lines for mean, min, max
         const meanLine = {
             x: [${frameNumbers[0]}, ${frameNumbers[frameNumbers.length - 1]}],
             y: [${hipsData.stats.mean}, ${hipsData.stats.mean}],
             type: 'scatter',
             mode: 'lines',
             name: 'Mean (${hipsData.stats.mean.toFixed(2)}°)',
             line: { 
                 color: 'red', 
                 width: 2, 
                 dash: 'dash' 
             }
         };
         
         const minLine = {
             x: [${frameNumbers[0]}, ${frameNumbers[frameNumbers.length - 1]}],
             y: [${hipsData.stats.min}, ${hipsData.stats.min}],
             type: 'scatter',
             mode: 'lines',
             name: 'Min (${hipsData.stats.min.toFixed(2)}°)',
             line: { 
                 color: 'green', 
                 width: 1, 
                 dash: 'dot' 
             }
         };
         
         const maxLine = {
             x: [${frameNumbers[0]}, ${frameNumbers[frameNumbers.length - 1]}],
             y: [${hipsData.stats.max}, ${hipsData.stats.max}],
             type: 'scatter',
             mode: 'lines',
             name: 'Max (${hipsData.stats.max.toFixed(2)}°)',
             line: { 
                 color: 'orange', 
                 width: 1, 
                 dash: 'dot' 
             }
         };
        
        const data = [trace, meanLine, minLine, maxLine];
        
        const layout = {
            title: {
                text: '${hipsData.channelName} - Complete Motion Timeline',
                font: { size: 18 }
            },
            xaxis: { 
                title: 'Frame Number',
                showgrid: true,
                gridcolor: '#eee'
            },
            yaxis: { 
                title: 'Rotation Angle (degrees)',
                showgrid: true,
                gridcolor: '#eee'
            },
            hovermode: 'x unified',
            legend: {
                x: 1.02,
                y: 1
            },
            margin: { t: 80, l: 80, r: 150, b: 80 }
        };
        
        Plotly.newPlot('plot', data, layout, {responsive: true});
    </script>
</body>
</html>`;

  fs.writeFileSync(filename, htmlContent);
  console.log(`\n📊 Hips X rotation plot saved: ${filename}`);
}

// Main execution - Focus on Hips X Rotation
console.log('🎯 HIPS X ROTATION ANALYZER');
console.log('============================\n');

// List of BVH files to analyze
const bvhFiles = [
  './BVH/Bending/Train_Bending.bvh',
  './BVH/Bending/Test_Bending.bvh',
  './BVH/Glassblowing/Train_Glassblowing.bvh',
  './BVH/Glassblowing/Test_Glassblowing.bvh'
];

// Analyze Hips X rotation for each file
bvhFiles.forEach(filePath => {
  if (fs.existsSync(filePath)) {
    const hipsData = analyzeHipsXRotation(filePath);
    
    if (hipsData) {
      // Create focused plot
      const baseName = filePath.split('/').pop().replace('.bvh', '');
      const plotFilename = `hips_x_rotation_${baseName.toLowerCase()}.html`;
      const title = baseName.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase());
      
      createHipsXPlot(hipsData, plotFilename, title);
      
      console.log(`\n✅ Analysis complete for ${baseName}`);
    }
  } else {
    console.log(`⚠️  File not found: ${filePath}`);
  }
});

console.log('\n🎉 HIPS X ROTATION ANALYSIS COMPLETE!');
console.log('\n📂 Generated Files:');
console.log('   🎯 hips_x_rotation_train_bending.html - Training bending Hips X rotation');
console.log('   🎯 hips_x_rotation_test_bending.html - Test bending Hips X rotation');  
console.log('   🎯 hips_x_rotation_train_glassblowing.html - Training glassblowing Hips X rotation');
console.log('   🎯 hips_x_rotation_test_glassblowing.html - Test glassblowing Hips X rotation');
console.log('\n💡 Open the HTML files to view detailed Hips X rotation motion data!'); 