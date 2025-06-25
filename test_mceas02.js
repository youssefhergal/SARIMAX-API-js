// Test MCEAS02 BVH files with RightArm_Yrotation target
// Training: MCEAS02G01R03.bvh | Testing: MCEAS02G01R02.bvh

// Import all modules
import { StandardScaler } from './classes/StandardScaler.js';
import { MinMaxScaler } from './classes/MinMaxScaler.js';
import { SARIMAX } from './classes/SARIMAX.js';
import { extractDataFromBVH, extractEulerAngles } from './utils/bvhUtils.js';
import { MSE, MAE, UTheil, createModelSummary } from './utils/metrics.js';
import { staticForecasting } from './forecasting/staticForecasting.js';
import { dynamicForecasting } from './forecasting/dynamicForecasting.js';
import { createPlot, createConsolePlot } from './visualization/plotUtils.js';
import { displayModelTable, createDataFrame, compareModels } from './utils/modelVisualization.js';

// Main execution - MCEAS02 Analysis with separate train/test files
console.log("🚀 MCEAS02 Analysis - RightArm_Yrotation Target (SEPARATE TRAIN/TEST FILES)");
console.log("🏋️ Training: MCEAS02G01R03.bvh | 🧪 Testing: MCEAS02G01R02.bvh");
console.log("==============================================================================\n");

// Define all BVH angles to use
const ALL_BVH_ANGLES = [
  'LeftArm_Zrotation', 'LeftArm_Xrotation', 'LeftArm_Yrotation',
  'LeftForeArm_Zrotation', 'LeftForeArm_Xrotation', 'LeftForeArm_Yrotation',
  'RightArm_Zrotation', 'RightArm_Xrotation', 'RightArm_Yrotation',
  'RightForeArm_Zrotation', 'RightForeArm_Xrotation', 'RightForeArm_Yrotation',
  'Spine_Zrotation', 'Spine_Xrotation', 'Spine_Yrotation',
  'Spine1_Zrotation', 'Spine1_Xrotation', 'Spine1_Yrotation',
  'Spine2_Zrotation', 'Spine2_Xrotation', 'Spine2_Yrotation',
  'Spine3_Zrotation', 'Spine3_Xrotation', 'Spine3_Yrotation',
  'Hips_Zrotation', 'Hips_Xrotation', 'Hips_Yrotation'
];

console.log(`📋 Using ${ALL_BVH_ANGLES.length} BVH angles for analysis`);

// 1. Load BVH files - TRAINING and TESTING separately
console.log("1. Loading MCEAS02 BVH files...");

// Target variable setup
const targetAngle = 'RightArm_Yrotation';
const exogAngles = ALL_BVH_ANGLES.filter(angle => angle !== targetAngle);

console.log(`🎯 Target Variable: ${targetAngle}`);
console.log(`📊 Exogenous Variables: ${exogAngles.length} joint angles`);

// Load TRAINING data from MCEAS02G01R03.bvh
console.log("🔄 Loading TRAINING data: MCEAS02G01R03.bvh...");
const bvhTrainData = extractDataFromBVH(
  './BVH/MCEAS02G01R03.bvh',
  targetAngle,
  exogAngles
);

// Load TESTING data from MCEAS02G01R02.bvh  
console.log("🔄 Loading TESTING data: MCEAS02G01R02.bvh...");
const bvhTestData = extractDataFromBVH(
  './BVH/MCEAS02G01R02.bvh',
  targetAngle,
  exogAngles
);

// Check if both files loaded successfully
if (!bvhTrainData || !bvhTrainData.endog || bvhTrainData.endog.length === 0) {
    console.log("❌ Failed to load TRAINING BVH data");
    process.exit(1);
}

if (!bvhTestData || !bvhTestData.endog || bvhTestData.endog.length === 0) {
    console.log("❌ Failed to load TESTING BVH data");
    process.exit(1);
}

console.log(`✅ Loaded TRAINING data: ${bvhTrainData.endog.length} frames`);
console.log(`✅ Loaded TESTING data: ${bvhTestData.endog.length} frames`);
console.log(`🎯 TOTAL VISUALIZATION FRAMES: ${bvhTestData.endog.length} (ENTIRE TEST FILE!)`);

// 2. Prepare data for training/testing - NO SPLITTING needed!
console.log("2. Preparing data for analysis (using separate files)...");

// Create training data from MCEAS02G01R03.bvh
const trainDataRaw = bvhTrainData.endog.map((endogValue, i) => [
  endogValue,
  ...bvhTrainData.exog[i]
]);

// Create testing data from MCEAS02G01R02.bvh  
const testDataRaw = bvhTestData.endog.map((endogValue, i) => [
  endogValue,
  ...bvhTestData.exog[i]
]);

console.log(`📊 Training data: ${trainDataRaw.length} frames from MCEAS02G01R03.bvh`);
console.log(`📊 Testing data: ${testDataRaw.length} frames from MCEAS02G01R02.bvh`);
console.log(`🎯 ENHANCED VISUALIZATION: Using ALL ${testDataRaw.length} frames from test file!`);

// 3. Normalize data
console.log("3. Normalizing data with StandardScaler...");
const scalerTrain = new StandardScaler();
const scalerTest = new StandardScaler();

const trainData = scalerTrain.fitTransform(trainDataRaw);
const testData = scalerTest.fitTransform(testDataRaw);

console.log(`🔧 Normalized data - Training: ${trainData.length} frames, Testing: ${testData.length} frames`);

// 4. Train SARIMAX model
console.log("4. Training SARIMAX model...");

const order = 2;
const bvhAngles = [targetAngle, ...exogAngles];
const indEnd = 0; // Target is at index 0
const indExo = Array.from({length: exogAngles.length}, (_, i) => i + 1); // All other indices

const endog = trainData.map(row => row[indEnd]);
const exog = trainData.map(row => indExo.map(idx => row[idx]));

console.log(`🎯 Training model on MCEAS02G01R03 - Target: ${targetAngle}`);
console.log(`📊 Training data: ${endog.length} frames, ${exog[0].length} exogenous variables`);

const model = new SARIMAX(endog, exog, order);
model.fit();

const summary = createModelSummary(model, bvhAngles, targetAngle, indExo);

// Display detailed model table
console.log('\n📊 DETAILED MODEL ANALYSIS - MCEAS02 MOTION');
const modelTable = displayModelTable(model, bvhAngles, targetAngle, indExo);
const dfModel = createDataFrame(modelTable);

// Show DataFrame-like operations
dfModel.head();
dfModel.describe();
dfModel.info();

// 5. Static Forecasting - TESTING on MCEAS02G01R02.bvh
console.log("\n5. Performing Static Forecasting on MCEAS02G01R02.bvh (ALL FRAMES)...");

const { predStatic, origValues: origStatic } = staticForecasting(model, testData, indEnd, indExo, scalerTest, indEnd);

// Evaluate static forecasting
console.log(`📊 Static Forecasting Results - MCEAS02G01R02 (${targetAngle}):`);
console.log(`🎯 PLOTTING ALL ${origStatic.length} FRAMES from test file (MAXIMUM VISUALIZATION)`);
const mseStatic = MSE(origStatic, predStatic);
const maeStatic = MAE(origStatic, predStatic);
const u1Static = UTheil(origStatic, predStatic);
console.log(`MSE: ${mseStatic.toFixed(6)} | MAE: ${maeStatic.toFixed(6)} | U1: ${u1Static.toFixed(6)}`);

// Calculate correlation
const correlation = calculateCorrelation(origStatic, predStatic);
console.log(`Correlation: ${correlation.toFixed(6)}`);

// Generate static forecasting plots
console.log('\n🎨 Generating Static Forecasting Plots (ALL TEST FRAMES)...');
createPlot(origStatic, predStatic, `Static forecasting - MCEAS02G01R02: ${targetAngle} (ALL ${origStatic.length} FRAMES)`, 'static_mceas02_plot.html', 'Static', summary);

// Console plots for quick visualization
createConsolePlot(origStatic, predStatic, `Static Forecasting - MCEAS02G01R02 (${targetAngle}) - ALL ${origStatic.length} FRAMES`);

// 6. Dynamic Forecasting - TESTING on MCEAS02G01R02.bvh
console.log("\n6. Performing Dynamic Forecasting on MCEAS02G01R02.bvh (ALL FRAMES)...");

const { predDynamic, origValues: origDynamic } = dynamicForecasting(model, testData, indEnd, indExo, scalerTest, indEnd);

// Evaluate dynamic forecasting
console.log(`📊 Dynamic Forecasting Results - MCEAS02G01R02 (${targetAngle}):`);
console.log(`🎯 PLOTTING ALL ${origDynamic.length} FRAMES from test file (MAXIMUM VISUALIZATION)`);
const mseDynamic = MSE(origDynamic, predDynamic);
const maeDynamic = MAE(origDynamic, predDynamic);
const u1Dynamic = UTheil(origDynamic, predDynamic);
console.log(`MSE: ${mseDynamic.toFixed(6)} | MAE: ${maeDynamic.toFixed(6)} | U1: ${u1Dynamic.toFixed(6)}`);

const correlationDynamic = calculateCorrelation(origDynamic, predDynamic);
console.log(`Correlation: ${correlationDynamic.toFixed(6)}`);

// Generate dynamic forecasting plots
console.log('\n🎨 Generating Dynamic Forecasting Plots (ALL TEST FRAMES)...');
createPlot(origDynamic, predDynamic, `Dynamic forecasting - MCEAS02G01R02: ${targetAngle} (ALL ${origDynamic.length} FRAMES)`, 'dynamic_mceas02_plot.html', 'Dynamic', summary);

// Console plots for quick visualization
createConsolePlot(origDynamic, predDynamic, `Dynamic Forecasting - MCEAS02G01R02 (${targetAngle}) - ALL ${origDynamic.length} FRAMES`);

// 7. Final Summary
console.log("\n=== MCEAS02 ANALYSIS SUMMARY (SEPARATE TRAIN/TEST FILES) ===");
console.log("🏆 Model successfully trained and evaluated!");
console.log(`🏋️ Training: MCEAS02G01R03.bvh (${trainData.length} frames)`);
console.log(`🧪 Testing: MCEAS02G01R02.bvh (${testData.length} frames)`);
console.log(`📈 Target: ${targetAngle} with ${exogAngles.length} exogenous variables`);
console.log(`🎯 VISUALIZATION: Static Forecasting with ${origStatic.length} frames (ALL TEST FRAMES)`);
console.log(`🎯 VISUALIZATION: Dynamic Forecasting with ${origDynamic.length} frames (ALL TEST FRAMES)`);
console.log(`   Static:  MSE=${mseStatic.toFixed(6)}, MAE=${maeStatic.toFixed(6)}, U1=${u1Static.toFixed(6)}, Corr=${correlation.toFixed(6)}`);
console.log(`   Dynamic: MSE=${mseDynamic.toFixed(6)}, MAE=${maeDynamic.toFixed(6)}, U1=${u1Dynamic.toFixed(6)}, Corr=${correlationDynamic.toFixed(6)}`);

console.log("\n🎉 MCEAS02 Analysis Complete (ALL TEST FRAMES PLOTTED)!");
console.log("\n📂 Generated Files:");
console.log("   🌐 static_mceas02_plot.html - Interactive static forecasting plot (ALL TEST FRAMES)");
console.log("   🌐 dynamic_mceas02_plot.html - Interactive dynamic forecasting plot (ALL TEST FRAMES)");
console.log("\n💡 Open the HTML files in your browser to view ALL frames from MCEAS02G01R02.bvh!");

// Utility function for correlation calculation
function calculateCorrelation(x, y) {
    const n = x.length;
    const sumX = x.reduce((a, b) => a + b, 0);
    const sumY = y.reduce((a, b) => a + b, 0);
    const sumXY = x.reduce((sum, xi, i) => sum + xi * y[i], 0);
    const sumX2 = x.reduce((sum, xi) => sum + xi * xi, 0);
    const sumY2 = y.reduce((sum, yi) => sum + yi * yi, 0);
    
    const numerator = n * sumXY - sumX * sumY;
    const denominator = Math.sqrt((n * sumX2 - sumX * sumX) * (n * sumY2 - sumY * sumY));
    
    return numerator / denominator;
}

// Export for testing
export {
    SARIMAX,
    MinMaxScaler,
    StandardScaler,
    extractDataFromBVH,
    staticForecasting,
    dynamicForecasting,
    MSE,
    MAE,
    UTheil,
    createModelSummary,
    createPlot,
    createConsolePlot
}; 