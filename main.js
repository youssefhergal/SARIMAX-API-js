// Main SARIMAX Motion Analysis Script
// Modular implementation with separated concerns

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

// Main execution - JavaScript Notebook Implementation
console.log("🚀 State Space Representation and Forecasting of Joint Angles - JavaScript Implementation");
console.log("==================================================================================\n");

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

// 1. Load BVH files to extract local joint angles
console.log("1. Loading BVH files...");

// Define paths (modify as needed)
const pathsB1 = './BVH/Bending/Train_Bending.bvh';
const pathsB2 = './BVH/Bending/Test_Bending.bvh';
const pathsG1 = './BVH/Glassblowing/Train_Glassblowing.bvh';
const pathsG2 = './BVH/Glassblowing/Test_Glassblowing.bvh';

// Load real BVH data for all motions
console.log("📁 Loading motion capture data...");

// Load Bending motion data - using Hips_Xrotation as target
const targetAngleB = 'Hips_Xrotation';
const exogAnglesB = ALL_BVH_ANGLES.filter(angle => angle !== targetAngleB);

console.log("🔄 Loading Bending motion files...");
const bvhTrainB = extractDataFromBVH(
  './BVH/Bending/Train_Bending.bvh',
  targetAngleB,
  exogAnglesB
);

const bvhTestB = extractDataFromBVH(
  './BVH/Bending/Test_Bending.bvh',
  targetAngleB,
  exogAnglesB
);

// Load Glassblowing motion data - using LeftForeArm_Yrotation as target
const targetAngleG = 'LeftForeArm_Yrotation';
const exogAnglesG = ALL_BVH_ANGLES.filter(angle => angle !== targetAngleG);

console.log("🔄 Loading Glassblowing motion files...");
const bvhTrainG = extractDataFromBVH(
  './BVH/Glassblowing/Train_Glassblowing.bvh',
  targetAngleG,
  exogAnglesG
);

const bvhTestG = extractDataFromBVH(
  './BVH/Glassblowing/Test_Glassblowing.bvh',
  targetAngleG,
  exogAnglesG
);

console.log(`✅ Loaded Bending: Train=${bvhTrainB.endog.length} frames, Test=${bvhTestB.endog.length} frames`);
console.log(`✅ Loaded Glassblowing: Train=${bvhTrainG.endog.length} frames, Test=${bvhTestG.endog.length} frames`);
console.log(`📊 Using ${exogAnglesB.length} exogenous variables for each model`);

// 2. Convert BVH data to multi-channel format for SARIMAX
console.log("2. Converting BVH data to multi-channel format...");

// Create multi-channel datasets from BVH data
const trainDataB = bvhTrainB.endog.map((endogValue, i) => [
  endogValue,
  ...bvhTrainB.exog[i]
]);

const testDataB = bvhTestB.endog.map((endogValue, i) => [
  endogValue,
  ...bvhTestB.exog[i]
]);

const trainDataG = bvhTrainG.endog.map((endogValue, i) => [
  endogValue,
  ...bvhTrainG.exog[i]
]);

const testDataG = bvhTestG.endog.map((endogValue, i) => [
  endogValue,
  ...bvhTestG.exog[i]
]);

console.log(`🔧 Converted data formats - Total channels: ${trainDataB[0].length} (1 target + ${trainDataB[0].length - 1} exogenous)`);

// 3. Normalize data (using StandardScaler to match Python sklearn behavior)
console.log("3. Normalizing data with StandardScaler...");
const scalerTrainB = new StandardScaler();
const scalerTestB = new StandardScaler();
const scalerTrainG = new StandardScaler();
const scalerTestG = new StandardScaler();

const dataTrainB = scalerTrainB.fitTransform(trainDataB);
const dataTestB = scalerTestB.fitTransform(testDataB);
const dataTrainG = scalerTrainG.fitTransform(trainDataG);
const dataTestG = scalerTestG.fitTransform(testDataG);

// 4. Train SARIMAX models
console.log("4. Training SARIMAX models...");

// 4.1 Bending motion model
const order = 2;
const angB = targetAngleB;
// Create complete angle list: [target, ...exogenous_variables]
const bvhAnglesB = [targetAngleB, ...exogAnglesB];
const indEndB = 0; // Target is at index 0
const indExoB = Array.from({length: exogAnglesB.length}, (_, i) => i + 1); // All other indices

const endogB = dataTrainB.map(row => row[indEndB]);
const exogB = dataTrainB.map(row => indExoB.map(idx => row[idx]));

console.log(`🎯 Training model for Bending motion - Target: ${angB}`);
console.log(`📊 Training data: ${endogB.length} frames, ${exogB[0].length} exogenous variables`);
const modelB = new SARIMAX(endogB, exogB, order);
modelB.fit();

const summaryB = createModelSummary(modelB, bvhAnglesB, angB, indExoB);

// Display detailed model table (pandas-like)
console.log('\n📊 DETAILED MODEL ANALYSIS - BENDING MOTION');
const modelTableB = displayModelTable(modelB, bvhAnglesB, angB, indExoB);
const dfModelB = createDataFrame(modelTableB);

// Show DataFrame-like operations
dfModelB.head();
dfModelB.describe();
dfModelB.info();

// 4.2 Glassblowing motion model
const angG = targetAngleG;
// Create complete angle list: [target, ...exogenous_variables]
const bvhAnglesG = [targetAngleG, ...exogAnglesG];
const indEndG = 0; // Target is at index 0
const indExoG = Array.from({length: exogAnglesG.length}, (_, i) => i + 1); // All other indices

const endogG = dataTrainG.map(row => row[indEndG]);
const exogG = dataTrainG.map(row => indExoG.map(idx => row[idx]));

console.log(`🎯 Training model for Glassblowing motion - Target: ${angG}`);
console.log(`📊 Training data: ${endogG.length} frames, ${exogG[0].length} exogenous variables`);
const modelG = new SARIMAX(endogG, exogG, order);
modelG.fit();

const summaryG = createModelSummary(modelG, bvhAnglesG, angG, indExoG);

// Display detailed model table (pandas-like)
console.log('\n📊 DETAILED MODEL ANALYSIS - GLASSBLOWING MOTION');
const modelTableG = displayModelTable(modelG, bvhAnglesG, angG, indExoG);
const dfModelG = createDataFrame(modelTableG);

// Show DataFrame-like operations
dfModelG.head();
dfModelG.describe();
dfModelG.info();

// Compare both models
console.log('\n🔍 MODEL COMPARISON BETWEEN MOTIONS');
compareModels([
  { name: 'Bending', modelTable: modelTableB },
  { name: 'Glassblowing', modelTable: modelTableG }
]);

// 5. Static Forecasting
console.log("\n5. Performing Static Forecasting...");

const { predStatic: predStaticB, origValues: origStaticB } = staticForecasting(modelB, dataTestB, indEndB, indExoB, scalerTestB, indEndB);
const { predStatic: predStaticG, origValues: origStaticG } = staticForecasting(modelG, dataTestG, indEndG, indExoG, scalerTestG, indEndG);

// Evaluate static forecasting
console.log(`📊 Static Forecasting Results - Bending (${angB}):`);
const mseStaticB = MSE(origStaticB, predStaticB);
const maeStaticB = MAE(origStaticB, predStaticB);
const u1StaticB = UTheil(origStaticB, predStaticB);
console.log(`MSE: ${mseStaticB.toFixed(6)} | MAE: ${maeStaticB.toFixed(6)} | U1: ${u1StaticB.toFixed(6)}`);

console.log(`📊 Static Forecasting Results - Glassblowing (${angG}):`);
const mseStaticG = MSE(origStaticG, predStaticG);
const maeStaticG = MAE(origStaticG, predStaticG);
const u1StaticG = UTheil(origStaticG, predStaticG);
console.log(`MSE: ${mseStaticG.toFixed(6)} | MAE: ${maeStaticG.toFixed(6)} | U1: ${u1StaticG.toFixed(6)}`);

// Generate static forecasting plots
console.log('\n🎨 Generating Static Forecasting Plots...');
createPlot(origStaticB, predStaticB, `Static forecasting - Bending: ${angB}`, 'static_bending_plot.html', 'Static', summaryB);
createPlot(origStaticG, predStaticG, `Static forecasting - Glassblowing: ${angG}`, 'static_glassblowing_plot.html', 'Static', summaryG);

// Console plots for quick visualization
createConsolePlot(origStaticB, predStaticB, `Static Forecasting - Bending (${angB})`);
createConsolePlot(origStaticG, predStaticG, `Static Forecasting - Glassblowing (${angG})`);

// 6. Dynamic Forecasting
console.log("\n6. Performing Dynamic Forecasting...");

const { predDynamic: predDynamicB, origValues: origDynamicB } = dynamicForecasting(modelB, dataTestB, indEndB, indExoB, scalerTestB, indEndB);
const { predDynamic: predDynamicG, origValues: origDynamicG } = dynamicForecasting(modelG, dataTestG, indEndG, indExoG, scalerTestG, indEndG);

// Evaluate dynamic forecasting
console.log(`📊 Dynamic Forecasting Results - Bending (${angB}):`);
const mseDynamicB = MSE(origDynamicB, predDynamicB);
const maeDynamicB = MAE(origDynamicB, predDynamicB);
const u1DynamicB = UTheil(origDynamicB, predDynamicB);
console.log(`MSE: ${mseDynamicB.toFixed(6)} | MAE: ${maeDynamicB.toFixed(6)} | U1: ${u1DynamicB.toFixed(6)}`);

console.log(`📊 Dynamic Forecasting Results - Glassblowing (${angG}):`);
const mseDynamicG = MSE(origDynamicG, predDynamicG);
const maeDynamicG = MAE(origDynamicG, predDynamicG);
const u1DynamicG = UTheil(origDynamicG, predDynamicG);
console.log(`MSE: ${mseDynamicG.toFixed(6)} | MAE: ${maeDynamicG.toFixed(6)} | U1: ${u1DynamicG.toFixed(6)}`);

// Generate dynamic forecasting plots
console.log('\n🎨 Generating Dynamic Forecasting Plots...');
createPlot(origDynamicB, predDynamicB, `Dynamic forecasting - Bending: ${angB}`, 'dynamic_bending_plot.html', 'Dynamic', summaryB);
createPlot(origDynamicG, predDynamicG, `Dynamic forecasting - Glassblowing: ${angG}`, 'dynamic_glassblowing_plot.html', 'Dynamic', summaryG);

// Console plots for quick visualization
createConsolePlot(origDynamicB, predDynamicB, `Dynamic Forecasting - Bending (${angB})`);
createConsolePlot(origDynamicG, predDynamicG, `Dynamic Forecasting - Glassblowing (${angG})`);

// 7. Summary and Results
console.log("\n=== FINAL SUMMARY ===");
console.log("🏆 Models successfully trained and evaluated!");
console.log(`📈 Bending Motion (${angB}) with ${exogAnglesB.length} exogenous variables:`);
console.log(`   Static:  MSE=${mseStaticB.toFixed(6)}, MAE=${maeStaticB.toFixed(6)}, U1=${u1StaticB.toFixed(6)}`);
console.log(`   Dynamic: MSE=${mseDynamicB.toFixed(6)}, MAE=${maeDynamicB.toFixed(6)}, U1=${u1DynamicB.toFixed(6)}`);
console.log(`📈 Glassblowing Motion (${angG}) with ${exogAnglesG.length} exogenous variables:`);
console.log(`   Static:  MSE=${mseStaticG.toFixed(6)}, MAE=${maeStaticG.toFixed(6)}, U1=${u1StaticG.toFixed(6)}`);
console.log(`   Dynamic: MSE=${mseDynamicG.toFixed(6)}, MAE=${maeDynamicG.toFixed(6)}, U1=${u1DynamicG.toFixed(6)}`);

console.log("\n🎉 JavaScript SARIMAX Motion Analysis Complete!");
console.log("\n📂 Generated Files:");
console.log("   🌐 static_bending_plot.html - Interactive static forecasting plot for bending motion");
console.log("   🌐 static_glassblowing_plot.html - Interactive static forecasting plot for glassblowing motion");
console.log("   🌐 dynamic_bending_plot.html - Interactive dynamic forecasting plot for bending motion");
console.log("   🌐 dynamic_glassblowing_plot.html - Interactive dynamic forecasting plot for glassblowing motion");
console.log("\n💡 Open the HTML files in your browser to view interactive plots!");

// Export main functions for external use
export {
  SARIMAX,
  MinMaxScaler,
  StandardScaler,
  extractDataFromBVH,
  extractEulerAngles,
  staticForecasting,
  dynamicForecasting,
  MSE,
  MAE,
  UTheil,
  createModelSummary,
  createPlot,
  createConsolePlot
}; 