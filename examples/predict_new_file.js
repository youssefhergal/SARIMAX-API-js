// 🔮 How to Predict on New BVH Files with Trained SARIMAX Model
// Complete workflow: Train on one file, predict on another

import { SARIMAX } from '../classes/SARIMAX.js';
import { StandardScaler } from '../classes/StandardScaler.js';
import { extractDataFromBVH } from '../utils/bvhUtils.js';
import { staticForecasting } from '../forecasting/staticForecasting.js';
import { dynamicForecasting } from '../forecasting/dynamicForecasting.js';
import { MSE, MAE, UTheil } from '../utils/metrics.js';
import { createPlot } from '../visualization/plotUtils.js';

console.log("🔮 SARIMAX Prediction on New BVH Files");
console.log("=" .repeat(50));

// ==========================================
// STEP 1: TRAIN MODEL ON FIRST FILE
// ==========================================
console.log("\n📚 STEP 1: Training Model on Training File");
console.log("-".repeat(50));

// Load training data (Bending motion)
const trainingFile = './BVH/Bending/Train_Bending.bvh';
const targetJoint = 'Hips_Xrotation';
const exogenousJoints = ['Spine_Yrotation', 'Spine_Zrotation', 'LeftArm_Xrotation', 'RightArm_Xrotation'];

console.log(`📂 Training file: ${trainingFile}`);
console.log(`🎯 Target joint: ${targetJoint}`);
console.log(`🔗 Exogenous joints: [${exogenousJoints.join(', ')}]`);

// Extract training data
const bvhTrainData = extractDataFromBVH(trainingFile, targetJoint, exogenousJoints);

// Prepare training data
const trainData = bvhTrainData.endog.map((endogValue, i) => [
  endogValue,
  ...bvhTrainData.exog[i]
]);

// IMPORTANT: Create and fit scaler on training data
const scaler = new StandardScaler();
const normalizedTrainData = scaler.fitTransform(trainData);

console.log(`✅ Training data loaded: ${trainData.length} frames`);
console.log(`📊 Scaler fitted - Mean: [${scaler.mean.map(x => x.toFixed(3)).join(', ')}]`);
console.log(`📊 Scaler fitted - Std:  [${scaler.std.map(x => x.toFixed(3)).join(', ')}]`);

// Extract variables for SARIMAX
const indEnd = 0;        // Target variable index
const indExo = [1,2,3,4]; // Exogenous variable indices

const endogTrain = normalizedTrainData.map(row => row[indEnd]);
const exogTrain = normalizedTrainData.map(row => indExo.map(idx => row[idx]));

// Train the model
console.log("\n🏋️ Training SARIMAX model...");
const model = new SARIMAX(endogTrain, exogTrain, 2);
model.fit();

const summary = model.summary();
console.log(`✅ Model trained successfully!`);
console.log(`📊 R-squared: ${summary.rSquared.toFixed(6)}`);
console.log(`📊 MSE: ${summary.mse.toFixed(6)}`);

// ==========================================
// STEP 2: LOAD NEW FILE FOR PREDICTION
// ==========================================
console.log("\n🔮 STEP 2: Loading New File for Prediction");
console.log("-".repeat(50));

// Load prediction data (Test file - different motion sequence)
const predictionFile = './BVH/Bending/Test_Bending.bvh';
console.log(`📂 Prediction file: ${predictionFile}`);

// IMPORTANT: Extract data with SAME joints in SAME order
const bvhPredData = extractDataFromBVH(predictionFile, targetJoint, exogenousJoints);

// Prepare prediction data (same format as training)
const predData = bvhPredData.endog.map((endogValue, i) => [
  endogValue,
  ...bvhPredData.exog[i]
]);

// CRITICAL: Use the SAME scaler fitted on training data
const normalizedPredData = scaler.transform(predData);

console.log(`✅ Prediction data loaded: ${predData.length} frames`);
console.log(`⚠️  Important: Used same scaler from training (no refitting!)`);

// Check data compatibility
if (predData[0].length !== trainData[0].length) {
  throw new Error(`❌ Data dimension mismatch! Training: ${trainData[0].length}, Prediction: ${predData[0].length}`);
}

console.log(`✅ Data compatibility check passed`);

// ==========================================
// STEP 3: MAKE PREDICTIONS
// ==========================================
console.log("\n🎯 STEP 3: Making Predictions");
console.log("-".repeat(50));

// 3a. Static Forecasting (one-step ahead)
console.log("\n📊 Static Forecasting (using real observations):");
const staticResults = staticForecasting(model, normalizedPredData, indEnd, indExo, scaler, indEnd);

const mseStatic = MSE(staticResults.origValues, staticResults.predStatic);
const maeStatic = MAE(staticResults.origValues, staticResults.predStatic);
const u1Static = UTheil(staticResults.origValues, staticResults.predStatic);

console.log(`📈 Static Results:`);
console.log(`  MSE: ${mseStatic.toFixed(6)}`);
console.log(`  MAE: ${maeStatic.toFixed(6)}`);
console.log(`  U-Theil: ${u1Static.toFixed(6)}`);
console.log(`  Predictions: ${staticResults.predStatic.length} frames`);

// 3b. Dynamic Forecasting (multi-step ahead)
console.log("\n📊 Dynamic Forecasting (using predicted values):");
const dynamicResults = dynamicForecasting(model, normalizedPredData, indEnd, indExo, scaler, indEnd);

const mseDynamic = MSE(dynamicResults.origValues, dynamicResults.predDynamic);
const maeDynamic = MAE(dynamicResults.origValues, dynamicResults.predDynamic);
const u1Dynamic = UTheil(dynamicResults.origValues, dynamicResults.predDynamic);

console.log(`📈 Dynamic Results:`);
console.log(`  MSE: ${mseDynamic.toFixed(6)}`);
console.log(`  MAE: ${maeDynamic.toFixed(6)}`);
console.log(`  U-Theil: ${u1Dynamic.toFixed(6)}`);
console.log(`  Predictions: ${dynamicResults.predDynamic.length} frames`);

// ==========================================
// STEP 4: MANUAL PREDICTION EXAMPLE
// ==========================================
console.log("\n🔧 STEP 4: Manual Prediction Example");
console.log("-".repeat(50));

// Show how to make a single prediction manually
const frameIndex = 50; // Predict frame 50

// Get the required inputs
const exogAtFrame = normalizedPredData[frameIndex].slice(1); // Exogenous variables
const endogHistory = normalizedPredData.slice(frameIndex-2, frameIndex).map(row => row[0]); // Last 2 values

console.log(`📍 Predicting frame ${frameIndex}:`);
console.log(`🔗 Exogenous inputs: [${exogAtFrame.map(x => x.toFixed(3)).join(', ')}]`);
console.log(`⏮️  Endogenous history: [${endogHistory.map(x => x.toFixed(3)).join(', ')}]`);

// Manual prediction calculation
const inputs = [...exogAtFrame, ...endogHistory];
let manualPrediction = 0;
for (let i = 0; i < summary.coefficients.length; i++) {
  manualPrediction += summary.coefficients[i] * inputs[i];
}

// Denormalize the prediction
const denormalizedPrediction = manualPrediction * scaler.std[indEnd] + scaler.mean[indEnd];
const actualValue = predData[frameIndex][indEnd];

console.log(`🎯 Normalized prediction: ${manualPrediction.toFixed(6)}`);
console.log(`📊 Denormalized prediction: ${denormalizedPrediction.toFixed(6)}`);
console.log(`✅ Actual value: ${actualValue.toFixed(6)}`);
console.log(`📏 Error: ${Math.abs(denormalizedPrediction - actualValue).toFixed(6)}`);

// ==========================================
// STEP 5: SAVE RESULTS AND VISUALIZATIONS
// ==========================================
console.log("\n💾 STEP 5: Saving Results");
console.log("-".repeat(50));

// Generate plots
const staticPlotFile = `prediction_static_${Date.now()}.html`;
const dynamicPlotFile = `prediction_dynamic_${Date.now()}.html`;

createPlot(
  staticResults.origValues, 
  staticResults.predStatic, 
  `Static Prediction - ${targetJoint}`, 
  staticPlotFile, 
  'Static'
);

createPlot(
  dynamicResults.origValues, 
  dynamicResults.predDynamic, 
  `Dynamic Prediction - ${targetJoint}`, 
  dynamicPlotFile, 
  'Dynamic'
);

console.log(`📊 Static plot saved: ${staticPlotFile}`);
console.log(`📊 Dynamic plot saved: ${dynamicPlotFile}`);

// ==========================================
// STEP 6: USING DIFFERENT MOTION TYPES
// ==========================================
console.log("\n🔄 STEP 6: Predicting Different Motion Type");
console.log("-".repeat(50));

// Example: Use a Bending-trained model on Glassblowing data
// (Note: This may not work well due to different motion patterns)

try {
  const glassblowingFile = './BVH/Glassblowing/Test_Glassblowing.bvh';
  console.log(`🧪 Experimental: Applying Bending model to Glassblowing data`);
  console.log(`📂 File: ${glassblowingFile}`);
  
  // Try to extract the same joints from Glassblowing data
  const bvhGlassData = extractDataFromBVH(glassblowingFile, targetJoint, exogenousJoints);
  
  if (bvhGlassData.endog.length > 0) {
    const glassData = bvhGlassData.endog.map((endogValue, i) => [
      endogValue,
      ...bvhGlassData.exog[i]
    ]);
    
    const normalizedGlassData = scaler.transform(glassData);
    const glassResults = staticForecasting(model, normalizedGlassData, indEnd, indExo, scaler, indEnd);
    
    const mseGlass = MSE(glassResults.origValues, glassResults.predStatic);
    console.log(`📊 Cross-motion MSE: ${mseGlass.toFixed(6)} (expect poor performance)`);
    console.log(`⚠️  Note: Models trained on one motion type may not generalize well to others`);
  }
} catch (error) {
  console.log(`⚠️  Cross-motion prediction failed: ${error.message}`);
  console.log(`💡 Tip: Train separate models for different motion types`);
}

// ==========================================
// SUMMARY AND BEST PRACTICES
// ==========================================
console.log("\n📋 SUMMARY: Best Practices for New File Prediction");
console.log("=".repeat(60));

console.log("✅ CRITICAL STEPS:");
console.log("1. 🎯 Use EXACT same joints in SAME order");
console.log("2. 📏 Apply SAME scaler (fitted on training data)");
console.log("3. 🔄 Don't refit scaler on prediction data");
console.log("4. ✅ Check data compatibility before prediction");
console.log("5. 📊 Denormalize predictions for interpretation");

console.log("\n⚠️  COMMON MISTAKES TO AVOID:");
console.log("❌ Refitting scaler on new data");
console.log("❌ Using different joint names or order");
console.log("❌ Forgetting to denormalize predictions");
console.log("❌ Using models across very different motion types");

console.log("\n🎯 WHEN THIS APPROACH WORKS BEST:");
console.log("✅ Same person, similar motion, different session");
console.log("✅ Same motion type with slight variations");
console.log("✅ Data from same recording setup/calibration");

console.log("\n📈 PERFORMANCE EXPECTATIONS:");
console.log(`📊 Training MSE: ${summary.mse.toFixed(6)}`);
console.log(`📊 Prediction MSE: ${mseStatic.toFixed(6)}`);
if (mseStatic < summary.mse * 2) {
  console.log("✅ Good generalization (prediction MSE < 2x training MSE)");
} else {
  console.log("⚠️  Poor generalization - consider retraining or different approach");
}

// Export trained model and scaler for reuse
export { model, scaler, summary, indEnd, indExo, targetJoint, exogenousJoints }; 