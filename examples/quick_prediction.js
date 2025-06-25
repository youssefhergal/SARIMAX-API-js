// 🚀 Quick Prediction on New BVH File - Simple Example
// Use this when you already have a trained model and just want to predict

import { SARIMAX } from '../classes/SARIMAX.js';
import { StandardScaler } from '../classes/StandardScaler.js';
import { extractDataFromBVH } from '../utils/bvhUtils.js';
import { staticForecasting } from '../forecasting/staticForecasting.js';
import { MSE, MAE } from '../utils/metrics.js';

console.log("🚀 Quick Prediction Example");
console.log("=" .repeat(30));

// =================================
// STEP 1: Train on first file
// =================================
console.log("\n1️⃣ Training model...");

// Load training data
const trainData = extractDataFromBVH(
  './BVH/Bending/Train_Bending.bvh',
  'Hips_Xrotation',
  ['Spine_Yrotation', 'Spine_Zrotation', 'LeftArm_Xrotation', 'RightArm_Xrotation']
);

// Prepare and normalize
const rawTrain = trainData.endog.map((y, i) => [y, ...trainData.exog[i]]);
const scaler = new StandardScaler();
const normalizedTrain = scaler.fitTransform(rawTrain);

// Train model
const endogTrain = normalizedTrain.map(row => row[0]);
const exogTrain = normalizedTrain.map(row => [row[1], row[2], row[3], row[4]]);

const model = new SARIMAX(endogTrain, exogTrain, 2);
model.fit();

console.log("✅ Model trained!");
console.log(`📊 R²: ${model.summary().rSquared.toFixed(4)}`);

// =================================
// STEP 2: Predict on new file
// =================================
console.log("\n2️⃣ Predicting on new file...");

// Load NEW file for prediction
const newData = extractDataFromBVH(
  './BVH/Bending/Test_Bending.bvh',  // 👈 Your new file here
  'Hips_Xrotation',                  // 👈 Same target joint
  ['Spine_Yrotation', 'Spine_Zrotation', 'LeftArm_Xrotation', 'RightArm_Xrotation'] // 👈 Same exog joints
);

// Prepare data (same format as training)
const rawNew = newData.endog.map((y, i) => [y, ...newData.exog[i]]);

// ⚠️ CRITICAL: Use SAME scaler (don't refit!)
const normalizedNew = scaler.transform(rawNew);

// Make predictions
const predictions = staticForecasting(
  model, 
  normalizedNew, 
  0,           // target index
  [1,2,3,4],   // exog indices
  scaler, 
  0
);

// Evaluate results
const mse = MSE(predictions.origValues, predictions.predStatic);
const mae = MAE(predictions.origValues, predictions.predStatic);

console.log("✅ Predictions complete!");
console.log(`📊 MSE: ${mse.toFixed(6)}`);
console.log(`📊 MAE: ${mae.toFixed(6)}`);
console.log(`📈 Predicted ${predictions.predStatic.length} frames`);

// =================================
// BONUS: Single frame prediction
// =================================
console.log("\n🎯 Single frame prediction example:");

const frameToPredict = 100;

// Get inputs for this frame
const exogInputs = normalizedNew[frameToPredict].slice(1); // [spine_y, spine_z, leftarm_x, rightarm_x]
const pastValues = [
  normalizedNew[frameToPredict-1][0],  // y(t-1)
  normalizedNew[frameToPredict-2][0]   // y(t-2)
];

// Calculate prediction manually
const allInputs = [...exogInputs, ...pastValues];
let prediction = 0;
for (let i = 0; i < model.coefficients.length; i++) {
  prediction += model.coefficients[i] * allInputs[i];
}

// Denormalize
const denormalizedPrediction = prediction * scaler.std[0] + scaler.mean[0];
const actualValue = rawNew[frameToPredict][0];

console.log(`Frame ${frameToPredict}:`);
console.log(`  Predicted: ${denormalizedPrediction.toFixed(4)}`);
console.log(`  Actual:    ${actualValue.toFixed(4)}`);
console.log(`  Error:     ${Math.abs(denormalizedPrediction - actualValue).toFixed(4)}`);

export { model, scaler }; 