// 🎯 SARIMAX Step-by-Step Walkthrough with Real Motion Data
// This example shows exactly how SARIMAX processes your motion capture data

import { SARIMAX } from '../classes/SARIMAX.js';
import { StandardScaler } from '../classes/StandardScaler.js';
import { extractDataFromBVH } from '../utils/bvhUtils.js';

console.log("🎬 SARIMAX Motion Analysis - Complete Walkthrough");
console.log("=" .repeat(60));

// 🎯 Step 1: Understanding Our Motion Data
console.log("\n📊 Step 1: Understanding the Motion Data");
console.log("-".repeat(40));

// Load real motion data
const bvhData = extractDataFromBVH(
  './BVH/Bending/Train_Bending.bvh',
  'Hips_Xrotation',  // What we want to predict
  ['Spine_Yrotation', 'Spine_Zrotation', 'LeftArm_Xrotation', 'RightArm_Xrotation']  // What helps predict it
);

console.log(`📈 Target Variable: Hips_Xrotation`);
console.log(`🔗 Exogenous Variables: Spine_Y, Spine_Z, LeftArm_X, RightArm_X`);
console.log(`⏱️  Total Frames: ${bvhData.endog.length}`);

// Show first few data points
console.log("\n📋 Sample Data (first 5 frames):");
for (let i = 0; i < 5; i++) {
  console.log(`Frame ${i}: Hip_X=${bvhData.endog[i].toFixed(3)}, Exog=[${bvhData.exog[i].map(x => x.toFixed(3)).join(', ')}]`);
}

// 🎯 Step 2: Data Preparation and Normalization
console.log("\n🔧 Step 2: Data Preparation");
console.log("-".repeat(40));

// Combine endogenous and exogenous data
const rawData = bvhData.endog.map((endogValue, i) => [
  endogValue,           // Target: Hips_Xrotation at index 0
  ...bvhData.exog[i]    // Exogenous: [Spine_Y, Spine_Z, LeftArm_X, RightArm_X] at indices 1-4
]);

console.log(`📊 Raw data shape: ${rawData.length} frames × ${rawData[0].length} variables`);
console.log(`📋 Variable order: [Hips_X, Spine_Y, Spine_Z, LeftArm_X, RightArm_X]`);

// Normalize the data
const scaler = new StandardScaler();
const normalizedData = scaler.fitTransform(rawData);

console.log("\n📏 Normalization Results:");
console.log(`📊 Mean: [${scaler.mean.map(x => x.toFixed(3)).join(', ')}]`);
console.log(`📊 Std:  [${scaler.std.map(x => x.toFixed(3)).join(', ')}]`);

// Show effect of normalization
console.log("\n📈 Before/After Normalization (Frame 0):");
console.log(`Before: [${rawData[0].map(x => x.toFixed(3)).join(', ')}]`);
console.log(`After:  [${normalizedData[0].map(x => x.toFixed(3)).join(', ')}]`);

// 🎯 Step 3: Extracting Variables for SARIMAX
console.log("\n🎯 Step 3: Setting up SARIMAX Variables");
console.log("-".repeat(40));

const indEnd = 0;        // Target variable index (Hips_Xrotation)
const indExo = [1,2,3,4]; // Exogenous variable indices

const endogData = normalizedData.map(row => row[indEnd]);
const exogData = normalizedData.map(row => indExo.map(idx => row[idx]));

console.log(`🎯 Endogenous (target): ${endogData.length} values`);
console.log(`🔗 Exogenous (predictors): ${exogData.length} rows × ${exogData[0].length} columns`);
console.log(`📋 First endogenous values: [${endogData.slice(0,5).map(x => x.toFixed(3)).join(', ')}]`);
console.log(`📋 First exogenous row: [${exogData[0].map(x => x.toFixed(3)).join(', ')}]`);

// 🎯 Step 4: Understanding the SARIMAX Model Creation
console.log("\n🧮 Step 4: Creating SARIMAX Model");
console.log("-".repeat(40));

const order = 2;  // AR(2) - use last 2 time steps
console.log(`📊 Model Order: AR(${order}) - Using last ${order} time steps`);
console.log(`🔢 This means our model equation will be:`);
console.log(`   Hips_X(t) = β₀ + β₁×Spine_Y(t) + β₂×Spine_Z(t) + β₃×LeftArm_X(t) + β₄×RightArm_X(t)`);
console.log(`              + φ₁×Hips_X(t-1) + φ₂×Hips_X(t-2) + ε(t)`);
console.log(`🔢 Total parameters to estimate: ${4 + order} (4 exogenous + ${order} autoregressive)`);

// Create and train the model
const model = new SARIMAX(endogData, exogData, order);

// 🎯 Step 5: Understanding the Training Process
console.log("\n🏋️ Step 5: Model Training Process");
console.log("-".repeat(40));

console.log("📊 Training process breakdown:");
console.log("1. Create lagged endogenous variables (AR terms)");
console.log("2. Combine exogenous and lagged variables into design matrix X");
console.log("3. Solve linear regression: X × β = y");
console.log("4. Calculate statistical measures (p-values, R², etc.)");

// Manually show what the training data looks like
console.log("\n📋 Training Data Structure (first 3 rows):");
console.log("Row format: [Spine_Y, Spine_Z, LeftArm_X, RightArm_X, Hips_X(t-1), Hips_X(t-2)] -> Target: Hips_X(t)");

// Simulate what the model sees during training
for (let i = order; i < order + 3; i++) {
  const exogRow = exogData[i];
  const laggedEndog = [endogData[i-1], endogData[i-2]];
  const target = endogData[i];
  
  console.log(`Row ${i-order}: [${[...exogRow, ...laggedEndog].map(x => x.toFixed(3)).join(', ')}] -> ${target.toFixed(3)}`);
}

// Train the model
console.log("\n🔄 Training model...");
model.fit();
console.log("✅ Training complete!");

// 🎯 Step 6: Interpreting the Results
console.log("\n📊 Step 6: Model Results Interpretation");
console.log("-".repeat(40));

const summary = model.summary();
const variables = ['Spine_Yrotation', 'Spine_Zrotation', 'LeftArm_Xrotation', 'RightArm_Xrotation', 
                  'Hips_Xrotation_T-1', 'Hips_Xrotation_T-2'];

console.log("🔢 Estimated Coefficients:");
for (let i = 0; i < variables.length; i++) {
  const coef = summary.coefficients[i];
  const pVal = summary.pValues[i];
  const significance = pVal < 0.001 ? '***' : pVal < 0.01 ? '**' : pVal < 0.05 ? '*' : '';
  
  console.log(`  ${variables[i].padEnd(20)}: ${coef.toFixed(6)} (p=${pVal.toFixed(6)}) ${significance}`);
}

console.log(`\n📈 Model Quality:`);
console.log(`  R-squared: ${summary.rSquared.toFixed(6)} (${(summary.rSquared * 100).toFixed(1)}% variance explained)`);
console.log(`  MSE: ${summary.mse.toFixed(6)}`);
console.log(`  AIC: ${summary.aic.toFixed(6)}`);
console.log(`  BIC: ${summary.bic.toFixed(6)}`);

// 🎯 Step 7: Understanding Prediction
console.log("\n🔮 Step 7: How Prediction Works");
console.log("-".repeat(40));

// Show a single prediction calculation
const predictionIndex = order + 10;  // Predict frame 12 (index starts from order=2)
const exogAtTime = exogData[predictionIndex];
const pastValues = [endogData[predictionIndex-1], endogData[predictionIndex-2]];
const actualValue = endogData[predictionIndex];

console.log(`📍 Predicting Hips_Xrotation at frame ${predictionIndex}:`);
console.log(`📊 Exogenous inputs: [${exogAtTime.map(x => x.toFixed(3)).join(', ')}]`);
console.log(`⏮️  Past values: [t-1: ${pastValues[0].toFixed(3)}, t-2: ${pastValues[1].toFixed(3)}]`);

// Manual prediction calculation
const inputs = [...exogAtTime, ...pastValues];
let prediction = 0;
for (let i = 0; i < inputs.length; i++) {
  prediction += summary.coefficients[i] * inputs[i];
}

console.log(`🔢 Calculation:`);
for (let i = 0; i < variables.length; i++) {
  const contribution = summary.coefficients[i] * inputs[i];
  console.log(`  ${variables[i].padEnd(20)}: ${summary.coefficients[i].toFixed(3)} × ${inputs[i].toFixed(3)} = ${contribution.toFixed(3)}`);
}
console.log(`🎯 Predicted value: ${prediction.toFixed(6)}`);
console.log(`✅ Actual value: ${actualValue.toFixed(6)}`);
console.log(`📏 Error: ${Math.abs(prediction - actualValue).toFixed(6)}`);

// 🎯 Step 8: Biomechanical Interpretation
console.log("\n🏃 Step 8: Biomechanical Interpretation");
console.log("-".repeat(40));

console.log("🧠 What the coefficients tell us about human movement:");

for (let i = 0; i < variables.length; i++) {
  const coef = summary.coefficients[i];
  const variable = variables[i];
  
  if (variable.includes('Spine_Yrotation')) {
    console.log(`  📐 ${variable}: ${coef > 0 ? 'Positive' : 'Negative'} spine twist ${coef > 0 ? 'increases' : 'decreases'} hip forward bend`);
  } else if (variable.includes('Spine_Zrotation')) {
    console.log(`  📐 ${variable}: ${coef > 0 ? 'Positive' : 'Negative'} spine side bend ${coef > 0 ? 'increases' : 'decreases'} hip forward bend`);
  } else if (variable.includes('LeftArm_Xrotation')) {
    console.log(`  💪 ${variable}: Left arm movement ${coef > 0 ? 'positively' : 'negatively'} affects hip position`);
  } else if (variable.includes('RightArm_Xrotation')) {
    console.log(`  💪 ${variable}: Right arm movement ${coef > 0 ? 'positively' : 'negatively'} affects hip position`);
  } else if (variable.includes('T-1')) {
    console.log(`  ⏮️  ${variable}: Previous hip position has ${Math.abs(coef) > 0.5 ? 'strong' : 'moderate'} influence (momentum effect)`);
  } else if (variable.includes('T-2')) {
    console.log(`  ⏮️  ${variable}: Two-step-ago position has ${Math.abs(coef) > 0.3 ? 'notable' : 'weak'} influence`);
  }
}

console.log("\n🎉 Analysis Complete!");
console.log("💡 This demonstrates how SARIMAX captures both:");
console.log("   🔄 Temporal dependencies (momentum, inertia)");
console.log("   🤝 Inter-joint coordination (biomechanical coupling)");

// Export the trained model for further use
export { model, scaler, variables }; 