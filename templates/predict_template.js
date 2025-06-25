// 📝 Template for Predicting on Your Own BVH Files
// ✏️ Modify the paths and joints below for your specific use case

import { SARIMAX } from '../classes/SARIMAX.js';
import { StandardScaler } from '../classes/StandardScaler.js';
import { extractDataFromBVH } from '../utils/bvhUtils.js';
import { staticForecasting } from '../forecasting/staticForecasting.js';
import { dynamicForecasting } from '../forecasting/dynamicForecasting.js';
import { MSE, MAE, UTheil } from '../utils/metrics.js';
import { createPlot } from '../visualization/plotUtils.js';

// =================================
// 🔧 CONFIGURATION - MODIFY HERE
// =================================

// File paths
const TRAINING_FILE = './BVH/Bending/Train_Bending.bvh';     // 👈 Change to your training file
const PREDICTION_FILE = './BVH/Bending/Test_Bending.bvh';   // 👈 Change to your prediction file

// Joint configuration
const TARGET_JOINT = 'Hips_Xrotation';                      // 👈 Joint you want to predict
const EXOGENOUS_JOINTS = [                                  // 👈 Joints that help predict
  'Spine_Yrotation',
  'Spine_Zrotation', 
  'LeftArm_Xrotation',
  'RightArm_Xrotation'
];

// Model configuration
const AR_ORDER = 2;                                          // 👈 Number of past time steps to use

// Output configuration
const SAVE_PLOTS = true;                                     // 👈 Generate HTML plots?
const PLOT_PREFIX = 'my_prediction';                         // 👈 Prefix for plot filenames

// =================================
// 🚀 EXECUTION - DON'T MODIFY BELOW
// =================================

console.log("🎯 SARIMAX Prediction Template");
console.log("=" .repeat(40));

console.log(`📂 Training file: ${TRAINING_FILE}`);
console.log(`📂 Prediction file: ${PREDICTION_FILE}`);
console.log(`🎯 Target joint: ${TARGET_JOINT}`);
console.log(`🔗 Exogenous joints: [${EXOGENOUS_JOINTS.join(', ')}]`);

// Step 1: Load and train model
console.log("\n1️⃣ Loading training data and training model...");

const trainData = extractDataFromBVH(TRAINING_FILE, TARGET_JOINT, EXOGENOUS_JOINTS);
const rawTrain = trainData.endog.map((y, i) => [y, ...trainData.exog[i]]);

const scaler = new StandardScaler();
const normalizedTrain = scaler.fitTransform(rawTrain);

const endogTrain = normalizedTrain.map(row => row[0]);
const exogTrain = normalizedTrain.map(row => row.slice(1));

const model = new SARIMAX(endogTrain, exogTrain, AR_ORDER);
model.fit();

console.log(`✅ Model trained on ${trainData.endog.length} frames`);
console.log(`📊 Training R²: ${model.summary().rSquared.toFixed(4)}`);

// Step 2: Load prediction data
console.log("\n2️⃣ Loading prediction data...");

const predData = extractDataFromBVH(PREDICTION_FILE, TARGET_JOINT, EXOGENOUS_JOINTS);
const rawPred = predData.endog.map((y, i) => [y, ...predData.exog[i]]);
const normalizedPred = scaler.transform(rawPred);

console.log(`✅ Prediction data loaded: ${predData.endog.length} frames`);

// Step 3: Make predictions
console.log("\n3️⃣ Making predictions...");

// Static forecasting
const staticResults = staticForecasting(
  model, 
  normalizedPred, 
  0,                    // target index
  [1,2,3,4].slice(0, EXOGENOUS_JOINTS.length), // exog indices
  scaler, 
  0
);

const mseStatic = MSE(staticResults.origValues, staticResults.predStatic);
const maeStatic = MAE(staticResults.origValues, staticResults.predStatic);
const u1Static = UTheil(staticResults.origValues, staticResults.predStatic);

console.log(`📊 Static Forecasting Results:`);
console.log(`   MSE: ${mseStatic.toFixed(6)}`);
console.log(`   MAE: ${maeStatic.toFixed(6)}`);
console.log(`   U-Theil: ${u1Static.toFixed(6)}`);

// Dynamic forecasting  
const dynamicResults = dynamicForecasting(
  model, 
  normalizedPred, 
  0, 
  [1,2,3,4].slice(0, EXOGENOUS_JOINTS.length), 
  scaler, 
  0
);

const mseDynamic = MSE(dynamicResults.origValues, dynamicResults.predDynamic);
const maeDynamic = MAE(dynamicResults.origValues, dynamicResults.predDynamic);

console.log(`📊 Dynamic Forecasting Results:`);
console.log(`   MSE: ${mseDynamic.toFixed(6)}`);
console.log(`   MAE: ${maeDynamic.toFixed(6)}`);

// Step 4: Save results
if (SAVE_PLOTS) {
  console.log("\n4️⃣ Saving plots...");
  
  const timestamp = Date.now();
  const staticFile = `${PLOT_PREFIX}_static_${timestamp}.html`;
  const dynamicFile = `${PLOT_PREFIX}_dynamic_${timestamp}.html`;
  
  createPlot(
    staticResults.origValues, 
    staticResults.predStatic,
    `Static Prediction - ${TARGET_JOINT}`,
    staticFile,
    'Static'
  );
  
  createPlot(
    dynamicResults.origValues, 
    dynamicResults.predDynamic,
    `Dynamic Prediction - ${TARGET_JOINT}`,
    dynamicFile,
    'Dynamic'
  );
  
  console.log(`📊 Plots saved: ${staticFile}, ${dynamicFile}`);
}

// Step 5: Summary
console.log("\n🎉 Prediction Complete!");
console.log(`✅ Static prediction MSE: ${mseStatic.toFixed(6)}`);
console.log(`✅ Dynamic prediction MSE: ${mseDynamic.toFixed(6)}`);

if (mseStatic < 1.0) {
  console.log("🎯 Excellent prediction quality!");
} else if (mseStatic < 5.0) {
  console.log("👍 Good prediction quality");
} else {
  console.log("⚠️ Consider retraining or using different joints");
}

// Export for further use
export { model, scaler, staticResults, dynamicResults };

// =================================
// 📋 USAGE INSTRUCTIONS
// =================================

/*
🔧 How to use this template:

1. Modify the configuration section above:
   - Change TRAINING_FILE and PREDICTION_FILE paths
   - Change TARGET_JOINT to the joint you want to predict
   - Change EXOGENOUS_JOINTS to joints that influence your target
   - Adjust AR_ORDER if needed (usually 2 is good)

2. Run the script:
   node templates/predict_template.js

3. Check the console output for:
   - Training quality (R²)
   - Prediction errors (MSE, MAE)
   - Generated plot filenames

4. Open the HTML files in your browser to see the results

📊 Understanding the results:
   - MSE < 1.0 = Excellent
   - MSE 1.0-5.0 = Good  
   - MSE > 5.0 = Poor (consider different joints or more training data)

⚠️ Important notes:
   - Training and prediction files must have the same joint names
   - Works best with similar motion types
   - Model quality depends on how predictable your target joint is
*/ 