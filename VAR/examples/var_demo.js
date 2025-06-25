// 🎪 VAR Model Demo - Full-Body Motion Prediction
// Demonstrates how to use the JavaScript VAR implementation

import { VARModel } from '../VARModel.js';
import { KfGom } from '../KfGom.js';
import { extractDataFromBVH } from '../../utils/bvhUtils.js';
import { StandardScaler } from '../../classes/StandardScaler.js';

console.log("🎪 VAR Model Demo - Full-Body Motion Prediction");
console.log("=" .repeat(60));

// =================================
// DEMO 1: Simple VAR Model
// =================================
console.log("\n📊 DEMO 1: Simple VAR Model with 3 Variables");
console.log("-".repeat(50));

// Create synthetic data for demonstration
const generateSyntheticData = (length = 100) => {
  const data = [];
  let x1 = 0, x2 = 0, x3 = 0;
  
  for (let t = 0; t < length; t++) {
    // Simple VAR(1) system:
    // x1(t) = 0.5*x1(t-1) + 0.2*x2(t-1) + noise
    // x2(t) = 0.3*x1(t-1) + 0.6*x2(t-1) + 0.1*x3(t-1) + noise
    // x3(t) = 0.1*x1(t-1) + 0.4*x3(t-1) + noise
    
    const newX1 = 0.5 * x1 + 0.2 * x2 + (Math.random() - 0.5) * 0.1;
    const newX2 = 0.3 * x1 + 0.6 * x2 + 0.1 * x3 + (Math.random() - 0.5) * 0.1;
    const newX3 = 0.1 * x1 + 0.4 * x3 + (Math.random() - 0.5) * 0.1;
    
    data.push([newX1, newX2, newX3]);
    
    x1 = newX1;
    x2 = newX2;
    x3 = newX3;
  }
  
  return data;
};

const syntheticData = generateSyntheticData(200);
console.log(`📈 Generated ${syntheticData.length} synthetic observations`);

// Train VAR model
const varModel = new VARModel(2); // VAR(2)
varModel.fit(syntheticData, ['X1', 'X2', 'X3']);

// Make predictions
const testData = syntheticData.slice(0, 10); // Use first 10 for prediction
const predictions = varModel.predict(testData, 5); // Predict 5 steps ahead

console.log("📊 Synthetic Data Results:");
console.log(`✅ Model trained on ${syntheticData.length} observations`);
console.log(`🔮 Generated ${predictions.length} predictions`);
console.log(`📋 Sample prediction:`, predictions[0].map(x => x.toFixed(4)));

// =================================
// DEMO 2: KfGom with Real BVH Data
// =================================
console.log("\n🎪 DEMO 2: KfGom with BVH Motion Data");
console.log("-".repeat(50));

try {
  // Define a subset of variables for faster processing
  const targetVariables = [
    'Hips_Xrotation', 'Hips_Yrotation', 'Hips_Zrotation',
    'Spine_Xrotation', 'Spine_Yrotation', 'Spine_Zrotation',
    'LeftArm_Xrotation', 'LeftArm_Yrotation', 'LeftArm_Zrotation',
    'RightArm_Xrotation', 'RightArm_Yrotation', 'RightArm_Zrotation'
  ];

  // Load BVH data
  console.log("📂 Loading BVH data...");
  const bvhData = extractDataFromBVH(
    './BVH/Bending/Train_Bending.bvh',
    targetVariables[0],
    targetVariables.slice(1)
  );

  // Prepare data matrix
  const motionData = bvhData.endog.map((endogValue, i) => [
    endogValue,
    ...bvhData.exog[i]
  ]);

  console.log(`📊 Loaded motion data: ${motionData.length} frames × ${motionData[0].length} joints`);

  // Take a subset for faster processing
  const sampleSize = Math.min(500, motionData.length);
  const sampledData = motionData.slice(0, sampleSize);
  
  console.log(`🎯 Using ${sampleSize} frames for demo`);

  // Normalize data
  const scaler = new StandardScaler();
  const normalizedData = scaler.fitTransform(sampledData);

  // Create and train KfGom model
  console.log("🏋️ Training KfGom model...");
  const kfGom = new KfGom(targetVariables);
  const { coef, dfPred } = kfGom.doGom(normalizedData);

  // Calculate metrics
  const metrics = kfGom.calculateMetrics();

  console.log("📊 KfGom Results:");
  console.log(`✅ Model trained on ${targetVariables.length} variables`);
  console.log(`🔮 Generated ${dfPred.length} predictions`);

  // Show metrics for first few variables
  console.log("\n📈 Performance Metrics (first 5 variables):");
  Object.entries(metrics).slice(0, 5).forEach(([variable, metric]) => {
    console.log(`  ${variable}:`);
    console.log(`    MSE: ${metric.mse.toFixed(6)}`);
    console.log(`    Correlation: ${metric.correlation.toFixed(4)}`);
  });

  // Show some coefficients
  console.log("\n🔢 Sample Coefficients (Hips_Xrotation):");
  const hipsCoefs = coef['Hips_Xrotation'];
  Object.entries(hipsCoefs).slice(0, 10).forEach(([label, value]) => {
    console.log(`  ${label}: ${value.toFixed(6)}`);
  });

  // =================================
  // DEMO 3: Prediction on New Data
  // =================================
  console.log("\n🔮 DEMO 3: Prediction on New Motion Data");
  console.log("-".repeat(50));

  try {
    // Load test data
    const testBvhData = extractDataFromBVH(
      './BVH/Bending/Test_Bending.bvh',
      targetVariables[0],
      targetVariables.slice(1)
    );

    const testMotionData = testBvhData.endog.map((endogValue, i) => [
      endogValue,
      ...testBvhData.exog[i]
    ]);

    // Take subset and normalize with same scaler
    const testSample = testMotionData.slice(0, Math.min(100, testMotionData.length));
    const normalizedTestData = scaler.transform(testSample);

    console.log(`📂 Loaded test data: ${testSample.length} frames`);

    // Make predictions using trained coefficients
    const testPredictions = kfGom.predAngCoef(normalizedTestData, coef);

    console.log(`🎯 Generated ${testPredictions.length} test predictions`);

    // Calculate test metrics
    const testMetrics = {};
    for (let varIdx = 0; varIdx < targetVariables.length; varIdx++) {
      const variable = targetVariables[varIdx];
      const predictions = testPredictions.map(row => row[varIdx]);
      const actuals = normalizedTestData.slice(kfGom.lags).map(row => row[varIdx]);

      const mse = predictions.reduce((sum, pred, i) => {
        return sum + Math.pow(pred - actuals[i], 2);
      }, 0) / predictions.length;

      testMetrics[variable] = { mse };
    }

    console.log("📊 Test Performance (first 5 variables):");
    Object.entries(testMetrics).slice(0, 5).forEach(([variable, metric]) => {
      console.log(`  ${variable}: MSE = ${metric.mse.toFixed(6)}`);
    });

  } catch (testError) {
    console.log(`⚠️ Test data demo failed: ${testError.message}`);
  }

  // =================================
  // DEMO 4: Model Export/Import
  // =================================
  console.log("\n💾 DEMO 4: Model Export and Import");
  console.log("-".repeat(50));

  // Export model
  const exportedModel = kfGom.export();
  console.log("✅ Model exported successfully");
  console.log(`📊 Exported data contains: ${Object.keys(exportedModel).join(', ')}`);

  // Create new model and import
  const newKfGom = new KfGom();
  newKfGom.import(exportedModel);
  console.log("✅ Model imported to new instance");

  // Test imported model
  const importedPredictions = newKfGom.predAngCoef(normalizedData.slice(0, 20), newKfGom.coef);
  console.log(`🔮 Imported model generated ${importedPredictions.length} predictions`);

} catch (error) {
  console.error(`❌ BVH demo failed: ${error.message}`);
  console.log("💡 Make sure BVH files exist and contain the required joints");
}

// =================================
// COMPARISON: VAR vs SARIMAX
// =================================
console.log("\n🔍 COMPARISON: VAR vs SARIMAX Approaches");
console.log("=".repeat(60));

console.log("📊 VAR (Vector Autoregression) Approach:");
console.log("  ✅ Predicts ALL joints simultaneously");
console.log("  ✅ Captures inter-joint dependencies");
console.log("  ✅ Ensures motion coherence");
console.log("  ⚠️ High computational complexity");
console.log("  ⚠️ Many parameters to estimate");

console.log("\n📊 SARIMAX (Individual Joint) Approach:");
console.log("  ✅ Simple and interpretable");
console.log("  ✅ Fast training and prediction");
console.log("  ✅ Easy to debug and modify");
console.log("  ⚠️ No guarantee of motion coherence");
console.log("  ⚠️ Manual selection of influencing joints");

console.log("\n🎯 When to Use Each:");
console.log("📈 Use VAR when:");
console.log("  • You need coherent full-body motion");
console.log("  • Generating complete motion sequences");
console.log("  • Filling missing motion capture data");
console.log("  • Animation and motion synthesis");

console.log("📈 Use SARIMAX when:");
console.log("  • Analyzing specific joint relationships");
console.log("  • Fast prototyping and testing");
console.log("  • Understanding biomechanical dependencies");
console.log("  • Limited computational resources");

console.log("\n🎉 VAR Demo Complete!");
console.log("💡 The VAR model provides a powerful alternative for full-body motion prediction");

export { VARModel, KfGom }; 