// 🎪 VAR Template - Full-Body Motion Prediction
// ✏️ Modify the configuration below for your specific use case

import { KfGom, UPPER_BODY_VARIABLES, CORE_VARIABLES } from '../index.js';
import { extractDataFromBVH } from '../../utils/bvhUtils.js';
import { StandardScaler } from '../../classes/StandardScaler.js';

// =================================
// 🔧 CONFIGURATION - MODIFY HERE
// =================================

// File paths
const TRAINING_FILE = './BVH/Bending/Train_Bending.bvh';     // 👈 Change to your training file
const PREDICTION_FILE = './BVH/Bending/Test_Bending.bvh';   // 👈 Change to your prediction file

// Variables to predict (choose one of the sets below)
const TARGET_VARIABLES = UPPER_BODY_VARIABLES;              // 👈 Change to your variable set
// const TARGET_VARIABLES = CORE_VARIABLES;                 // Smaller set for faster processing
// const TARGET_VARIABLES = FULL_BODY_VARIABLES;            // Complete body (slower)

// Or define your own custom variables:
// const TARGET_VARIABLES = [
//   'Hips_Xrotation', 'Hips_Yrotation', 'Hips_Zrotation',
//   'Spine_Xrotation', 'Spine_Yrotation', 'Spine_Zrotation'
// ];

// Model configuration
const VAR_LAGS = 2;                                          // 👈 Number of past time steps to use
const SAMPLE_SIZE = 500;                                     // 👈 Number of frames to use (for speed)

// =================================
// 🚀 EXECUTION - DON'T MODIFY BELOW
// =================================

// Declare variables for export
let kfGom, scaler, coef, metrics;

console.log("🎪 VAR Template - Full-Body Motion Prediction");
console.log("=" .repeat(50));

console.log(`📂 Training file: ${TRAINING_FILE}`);
console.log(`📂 Prediction file: ${PREDICTION_FILE}`);
console.log(`🎯 Target variables: ${TARGET_VARIABLES.length} joints`);
console.log(`📊 VAR lags: ${VAR_LAGS}`);

// Step 1: Load and prepare training data
console.log("\n1️⃣ Loading and preparing training data...");

try {
  // Load BVH data
  const bvhData = extractDataFromBVH(
    TRAINING_FILE,
    TARGET_VARIABLES[0],
    TARGET_VARIABLES.slice(1)
  );

  // Prepare data matrix
  const motionData = bvhData.endog.map((endogValue, i) => [
    endogValue,
    ...bvhData.exog[i]
  ]);

  // Take subset for faster processing
  const sampleSize = Math.min(SAMPLE_SIZE, motionData.length);
  const sampledData = motionData.slice(0, sampleSize);
  
  console.log(`✅ Loaded ${motionData.length} frames, using ${sampleSize} for training`);

  // Normalize data
  const scaler = new StandardScaler();
  const normalizedData = scaler.fitTransform(sampledData);

  // Step 2: Train VAR model
  console.log("\n2️⃣ Training VAR model...");

  const kfGom = new KfGom(TARGET_VARIABLES);
  const { coef, dfPred } = kfGom.doGom(normalizedData);

  console.log(`✅ Model trained on ${TARGET_VARIABLES.length} variables`);
  console.log(`📈 Generated ${dfPred.length} training predictions`);

  // Step 3: Evaluate training performance
  console.log("\n3️⃣ Evaluating training performance...");

  const metrics = kfGom.calculateMetrics();
  
  console.log("📊 Training Performance (all variables):");
  let totalMSE = 0;
  let totalCorr = 0;
  let validCount = 0;

  Object.entries(metrics).forEach(([variable, metric]) => {
    if (isFinite(metric.mse) && isFinite(metric.correlation)) {
      totalMSE += metric.mse;
      totalCorr += metric.correlation;
      validCount++;
    }
    console.log(`  ${variable}: MSE=${metric.mse.toFixed(6)}, Corr=${metric.correlation.toFixed(4)}`);
  });

  const avgMSE = totalMSE / validCount;
  const avgCorr = totalCorr / validCount;
  
  console.log(`\n📈 Average Performance:`);
  console.log(`  MSE: ${avgMSE.toFixed(6)}`);
  console.log(`  Correlation: ${avgCorr.toFixed(4)}`);

  // Step 4: Test on new data
  console.log("\n4️⃣ Testing on new data...");

  try {
    // Load test data
    const testBvhData = extractDataFromBVH(
      PREDICTION_FILE,
      TARGET_VARIABLES[0],
      TARGET_VARIABLES.slice(1)
    );

    const testMotionData = testBvhData.endog.map((endogValue, i) => [
      endogValue,
      ...testBvhData.exog[i]
    ]);

    // Take subset and normalize with same scaler
    const testSample = testMotionData.slice(0, Math.min(200, testMotionData.length));
    const normalizedTestData = scaler.transform(testSample);

    console.log(`📂 Loaded test data: ${testSample.length} frames`);

    // Make predictions
    const testPredictions = kfGom.predAngCoef(normalizedTestData, coef);
    console.log(`🎯 Generated ${testPredictions.length} test predictions`);

    // Calculate test metrics
    const testMetrics = {};
    let testTotalMSE = 0;
    let testValidCount = 0;

    for (let varIdx = 0; varIdx < TARGET_VARIABLES.length; varIdx++) {
      const variable = TARGET_VARIABLES[varIdx];
      const predictions = testPredictions.map(row => row[varIdx]);
      const actuals = normalizedTestData.slice(kfGom.lags).map(row => row[varIdx]);

      if (predictions.length > 0 && actuals.length > 0) {
        const mse = predictions.reduce((sum, pred, i) => {
          return sum + Math.pow(pred - actuals[i], 2);
        }, 0) / predictions.length;

        testMetrics[variable] = { mse };
        
        if (isFinite(mse)) {
          testTotalMSE += mse;
          testValidCount++;
        }
      }
    }

    const avgTestMSE = testTotalMSE / testValidCount;

    console.log("📊 Test Performance Summary:");
    console.log(`  Average MSE: ${avgTestMSE.toFixed(6)}`);
    
    if (avgTestMSE < avgMSE * 2) {
      console.log("✅ Good generalization (test MSE < 2x training MSE)");
    } else {
      console.log("⚠️ Poor generalization - consider more training data or different variables");
    }

  } catch (testError) {
    console.log(`⚠️ Test data failed: ${testError.message}`);
  }

  // Step 5: Save model (optional)
  console.log("\n5️⃣ Model export...");

  const exportedModel = kfGom.export();
  console.log("✅ Model can be exported for later use");
  console.log(`📊 Model contains ${Object.keys(exportedModel.coef).length} variable equations`);

  // Step 6: Results summary
  console.log("\n🎉 VAR Analysis Complete!");
  console.log(`✅ Trained on ${TARGET_VARIABLES.length} joint variables`);
  console.log(`✅ Average training correlation: ${avgCorr.toFixed(4)}`);
  
  if (avgCorr > 0.8) {
    console.log("🎯 Excellent motion prediction quality!");
  } else if (avgCorr > 0.6) {
    console.log("👍 Good motion prediction quality");
  } else {
    console.log("⚠️ Consider using different variables or more training data");
  }

} catch (error) {
  console.error(`❌ VAR analysis failed: ${error.message}`);
  console.log("💡 Make sure BVH files exist and contain the required joints");
  console.log("💡 Try using a smaller variable set (CORE_VARIABLES) for testing");
}

// Export for further use (will be undefined if training failed)
export { kfGom, scaler, coef, metrics };

// =================================
// 📋 USAGE INSTRUCTIONS
// =================================

/*
🔧 How to use this template:

1. Modify the configuration section above:
   - Change TRAINING_FILE and PREDICTION_FILE paths
   - Choose TARGET_VARIABLES (CORE_VARIABLES for testing, UPPER_BODY_VARIABLES for more detail)
   - Adjust VAR_LAGS (2 is usually good)
   - Adjust SAMPLE_SIZE based on your computational resources

2. Run the script:
   node VAR/templates/var_template.js

3. Check the console output for:
   - Training quality (correlations, MSE)
   - Test performance
   - Model export status

4. Understand the results:
   - Correlation > 0.8 = Excellent
   - Correlation 0.6-0.8 = Good  
   - Correlation < 0.6 = Poor (try different variables)

⚠️ Important notes:
   - VAR models work best with coherent motion data
   - More variables = more complex model (slower training)
   - Start with CORE_VARIABLES or UPPER_BODY_VARIABLES for testing
   - Full-body models require significant computational resources
*/ 