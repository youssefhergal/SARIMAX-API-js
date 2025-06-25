// 🎪 KfGom - Kalman Filter based General Movement Model
// JavaScript equivalent of the Python KfGom class for full-body motion prediction

import { VARModel } from './VARModel.js';
import { extractDataFromBVH } from '../utils/bvhUtils.js';

export class KfGom {
  constructor(variables = null, coefLabels = null) {
    // Default variables - all major body joints with X, Y, Z rotations
    this.variables = variables || [
      'Spine_Xrotation', 'Spine_Yrotation', 'Spine_Zrotation',
      'Spine1_Xrotation', 'Spine1_Yrotation', 'Spine1_Zrotation',
      'Spine2_Xrotation', 'Spine2_Yrotation', 'Spine2_Zrotation',
      'Spine3_Xrotation', 'Spine3_Yrotation', 'Spine3_Zrotation',
      'Hips_Xrotation', 'Hips_Yrotation', 'Hips_Zrotation',
      'Neck_Xrotation', 'Neck_Yrotation', 'Neck_Zrotation',
      'Head_Xrotation', 'Head_Yrotation', 'Head_Zrotation',
      'LeftArm_Xrotation', 'LeftArm_Yrotation', 'LeftArm_Zrotation',
      'LeftForeArm_Xrotation', 'LeftForeArm_Yrotation', 'LeftForeArm_Zrotation',
      'RightArm_Xrotation', 'RightArm_Yrotation', 'RightArm_Zrotation',
      'RightForeArm_Xrotation', 'RightForeArm_Yrotation', 'RightForeArm_Zrotation',
      'LeftShoulder_Xrotation', 'LeftShoulder_Yrotation', 'LeftShoulder_Zrotation',
      'LeftShoulder2_Xrotation', 'LeftShoulder2_Yrotation', 'LeftShoulder2_Zrotation',
      'RightShoulder_Xrotation', 'RightShoulder_Yrotation', 'RightShoulder_Zrotation',
      'RightShoulder2_Xrotation', 'RightShoulder2_Yrotation', 'RightShoulder2_Zrotation',
      'LeftUpLeg_Xrotation', 'LeftUpLeg_Yrotation', 'LeftUpLeg_Zrotation',
      'LeftLeg_Xrotation', 'LeftLeg_Yrotation', 'LeftLeg_Zrotation',
      'RightUpLeg_Xrotation', 'RightUpLeg_Yrotation', 'RightUpLeg_Zrotation',
      'RightLeg_Xrotation', 'RightLeg_Yrotation', 'RightLeg_Zrotation'
    ];

    // Create coefficient labels
    const variables_1 = this.variables.map(v => `${v}(t-1)`);
    const variables_2 = this.variables.map(v => `${v}(t-2)`);
    this.coefLabels = coefLabels || ['Bias', ...variables_1, ...variables_2];

    this.model = null;
    this.coef = null;
    this.pvalues = null;
    this.dfPred = null;
    this.dfY = null;
    this.lags = 2;
    this.trained = false;
  }

  /**
   * Extract angles data from object/dict format to matrix format
   * @param {Object} eulerAngles - Object with joint names as keys and arrays as values
   * @returns {Array} - 2D array [frames, joints]
   */
  extractAnglesMatrix(eulerAngles) {
    const frames = Object.values(eulerAngles)[0].length; // Assume all have same length
    const matrix = [];

    for (let frame = 0; frame < frames; frame++) {
      const row = [];
      for (const variable of this.variables) {
        if (eulerAngles[variable]) {
          row.push(eulerAngles[variable][frame]);
        } else {
          console.warn(`Variable ${variable} not found, using 0`);
          row.push(0);
        }
      }
      matrix.push(row);
    }

    return matrix;
  }

  /**
   * Main training function - equivalent to do_gom in Python
   * @param {Object|Array} eulerAngles - Either object with joint data or 2D array
   * @returns {Object} - {coef: coefficients, dfPred: predictions}
   */
  doGom(eulerAngles) {
    console.log("🎪 Starting VAR-based General Movement Model training...");

    let dataMatrix;
    
    // Handle different input formats
    if (Array.isArray(eulerAngles)) {
      dataMatrix = eulerAngles;
    } else {
      dataMatrix = this.extractAnglesMatrix(eulerAngles);
    }

    console.log(`📊 Data shape: ${dataMatrix.length} frames × ${dataMatrix[0].length} joints`);
    console.log(`🎯 Target variables: ${this.variables.length}`);

    // Create and train VAR model
    this.model = new VARModel(this.lags);
    this.model.fit(dataMatrix, this.variables);

    // Extract coefficients and p-values in the format expected
    const summary = this.model.summary();
    
    // Reshape coefficients to match Python format
    this.coef = this.formatCoefficients(summary.params);
    this.pvalues = this.formatPValues(summary.pvalues);

    // Generate predictions
    const predictions = this.generatePredictions(dataMatrix);
    this.dfPred = predictions.predictions;
    this.dfY = predictions.actuals;

    this.trained = true;

    console.log("✅ VAR-based GOM training complete!");
    console.log(`📈 Generated ${this.dfPred.length} predictions`);

    return { coef: this.coef, dfPred: this.dfPred };
  }

  /**
   * Format coefficients to match Python output structure
   */
  formatCoefficients(params) {
    const result = {};
    
    for (let varIdx = 0; varIdx < this.variables.length; varIdx++) {
      const variable = this.variables[varIdx];
      result[variable] = {};
      
      for (let paramIdx = 0; paramIdx < this.coefLabels.length; paramIdx++) {
        const label = this.coefLabels[paramIdx];
        result[variable][label] = params[paramIdx][varIdx];
      }
    }
    
    return result;
  }

  /**
   * Format p-values to match Python output structure
   */
  formatPValues(pvalues) {
    const result = {};
    
    for (let varIdx = 0; varIdx < this.variables.length; varIdx++) {
      const variable = this.variables[varIdx];
      result[variable] = {};
      
      for (let paramIdx = 0; paramIdx < this.coefLabels.length; paramIdx++) {
        const label = this.coefLabels[paramIdx];
        result[variable][label] = pvalues[varIdx][paramIdx];
      }
    }
    
    return result;
  }

  /**
   * Generate predictions using the trained model
   */
  generatePredictions(dataMatrix) {
    const predictions = [];
    const actuals = [];

    // Start from lag position
    for (let t = this.lags; t < dataMatrix.length - 1; t++) {
      // Get recent data for prediction
      const recentData = dataMatrix.slice(t - this.lags, t);
      
      // Predict next frame
      const pred = this.model.predict(recentData, 1)[0];
      predictions.push(pred);
      
      // Actual value for comparison
      actuals.push(dataMatrix[t]);
    }

    return { predictions, actuals };
  }

  /**
   * Predict using given coefficients (equivalent to pred_ang_coef in Python)
   * @param {Array} datMod - Data matrix for prediction
   * @param {Object} coefMod - Coefficients object
   * @returns {Array} - Predicted values
   */
  predAngCoef(datMod, coefMod) {
    const predictions = [];

    for (let t = this.lags; t < datMod.length; t++) {
      const prediction = [];
      
      for (let varIdx = 0; varIdx < this.variables.length; varIdx++) {
        const variable = this.variables[varIdx];
        const variableCoefs = coefMod[variable];
        
        let pred = variableCoefs['Bias']; // Constant term
        
        // Add lagged terms
        for (let lag = 1; lag <= this.lags; lag++) {
          for (let lagVarIdx = 0; lagVarIdx < this.variables.length; lagVarIdx++) {
            const lagVariable = this.variables[lagVarIdx];
            const lagLabel = `${lagVariable}(t-${lag})`;
            const lagValue = datMod[t - lag][lagVarIdx];
            
            if (variableCoefs[lagLabel]) {
              pred += variableCoefs[lagLabel] * lagValue;
            }
          }
        }
        
        prediction.push(pred);
      }
      
      predictions.push(prediction);
    }

    return predictions;
  }

  /**
   * Load data from BVH file and extract required variables
   * @param {string} bvhPath - Path to BVH file
   * @returns {Array} - Data matrix
   */
  loadFromBVH(bvhPath) {
    console.log(`📂 Loading BVH file: ${bvhPath}`);
    
    try {
      // Extract all available joints
      const bvhData = extractDataFromBVH(bvhPath, this.variables[0], this.variables.slice(1));
      
      // Combine all data
      const dataMatrix = bvhData.endog.map((endogValue, i) => [
        endogValue,
        ...bvhData.exog[i]
      ]);
      
      console.log(`✅ Loaded ${dataMatrix.length} frames from BVH`);
      return dataMatrix;
      
    } catch (error) {
      console.error(`❌ Error loading BVH: ${error.message}`);
      throw error;
    }
  }

  /**
   * Calculate model quality metrics
   */
  calculateMetrics() {
    if (!this.trained || !this.dfPred || !this.dfY) {
      throw new Error("Model not trained or predictions not available");
    }

    const metrics = {};
    
    for (let varIdx = 0; varIdx < this.variables.length; varIdx++) {
      const variable = this.variables[varIdx];
      const predictions = this.dfPred.map(row => row[varIdx]);
      const actuals = this.dfY.map(row => row[varIdx]);
      
      // Calculate MSE
      const mse = predictions.reduce((sum, pred, i) => {
        return sum + Math.pow(pred - actuals[i], 2);
      }, 0) / predictions.length;
      
      // Calculate correlation
      const meanPred = predictions.reduce((sum, val) => sum + val, 0) / predictions.length;
      const meanActual = actuals.reduce((sum, val) => sum + val, 0) / actuals.length;
      
      const numerator = predictions.reduce((sum, pred, i) => {
        return sum + (pred - meanPred) * (actuals[i] - meanActual);
      }, 0);
      
      const denomPred = Math.sqrt(predictions.reduce((sum, pred) => sum + Math.pow(pred - meanPred, 2), 0));
      const denomActual = Math.sqrt(actuals.reduce((sum, actual) => sum + Math.pow(actual - meanActual, 2), 0));
      
      const correlation = denomPred * denomActual > 0 ? numerator / (denomPred * denomActual) : 0;
      
      metrics[variable] = { mse, correlation };
    }
    
    return metrics;
  }

  /**
   * Export model for later use
   */
  export() {
    if (!this.trained) {
      throw new Error("Model not trained");
    }

    return {
      variables: this.variables,
      coefLabels: this.coefLabels,
      coef: this.coef,
      pvalues: this.pvalues,
      lags: this.lags,
      modelParams: this.model.summary()
    };
  }

  /**
   * Import previously trained model
   */
  import(modelData) {
    this.variables = modelData.variables;
    this.coefLabels = modelData.coefLabels;
    this.coef = modelData.coef;
    this.pvalues = modelData.pvalues;
    this.lags = modelData.lags;
    
    // Recreate VAR model
    this.model = new VARModel(this.lags);
    this.model.params = modelData.modelParams.params;
    this.model.pvalues = modelData.modelParams.pvalues;
    this.model.trained = true;
    this.model.variableNames = this.variables;
    this.model.numVariables = this.variables.length;
    
    this.trained = true;
    console.log("✅ Model imported successfully");
  }
} 