// 🎪 Vector Autoregression (VAR) Model Implementation
// Predicts ALL variables simultaneously using their past values

import * as math from 'mathjs';

export class VARModel {
  constructor(lags = 2) {
    this.lags = lags;
    this.params = null;
    this.pvalues = null;
    this.trained = false;
    this.variableNames = null;
    this.numVariables = null;
    this.residuals = null;
    this.fitted = null;
  }

  /**
   * Create lagged matrix for VAR model
   * @param {Array} data - 2D array [observations, variables]
   * @param {number} lags - Number of lags
   * @returns {Object} - {X: design matrix, y: target matrix}
   */
  createLaggedMatrix(data, lags) {
    const numObs = data.length;
    const numVars = data[0].length;
    
    // Each observation uses 'lags' previous observations
    const effectiveObs = numObs - lags;
    
    // Design matrix: [constant, var1(t-1), var2(t-1), ..., varN(t-1), var1(t-2), ..., varN(t-2)]
    const numFeatures = 1 + (numVars * lags); // 1 for constant + lags * variables
    
    const X = [];
    const y = [];
    
    for (let t = lags; t < numObs; t++) {
      // Create row for observation t
      const row = [1]; // Constant term
      
      // Add lagged values: t-1, t-2, ..., t-lags
      for (let lag = 1; lag <= lags; lag++) {
        for (let var_idx = 0; var_idx < numVars; var_idx++) {
          row.push(data[t - lag][var_idx]);
        }
      }
      
      X.push(row);
      y.push([...data[t]]); // Target: all variables at time t
    }
    
    return { X, y };
  }

  /**
   * Add noise to constant columns to avoid singular matrix
   * @param {Array} data - 2D array
   * @returns {Array} - Data with noise added to constant columns
   */
  addNoiseToConstantColumns(data) {
    const result = data.map(row => [...row]); // Deep copy
    const numVars = data[0].length;
    
    // Check each variable for constant values
    for (let varIdx = 0; varIdx < numVars; varIdx++) {
      const values = data.map(row => row[varIdx]);
      const uniqueValues = [...new Set(values)];
      
      if (uniqueValues.length <= 1) {
        console.warn(`Adding noise to constant variable at index ${varIdx}`);
        // Add small random noise
        for (let obsIdx = 0; obsIdx < data.length; obsIdx++) {
          result[obsIdx][varIdx] += (Math.random() - 0.5) * 0.001;
        }
      }
    }
    
    return result;
  }

  /**
   * Fit VAR model using OLS equation by equation
   * @param {Array} data - 2D array [observations, variables]
   * @param {Array} variableNames - Names of variables
   */
  fit(data, variableNames = null) {
    console.log(`🎪 Training VAR(${this.lags}) model...`);
    
    this.numVariables = data[0].length;
    this.variableNames = variableNames || Array.from({length: this.numVariables}, (_, i) => `Var_${i}`);
    
    console.log(`📊 Variables: ${this.numVariables}`);
    console.log(`📈 Observations: ${data.length}`);
    
    // Add noise to constant columns
    const processedData = this.addNoiseToConstantColumns(data);
    
    // Create lagged matrices
    const { X, y } = this.createLaggedMatrix(processedData, this.lags);
    
    console.log(`🔧 Design matrix: ${X.length} × ${X[0].length}`);
    console.log(`🎯 Target matrix: ${y.length} × ${y[0].length}`);
    
    // Convert to matrices
    const XMatrix = math.matrix(X);
    const yMatrix = math.matrix(y);
    
    // Calculate coefficients for each equation: β = (X'X)^(-1)X'y
    const XT = math.transpose(XMatrix);
    const XTX = math.multiply(XT, XMatrix);
    
    // Add regularization for numerical stability
    const lambda = 1e-8;
    const identity = math.identity(XTX.size());
    const regularizedXTX = math.add(XTX, math.multiply(lambda, identity));
    
    const XTy = math.multiply(XT, yMatrix);
    const beta = math.multiply(math.inv(regularizedXTX), XTy);
    
    this.params = beta._data;
    
    // Calculate fitted values and residuals
    const yPred = math.multiply(XMatrix, beta);
    this.fitted = yPred._data;
    this.residuals = math.subtract(yMatrix, yPred)._data;
    
    // Calculate standard errors and p-values
    this.calculateStatistics(XMatrix, yMatrix, beta);
    
    this.trained = true;
    console.log(`✅ VAR model trained successfully!`);
    
    return this;
  }

  /**
   * Calculate standard errors, t-statistics, and p-values
   */
  calculateStatistics(XMatrix, yMatrix, beta) {
    const n = XMatrix.size()[0]; // Number of observations
    const k = XMatrix.size()[1]; // Number of parameters per equation
    const m = this.numVariables;   // Number of equations (variables)
    
    // Calculate residual covariance matrix
    const residuals = math.subtract(yMatrix, math.multiply(XMatrix, beta));
    const residualsT = math.transpose(residuals);
    const sigmaMat = math.multiply(residualsT, residuals);
    const sigma = math.divide(sigmaMat, n - k);
    
    // Calculate parameter covariance matrix for each equation
    const XTX_inv = math.inv(math.multiply(math.transpose(XMatrix), XMatrix));
    
    this.pvalues = [];
    
    for (let eqIdx = 0; eqIdx < m; eqIdx++) {
      const sigma_eq = sigma._data[eqIdx][eqIdx];
      const paramCov = math.multiply(sigma_eq, XTX_inv);
      const stdErrors = math.diag(paramCov).map(val => Math.sqrt(Math.abs(val)));
      
      const tStats = this.params.map((paramRow, paramIdx) => {
        const param = paramRow[eqIdx];
        const se = stdErrors[paramIdx];
        return se > 0 ? param / se : 0;
      });
      
      const pVals = tStats.map(t => {
        const absT = Math.abs(t);
        if (!isFinite(absT) || isNaN(absT)) return 0.999;
        
        // Approximate p-value using t-distribution
        const df = n - k;
        if (df > 30) {
          // Normal approximation for large df
          const z = absT;
          const pValue = 2 * (1 - this.normalCDF(z));
          return Math.max(0.001, Math.min(0.999, pValue));
        } else {
          // Simple approximation for small df
          if (absT > 4) return 0.001;
          else if (absT > 3) return 0.01;
          else if (absT > 2.5) return 0.02;
          else if (absT > 2) return 0.05;
          else if (absT > 1.5) return 0.1;
          else return 0.2;
        }
      });
      
      this.pvalues.push(pVals);
    }
  }

  /**
   * Normal CDF approximation
   */
  normalCDF(x) {
    const erfApprox = (x) => {
      const a1 =  0.254829592;
      const a2 = -0.284496736;
      const a3 =  1.421413741;
      const a4 = -1.453152027;
      const a5 =  1.061405429;
      const p  =  0.3275911;
      const sign = x < 0 ? -1 : 1;
      x = Math.abs(x);
      const t = 1.0 / (1.0 + p * x);
      const y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * Math.exp(-x * x);
      return sign * y;
    };
    return 0.5 * (1 + erfApprox(x / Math.sqrt(2)));
  }

  /**
   * Predict future values
   * @param {Array} data - Recent observations for prediction
   * @param {number} steps - Number of steps to forecast
   * @returns {Array} - Predicted values
   */
  predict(data, steps = 1) {
    if (!this.trained) {
      throw new Error("Model not trained. Call fit() first.");
    }
    
    const recentData = data.slice(-this.lags); // Get last 'lags' observations
    const predictions = [];
    
    // Use the most recent data for prediction
    let currentData = [...recentData];
    
    for (let step = 0; step < steps; step++) {
      // Create input vector: [1, var1(t-1), var2(t-1), ..., varN(t-1), var1(t-2), ..., varN(t-2)]
      const inputVector = [1]; // Constant
      
      for (let lag = 0; lag < this.lags; lag++) {
        const laggedObs = currentData[currentData.length - 1 - lag];
        inputVector.push(...laggedObs);
      }
      
      // Calculate prediction for all variables
      const prediction = [];
      for (let varIdx = 0; varIdx < this.numVariables; varIdx++) {
        let pred = 0;
        for (let paramIdx = 0; paramIdx < inputVector.length; paramIdx++) {
          pred += this.params[paramIdx][varIdx] * inputVector[paramIdx];
        }
        prediction.push(pred);
      }
      
      predictions.push(prediction);
      
      // Add prediction to current data for next step
      currentData.push(prediction);
      if (currentData.length > this.lags) {
        currentData.shift(); // Keep only last 'lags' observations
      }
    }
    
    return predictions;
  }

  /**
   * Get model summary
   */
  summary() {
    if (!this.trained) return "Model not trained.";
    
    return {
      lags: this.lags,
      numVariables: this.numVariables,
      numParameters: this.params.length,
      variableNames: this.variableNames,
      params: this.params,
      pvalues: this.pvalues,
      residuals: this.residuals,
      fitted: this.fitted
    };
  }
} 