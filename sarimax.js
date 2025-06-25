// SARIMAX-like multivariate time series model in JavaScript
// State space representation and forecasting of joint angles
// Dependencies: mathjs, jstat, bvh-parser

import * as math from 'mathjs';
import pkg from 'jstat';
const { studentt } = pkg;
import fs from 'fs';
import bvh from 'bvh-parser';
import plotly from 'plotly';

// MinMaxScaler equivalent for JavaScript
class MinMaxScaler {
  constructor() {
    this.min = null;
    this.max = null;
    this.scale = null;
    this.minScale = null;
  }

  fit(data) {
    const matrix = math.matrix(data);
    this.min = math.min(matrix, 0)._data;
    this.max = math.max(matrix, 0)._data;
    this.scale = this.max.map((max, i) => max - this.min[i]);
    this.minScale = this.min;
    return this;
  }

  transform(data) {
    if (!this.scale) throw new Error("Scaler not fitted");
    return data.map(row => 
      row.map((val, i) => (val - this.minScale[i]) / this.scale[i])
    );
  }

  fitTransform(data) {
    return this.fit(data).transform(data);
  }

  inverseTransform(data) {
    if (!this.scale) throw new Error("Scaler not fitted");
    return data.map(row => 
      row.map((val, i) => val * this.scale[i] + this.minScale[i])
    );
  }

  inverseTransformSingle(values) {
    if (!this.scale) throw new Error("Scaler not fitted");
    return values.map((val, i) => val * this.scale[i] + this.minScale[i]);
  }
}

// StandardScaler equivalent for JavaScript (like sklearn.preprocessing.StandardScaler)
class StandardScaler {
  constructor() {
    this.mean = null;
    this.std = null;
  }

  fit(data) {
    const matrix = math.matrix(data);
    this.mean = math.mean(matrix, 0)._data;
    
    // Calculate standard deviation
    const numCols = data[0].length;
    this.std = new Array(numCols);
    
    for (let col = 0; col < numCols; col++) {
      const columnData = data.map(row => row[col]);
      const meanVal = this.mean[col];
      const variance = columnData.reduce((sum, val) => sum + Math.pow(val - meanVal, 2), 0) / columnData.length;
      this.std[col] = Math.sqrt(variance);
      // Avoid division by zero
      if (this.std[col] === 0) this.std[col] = 1;
    }
    
    return this;
  }

  transform(data) {
    if (!this.mean || !this.std) throw new Error("Scaler not fitted");
    return data.map(row => 
      row.map((val, i) => (val - this.mean[i]) / this.std[i])
    );
  }

  fitTransform(data) {
    return this.fit(data).transform(data);
  }

  inverseTransform(data) {
    if (!this.mean || !this.std) throw new Error("Scaler not fitted");
    return data.map(row => 
      row.map((val, i) => val * this.std[i] + this.mean[i])
    );
  }

  inverseTransformSingle(values) {
    if (!this.mean || !this.std) throw new Error("Scaler not fitted");
    return values.map((val, i) => val * this.std[i] + this.mean[i]);
  }
}

// Enhanced SARIMAX class
class SARIMAX {
  constructor(endog, exog, order = 2) {
    this.endog = endog;
    this.exog = exog;
    this.order = order;
    this.coefficients = null;
    this.trained = false;
    this.stdErrors = null;
    this.tStats = null;
    this.pValues = null;
    this.residuals = null;
    this.rSquared = null;
    this.mse = null;
    this.aic = null;
    this.bic = null;
  }

  laggedMatrix(data, lags) {
    const result = [];
    for (let i = lags; i < data.length; i++) {
      const row = [];
      for (let j = 1; j <= lags; j++) {
        row.push(data[i - j]);
      }
      result.push(row);
    }
    return result;
  }

  fit() {
    const X = [];
    const y = [];

    const laggedEndog = this.laggedMatrix(this.endog, this.order);
    const laggedExog = this.exog.slice(this.order);

    for (let i = 0; i < laggedEndog.length; i++) {
      X.push([...laggedExog[i], ...laggedEndog[i]]);
      y.push(this.endog[i + this.order]);
    }

    const XMatrix = math.matrix(X);
    const yVector = math.matrix(y);

    const XT = math.transpose(XMatrix);
    const XTX = math.multiply(XT, XMatrix);
    
    // Add regularization to prevent numerical instability
    const lambda = 1e-6; // Small regularization parameter
    const identity = math.identity(XTX.size());
    const regularizedXTX = math.add(XTX, math.multiply(lambda, identity));
    
    const XTY = math.multiply(XT, yVector);
    const beta = math.multiply(math.inv(regularizedXTX), XTY);

    this.coefficients = beta._data;
    
    // Check for potential instability in AR coefficients
    const numExog = this.exog[0].length;
    const arCoeffs = this.coefficients.slice(numExog);
    const arSum = arCoeffs.reduce((sum, coef) => sum + coef, 0);
    
    if (Math.abs(arSum) > 0.999) {
      console.warn(`⚠️ Model stability warning: AR coefficients sum = ${arSum.toFixed(6)} (close to unit root)`);
      // Apply stability correction
      const stabilityFactor = 0.995 / Math.abs(arSum);
      for (let i = numExog; i < this.coefficients.length; i++) {
        this.coefficients[i] *= stabilityFactor;
      }
      console.log(`✅ Applied stability correction factor: ${stabilityFactor.toFixed(6)}`);
    }
    
    // Recalculate predictions with potentially corrected coefficients
    const correctedBeta = math.matrix(this.coefficients);
    const yPred = math.multiply(XMatrix, correctedBeta);
    const residuals = math.subtract(yVector, yPred);

    const n = y.length;
    const k = this.coefficients.length;
    const sse = math.sum(math.dotMultiply(residuals, residuals));
    const sigma2 = sse / (n - k);

    const covMatrix = math.multiply(sigma2, math.inv(XTX));
    const diagElements = math.diag(covMatrix);
    
    // Convert to regular array if needed
    const diagArray = Array.isArray(diagElements) ? diagElements : diagElements._data || [diagElements];
    
    const stdErrors = diagArray.map(val => {
      const sqrt = Math.sqrt(Math.abs(val)); // Ensure positive value
      return sqrt === 0 ? 1e-10 : sqrt; // Avoid division by zero
    });
    
    const tStats = this.coefficients.map((b, i) => {
      const tStat = b / stdErrors[i];
      return isNaN(tStat) || !isFinite(tStat) ? 0 : tStat;
    });
    
    const pValues = tStats.map(t => {
      try {
        const absT = Math.abs(t);
        if (!isFinite(absT) || isNaN(absT)) return 0.999;
        
        // Manual t-distribution approximation for p-values
        // Using normal approximation for large degrees of freedom
        const df = n - k;
        let pValue;
        
        if (df > 30) {
          // Normal approximation for large df
          const z = absT;
          // Approximate standard normal CDF
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
          const normalCdf = (x) => 0.5 * (1 + erfApprox(x / Math.sqrt(2)));
          pValue = 2 * (1 - normalCdf(z));
        } else {
          // Simple approximation for small df
          if (absT > 4) pValue = 0.001;
          else if (absT > 3) pValue = 0.01;
          else if (absT > 2.5) pValue = 0.02;
          else if (absT > 2) pValue = 0.05;
          else if (absT > 1.5) pValue = 0.1;
          else pValue = 0.2;
        }
        
        return Math.max(0.001, Math.min(0.999, pValue));
        
      } catch (e) {
        console.log('Error calculating p-value for t-stat:', t, e.message);
        return 0.999;
      }
          });

    const meanY = math.mean(y);
    const ssTotal = math.sum(y.map(v => Math.pow(v - meanY, 2)));
    const rSquared = 1 - (sse / ssTotal);

    // Calculate AIC and BIC
    this.aic = 2 * k - 2 * Math.log(sse / n);
    this.bic = k * Math.log(n) - 2 * Math.log(sse / n);

    this.trained = true;
    this.stdErrors = stdErrors;
    this.tStats = tStats;
    this.pValues = pValues;
    this.residuals = residuals._data;
    this.rSquared = rSquared;
    this.mse = sigma2;

    return this;
  }

  apply(endogData, exogData) {
    if (!this.trained) throw new Error("Model not trained");
    
    // Create a temporary model instance for prediction
    const tempModel = {
      endog: endogData,
      exog: exogData,
      coefficients: this.coefficients,
      order: this.order,
      getPrediction: () => {
        const lastEndog = endogData.slice(-this.order);
        const lastExog = exogData[exogData.length - 1];
        const input = [...lastExog, ...lastEndog];
        const prediction = math.dot(input, this.coefficients);
        return {
          predicted_mean: [prediction]
        };
      }
    };
    
    return tempModel;
  }

  predictNext(lastEndog, nextExog) {
    if (!this.trained) throw new Error("Model not trained");
    if (lastEndog.length !== this.order || nextExog.length !== this.exog[0].length)
      throw new Error("Mismatch in input dimensions");

    const input = [...nextExog, ...lastEndog];
    const prediction = math.dot(input, this.coefficients);
    return prediction;
  }

  summary() {
    if (!this.trained) return "Model not trained.";
    return {
      coefficients: this.coefficients,
      stdErrors: this.stdErrors,
      tStats: this.tStats,
      pValues: this.pValues,
      residuals: this.residuals,
      mse: this.mse,
      rSquared: this.rSquared,
      aic: this.aic,
      bic: this.bic
    };
  }
}

// BVH data extraction function with proper header parsing
function extractDataFromBVH(path, targetJoint, exogJoints) {
  try {
    console.log(`📂 Loading BVH file: ${path}`);
    const bvhContent = fs.readFileSync(path, 'utf-8');
    
    // Parse BVH header to get proper channel names (same logic as inspector)
    const headerInfo = parseBVHHeader(bvhContent);
    console.log(`✅ Parsed header: ${headerInfo.channels.length} channels found`);
    
    // Parse BVH data
    const result = bvh(bvhContent);
    const frames = result?.frames;
    
    if (!frames || frames.length === 0) {
      throw new Error('No frames found in BVH file');
    }
    
    console.log(`📊 Found ${frames.length} frames`);
    console.log(`🎯 Looking for target joint: ${targetJoint}`);
    console.log(`🔗 Looking for exogenous joints: ${exogJoints.join(', ')}`);
    
    // Find target joint index
    const targetIndex = headerInfo.channels.findIndex(ch => ch === targetJoint);
    if (targetIndex === -1) {
      console.log('❌ Available channels:', headerInfo.channels.slice(0, 20));
      throw new Error(`Target joint "${targetJoint}" not found`);
    }
    
    // Find exogenous joint indices
    const exogIndices = exogJoints.map(joint => {
      const index = headerInfo.channels.findIndex(ch => ch === joint);
      if (index === -1) {
        console.warn(`⚠️ Exogenous joint "${joint}" not found, using zeros`);
        return -1;
      }
      return index;
    });
    
    console.log(`✅ Target "${targetJoint}" found at index: ${targetIndex}`);
    console.log(`✅ Exogenous joints indices:`, exogIndices);
    
    // Extract data
    const endog = frames.map(frame => frame[targetIndex]);
    const exog = frames.map(frame => 
      exogIndices.map(idx => idx === -1 ? 0 : frame[idx])
    );
    
    // Verify data quality
    const validEndog = endog.filter(val => !isNaN(val) && isFinite(val));
    const validExog = exog.filter(row => row.every(val => !isNaN(val) && isFinite(val)));
    
    if (validEndog.length !== frames.length || validExog.length !== frames.length) {
      console.warn(`⚠️ Some invalid data found. Valid endog: ${validEndog.length}/${frames.length}`);
    }
    
    console.log(`✅ Successfully extracted:
      📈 Endogenous values: ${endog.length} frames
      📊 Exogenous values: ${exog.length} frames x ${exog[0].length} variables
      📋 Sample endogenous values: [${endog.slice(0, 5).map(v => v.toFixed(3)).join(', ')}...]
      📋 Sample exogenous values: [${exog[0].map(v => v.toFixed(3)).join(', ')}]`);
    
    return { endog, exog };
    
  } catch (error) {
    console.error('❌ Error loading BVH file:', error.message);
    console.log('⚠️ THIS SHOULD NOT HAPPEN - BVH parsing failed!');
    throw error; // Don't fall back to synthetic data, throw the error instead
  }
}

// Parse BVH header to extract channel names (from bvh_inspector.js)
function parseBVHHeader(bvhContent) {
  const lines = bvhContent.split('\n');
  const channels = [];
  
  let currentJoint = null;
  let inMotionSection = false;
  
  for (let i = 0; i < lines.length; i++) {
    const line = lines[i].trim();
    
    if (line === 'MOTION') {
      inMotionSection = true;
      break;
    }
    
    if (line.startsWith('ROOT ') || line.startsWith('JOINT ')) {
      const parts = line.split(/\s+/);
      currentJoint = parts[1];
    } else if (line.startsWith('CHANNELS ')) {
      const parts = line.split(/\s+/);
      const numChannels = parseInt(parts[1]);
      const channelTypes = parts.slice(2, 2 + numChannels);
      
      channelTypes.forEach(channelType => {
        channels.push(`${currentJoint}_${channelType}`);
      });
    }
  }
  
  return { channels };
}

// Get all rotation columns from BVH data
function getRotationColumns(bvhData) {
  const result = bvh(bvhData);
  const jointNames = result?.motionChannels?.map(ch => ch.name) || [];
  return jointNames.filter(name => name.includes('rotation'));
}

// Extract Euler angles from BVH
function extractEulerAngles(bvhPath) {
  const bvhContent = fs.readFileSync(bvhPath, 'utf-8');
  const result = bvh(bvhContent);
  const frames = result?.frames;
  const jointNames = result?.motionChannels?.map(ch => ch.name);
  
  const rotationCols = jointNames.filter(name => name.includes('rotation'));
  const jointMap = {};
  jointNames.forEach((name, i) => { jointMap[name] = i; });

  const eulerAngles = {};
  rotationCols.forEach(col => {
    eulerAngles[col] = frames.map(frame => frame[jointMap[col]]);
  });

  return eulerAngles;
}

// Evaluation metrics
function MSE(yTrue, yPred) {
  const errors = yTrue.map((val, i) => Math.pow(val - yPred[i], 2));
  return errors.reduce((sum, err) => sum + err, 0) / errors.length;
}

function MAE(yTrue, yPred) {
  const errors = yTrue.map((val, i) => Math.abs(val - yPred[i]));
  return errors.reduce((sum, err) => sum + err, 0) / errors.length;
}

function UTheil(yTrue, yPred) {
  const errors = yTrue.map((val, i) => val - yPred[i]);
  const mseError = errors.reduce((sum, err) => sum + Math.pow(err, 2), 0) / errors.length;
  const msePred = yPred.reduce((sum, val) => sum + Math.pow(val, 2), 0) / yPred.length;
  const mseTrue = yTrue.reduce((sum, val) => sum + Math.pow(val, 2), 0) / yTrue.length;
  
  return Math.sqrt(mseError) / (Math.sqrt(msePred) + Math.sqrt(mseTrue));
}

// Static forecasting function
function staticForecasting(model, testData, indEnd, indExo, scaler, targetAngleIndex) {
  const nob = testData.length;
  const endoData = testData.map(row => row[indEnd]);
  const exogData = testData.map(row => indExo.map(idx => row[idx]));
  
  const predStatic = [];
  const origValues = [];

  // Use all available data: start from order and predict until end
  for (let i = model.order; i < nob; i++) {
    const forecast = model.apply(
      endoData.slice(i - model.order, i), 
      exogData.slice(i - model.order, i)
    );
    const pred = forecast.getPrediction();
    const predMean = pred.predicted_mean[0];
    
    predStatic.push(predMean);
    origValues.push(endoData[i]); // Current actual value
  }

  console.log(`📊 Static forecasting: Generated ${predStatic.length} predictions from ${nob} total frames`);

  // Denormalize the data (StandardScaler inverse transform)
  const denormalizedPred = predStatic.map(val => val * scaler.std[targetAngleIndex] + scaler.mean[targetAngleIndex]);
  const denormalizedOrig = origValues.map(val => val * scaler.std[targetAngleIndex] + scaler.mean[targetAngleIndex]);

  return { predStatic: denormalizedPred, origValues: denormalizedOrig };
}

// Dynamic forecasting function
function dynamicForecasting(model, testData, indEnd, indExo, scaler, targetAngleIndex) {
  const nob = testData.length;
  const endoData = testData.map(row => row[indEnd]);
  const exogData = testData.map(row => indExo.map(idx => row[idx]));
  
  // Initialize with real first values
  const predDynamic = [...endoData.slice(0, model.order)];
  const origValues = [...endoData];

  // Dynamic prediction: use previous predictions
  for (let i = model.order; i < nob; i++) {
    const forecast = model.apply(
      predDynamic.slice(i - model.order, i), 
      exogData.slice(i - model.order, i)
    );
    const pred = forecast.getPrediction();
    const predMean = pred.predicted_mean[0];
    
    predDynamic.push(predMean);
  }

  console.log(`📊 Dynamic forecasting: Generated ${predDynamic.length} predictions from ${nob} total frames`);

  // Denormalize the data (StandardScaler inverse transform)
  const denormalizedPred = predDynamic.map(val => val * scaler.std[targetAngleIndex] + scaler.mean[targetAngleIndex]);
  const denormalizedOrig = origValues.map(val => val * scaler.std[targetAngleIndex] + scaler.mean[targetAngleIndex]);

  // For dynamic forecasting, use data from 2nd sample onwards (like Python: y = OrigValues_G[2:])
  const predFromSecond = denormalizedPred.slice(2);
  const origFromSecond = denormalizedOrig.slice(2);

  return { 
    predDynamic: predFromSecond, 
    origValues: origFromSecond,
    fullPredDynamic: denormalizedPred,
    fullOrigValues: denormalizedOrig
  };
}

// Create model summary table
function createModelSummary(model, angles, targetAngle, indExo) {
  const summary = model.summary();
  // Variables for the model: exogenous variables + lagged endogenous variables
  const variables = [...indExo.map(i => angles[i]), `${targetAngle}_T-1`, `${targetAngle}_T-2`];
  
  console.log('\n=== MODEL SUMMARY ===');
  console.log('Variables:', variables);
  console.log('Coefficients:', summary.coefficients);
  console.log('P-values:', summary.pValues);
  console.log('R-squared:', summary.rSquared);
  console.log('MSE:', summary.mse);
  console.log('AIC:', summary.aic);
  console.log('BIC:', summary.bic);
  console.log('Variables length:', variables.length);
  console.log('Coefficients length:', summary.coefficients.length);
  console.log('P-values length:', summary.pValues.length);
  
  return {
    variables,
    coefficients: summary.coefficients,
    pValues: summary.pValues,
    rSquared: summary.rSquared,
    mse: summary.mse,
    aic: summary.aic,
    bic: summary.bic
  };
}

// Generate HTML plot with Plotly
function createPlot(originalData, predictedData, title, filename, forecastType = 'Static', modelSummary = null) {
  const timeFrames = Array.from({length: originalData.length}, (_, i) => i);
  
  // Calculate confidence intervals using Python-like method
  // ci = (1-(alpha/2)) * np.std(y)/np.mean(y)
  const alpha = 0.05; // 95% confidence level
  const meanY = originalData.reduce((sum, val) => sum + val, 0) / originalData.length;
  const stdY = Math.sqrt(originalData.reduce((sum, val) => sum + Math.pow(val - meanY, 2), 0) / originalData.length);
  
  // Python-like confidence interval calculation
  const ci = (1 - (alpha / 2)) * (stdY / Math.abs(meanY));
  
  // Create confidence band around original data (like Python code)
  const upperBound = originalData.map(val => val + ci);
  const lowerBound = originalData.map(val => val - ci);
  
  // Generate model summary table HTML
  let modelTableHtml = '';
  if (modelSummary) {
    modelTableHtml = `
    <div class="model-summary">
        <h3>🧮 Model Summary</h3>
        <div class="table-container">
            <table class="summary-table">
                <thead>
                    <tr>
                        <th>Variable</th>
                        <th>Coefficient</th>
                        <th>P-value</th>
                        <th>Significance</th>
                    </tr>
                </thead>
                <tbody>
                    ${modelSummary.variables.map((variable, i) => {
                        const coeff = modelSummary.coefficients[i];
                        const pVal = modelSummary.pValues[i];
                        if (coeff === undefined || pVal === undefined) {
                            console.log(`Warning: Missing data for variable ${i}: ${variable}`);
                            return '';
                        }
                        return `
                        <tr class="${pVal < 0.05 ? 'significant' : ''}">
                            <td class="variable-name">${variable}</td>
                            <td class="coefficient">${coeff.toFixed(6)}</td>
                            <td class="p-value">${pVal.toFixed(6)}</td>
                            <td class="significance">${pVal < 0.001 ? '***' : pVal < 0.01 ? '**' : pVal < 0.05 ? '*' : ''}</td>
                        </tr>`;
                    }).join('')}
                </tbody>
            </table>
        </div>
        <div class="model-stats">
            <p><strong>R-squared:</strong> ${modelSummary.rSquared?.toFixed(6) || 'N/A'}</p>
            <p><strong>MSE:</strong> ${modelSummary.mse?.toFixed(6) || 'N/A'}</p>
            <p><strong>AIC:</strong> ${modelSummary.aic?.toFixed(6) || 'N/A'}</p>
            <p><strong>BIC:</strong> ${modelSummary.bic?.toFixed(6) || 'N/A'}</p>
        </div>
        <div class="significance-legend">
            <p><strong>Significance codes:</strong> 0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1</p>
        </div>
    </div>`;
  }
  
  const htmlContent = `
<!DOCTYPE html>
<html>
<head>
    <title>${title}</title>
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; }
        .plot-container { width: 100%; height: 600px; }
        .metrics { 
            background: #f5f5f5; 
            padding: 15px; 
            border-radius: 5px; 
            margin-bottom: 20px;
            font-family: monospace;
        }
        .model-summary {
            background: #fafafa;
            padding: 20px;
            border-radius: 8px;
            margin-bottom: 20px;
            border: 1px solid #ddd;
        }
        .table-container {
            overflow-x: auto;
            margin-bottom: 15px;
        }
        .summary-table {
            width: 100%;
            border-collapse: collapse;
            margin-bottom: 15px;
            font-size: 14px;
        }
        .summary-table th {
            background-color: #4CAF50;
            color: white;
            padding: 12px 8px;
            text-align: left;
            border: 1px solid #ddd;
            font-weight: bold;
        }
        .summary-table td {
            padding: 8px;
            border: 1px solid #ddd;
            text-align: left;
        }
        .summary-table tr:nth-child(even) {
            background-color: #f9f9f9;
        }
        .summary-table tr.significant {
            background-color: #e8f5e8;
        }
        .variable-name {
            font-weight: bold;
            color: #333;
            max-width: 200px;
            word-wrap: break-word;
        }
        .coefficient {
            font-family: monospace;
            text-align: right;
        }
        .p-value {
            font-family: monospace;
            text-align: right;
        }
        .significance {
            text-align: center;
            font-weight: bold;
            color: #d32f2f;
        }
        .model-stats {
            display: flex;
            flex-wrap: wrap;
            gap: 20px;
            margin-bottom: 10px;
        }
        .model-stats p {
            margin: 5px 0;
            padding: 8px 12px;
            background: #e3f2fd;
            border-radius: 4px;
            font-family: monospace;
        }
        .significance-legend {
            font-size: 12px;
            color: #666;
            font-style: italic;
        }
        h1 { color: #333; text-align: center; }
        h2 { color: #666; }
        h3 { color: #444; margin-bottom: 15px; }
    </style>
</head>
<body>
    <h1>${title}</h1>
    <h2>${forecastType} Forecasting Results</h2>
    
    ${modelTableHtml}
    
    <div class="metrics">
        <h3>📊 Performance Metrics:</h3>
        <p><strong>MSE:</strong> ${MSE(originalData, predictedData).toFixed(6)}</p>
        <p><strong>MAE:</strong> ${MAE(originalData, predictedData).toFixed(6)}</p>
        <p><strong>U_Theil:</strong> ${UTheil(originalData, predictedData).toFixed(6)}</p>
        <p><strong>Correlation:</strong> ${calculateCorrelation(originalData, predictedData).toFixed(6)}</p>
    </div>
    
    <div id="plot" class="plot-container"></div>
    
    <script>
        var trace1 = {
            x: [${timeFrames.join(', ')}],
            y: [${originalData.join(', ')}],
            type: 'scatter',
            mode: 'lines',
            name: 'Original/Truth',
            line: { color: 'red', width: 2 }
        };
        
        var trace2 = {
            x: [${timeFrames.join(', ')}],
            y: [${predictedData.join(', ')}],
            type: 'scatter',
            mode: 'lines',
            name: 'Predicted',
            line: { color: 'blue', width: 2 }
        };
        
        var trace3 = {
            x: [${timeFrames.join(', ')}],
            y: [${upperBound.join(', ')}],
            type: 'scatter',
            mode: 'lines',
            name: 'Upper CI (95%)',
            line: { color: 'lightblue', width: 1, dash: 'dash' },
            showlegend: false
        };
        
        var trace4 = {
            x: [${timeFrames.join(', ')}],
            y: [${lowerBound.join(', ')}],
            type: 'scatter',
            mode: 'lines',
            name: 'Lower CI (95%)',
            line: { color: 'lightblue', width: 1, dash: 'dash' },
            fill: 'tonexty',
            fillcolor: 'rgba(173, 216, 230, 0.2)',
            showlegend: false
        };
        
        var data = [trace4, trace3, trace1, trace2];
        
        var layout = {
            title: '${title} - ${forecastType} Forecasting',
            xaxis: { 
                title: 'Time Frames',
                showgrid: true,
                gridcolor: '#eee'
            },
            yaxis: { 
                title: 'Euler angles',
                showgrid: true,
                gridcolor: '#eee'
            },
            hovermode: 'x unified',
            legend: {
                x: 0.02,
                y: 0.98
            },
            margin: { t: 50, l: 60, r: 20, b: 60 }
        };
        
        Plotly.newPlot('plot', data, layout, {responsive: true});
    </script>
</body>
</html>`;

  fs.writeFileSync(filename, htmlContent);
  console.log(`📊 Plot saved: ${filename}`);
  return filename;
}

// Calculate correlation coefficient
function calculateCorrelation(x, y) {
  const n = x.length;
  const sumX = x.reduce((sum, val) => sum + val, 0);
  const sumY = y.reduce((sum, val) => sum + val, 0);
  const sumXY = x.reduce((sum, val, i) => sum + val * y[i], 0);
  const sumX2 = x.reduce((sum, val) => sum + val * val, 0);
  const sumY2 = y.reduce((sum, val) => sum + val * val, 0);
  
  const numerator = n * sumXY - sumX * sumY;
  const denominator = Math.sqrt((n * sumX2 - sumX * sumX) * (n * sumY2 - sumY * sumY));
  
  return denominator === 0 ? 0 : numerator / denominator;
}

// Create console-based simple plot
function createConsolePlot(originalData, predictedData, title, width = 80) {
  console.log(`\n📈 ${title}`);
  console.log('='.repeat(width));
  
  const maxVal = Math.max(...originalData, ...predictedData);
  const minVal = Math.min(...originalData, ...predictedData);
  const range = maxVal - minVal;
  
  // Show all points
  const showPoints = originalData.length;
  
  for (let i = 0; i < showPoints; i++) {
    const origNorm = Math.round((originalData[i] - minVal) / range * (width - 20));
    const predNorm = Math.round((predictedData[i] - minVal) / range * (width - 20));
    
    let line = ' '.repeat(width);
    line = line.substring(0, origNorm) + 'O' + line.substring(origNorm + 1);
    line = line.substring(0, predNorm) + 'P' + line.substring(predNorm + 1);
    
    console.log(`${i.toString().padStart(3)}: ${line}`);
  }
  
  console.log('\nLegend: O = Original, P = Predicted');
  console.log(`Correlation: ${calculateCorrelation(originalData, predictedData).toFixed(4)}`);
}

// Main execution - JavaScript Notebook Implementation
console.log("🚀 State Space Representation and Forecasting of Joint Angles - JavaScript Implementation");
console.log("==================================================================================\n");

// 1. Load BVH files to extract local joint angles
console.log("1. Loading BVH files...");

// Define paths (modify as needed)
const pathsB1 = './BVH/Bending/Train_Bending.bvh';
const pathsB2 = './BVH/Bending/Test_Bending.bvh';
const pathsG1 = './BVH/Glassblowing/Train_Glassblowing.bvh';
const pathsG2 = './BVH/Glassblowing/Test_Glassblowing.bvh';

// Load real BVH data for all motions
console.log("📁 Loading motion capture data...");

// Load Bending motion data
console.log("🔄 Loading Bending motion files...");
const bvhTrainB = extractDataFromBVH(
  './BVH/Bending/Train_Bending.bvh',
  'Hips_Xrotation',
  ['Spine_Yrotation', 'Spine_Zrotation', 'LeftArm_Xrotation', 'RightArm_Xrotation']
);

const bvhTestB = extractDataFromBVH(
  './BVH/Bending/Test_Bending.bvh',
  'Hips_Xrotation',
  ['Spine_Yrotation', 'Spine_Zrotation', 'LeftArm_Xrotation', 'RightArm_Xrotation']
);

// Load Glassblowing motion data
console.log("🔄 Loading Glassblowing motion files...");
const bvhTrainG = extractDataFromBVH(
  './BVH/Glassblowing/Train_Glassblowing.bvh',
  'Hips_Xrotation',
  ['Spine_Yrotation', 'Spine_Zrotation', 'LeftForeArm_Yrotation', 'RightForeArm_Yrotation']
);

const bvhTestG = extractDataFromBVH(
  './BVH/Glassblowing/Test_Glassblowing.bvh',
  'Hips_Xrotation',
  ['Spine_Yrotation', 'Spine_Zrotation', 'LeftForeArm_Yrotation', 'RightForeArm_Yrotation']
);

console.log(`✅ Loaded Bending: Train=${bvhTrainB.endog.length} frames, Test=${bvhTestB.endog.length} frames`);
console.log(`✅ Loaded Glassblowing: Train=${bvhTrainG.endog.length} frames, Test=${bvhTestG.endog.length} frames`);

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

console.log(`🔧 Converted data formats:`);

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
const angB = 'Hips_Xrotation';
// For BVH data: [Hips_Xrotation, Spine_Yrotation, Spine_Zrotation, LeftArm_Xrotation, RightArm_Xrotation]
const bvhAnglesB = ['Hips_Xrotation', 'Spine_Yrotation', 'Spine_Zrotation', 'LeftArm_Xrotation', 'RightArm_Xrotation'];
const indEndB = 0; // Hips_Xrotation is at index 0
const indExoB = [1, 2, 3, 4]; // Exogenous variables indices

const endogB = dataTrainB.map(row => row[indEndB]);
const exogB = dataTrainB.map(row => indExoB.map(idx => row[idx]));

console.log(`🎯 Training model for Bending motion - Target: ${angB}`);
console.log(`📊 Training data: ${endogB.length} frames, ${exogB[0].length} exogenous variables`);
const modelB = new SARIMAX(endogB, exogB, order);
modelB.fit();

const summaryB = createModelSummary(modelB, bvhAnglesB, angB, indExoB);

// 4.2 Glassblowing motion model
const angG = 'LeftForeArm_Yrotation';
// For BVH data: [Hips_Xrotation, Spine_Yrotation, Spine_Zrotation, LeftForeArm_Yrotation, RightForeArm_Yrotation]
const bvhAnglesG = ['Hips_Xrotation', 'Spine_Yrotation', 'Spine_Zrotation', 'LeftForeArm_Yrotation', 'RightForeArm_Yrotation'];
const indEndG = 3; // LeftForeArm_Yrotation is at index 3
const indExoG = [0, 1, 2, 4]; // Exogenous variables indices (Hips_Xrotation, Spine_Yrotation, Spine_Zrotation, RightForeArm_Yrotation)

const endogG = dataTrainG.map(row => row[indEndG]);
const exogG = dataTrainG.map(row => indExoG.map(idx => row[idx]));

console.log(`🎯 Training model for Glassblowing motion - Target: ${angG}`);
console.log(`📊 Training data: ${endogG.length} frames, ${exogG[0].length} exogenous variables`);
const modelG = new SARIMAX(endogG, exogG, order);
modelG.fit();

const summaryG = createModelSummary(modelG, bvhAnglesG, angG, indExoG);

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
console.log(`📈 Bending Motion (${angB}):`);
console.log(`   Static:  MSE=${mseStaticB.toFixed(6)}, MAE=${maeStaticB.toFixed(6)}, U1=${u1StaticB.toFixed(6)}`);
console.log(`   Dynamic: MSE=${mseDynamicB.toFixed(6)}, MAE=${maeDynamicB.toFixed(6)}, U1=${u1DynamicB.toFixed(6)}`);
console.log(`📈 Glassblowing Motion (${angG}):`);
console.log(`   Static:  MSE=${mseStaticG.toFixed(6)}, MAE=${maeStaticG.toFixed(6)}, U1=${u1StaticG.toFixed(6)}`);
console.log(`   Dynamic: MSE=${mseDynamicG.toFixed(6)}, MAE=${maeDynamicG.toFixed(6)}, U1=${u1DynamicG.toFixed(6)}`);

console.log("\n🎉 JavaScript SARIMAX Motion Analysis Complete!");
console.log("\n📂 Generated Files:");
console.log("   🌐 static_bending_plot.html - Interactive static forecasting plot for bending motion");
console.log("   🌐 static_glassblowing_plot.html - Interactive static forecasting plot for glassblowing motion");
console.log("   🌐 dynamic_bending_plot.html - Interactive dynamic forecasting plot for bending motion");
console.log("   🌐 dynamic_glassblowing_plot.html - Interactive dynamic forecasting plot for glassblowing motion");
console.log("\n💡 Open the HTML files in your browser to view interactive plots!");

// Export results for further analysis
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
  createConsolePlot,
  calculateCorrelation
};
