import * as math from 'mathjs';

// Enhanced SARIMAX class
export class SARIMAX {
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