// Evaluation metrics for forecasting

// Mean Squared Error
export function MSE(yTrue, yPred) {
  const errors = yTrue.map((val, i) => Math.pow(val - yPred[i], 2));
  return errors.reduce((sum, err) => sum + err, 0) / errors.length;
}

// Mean Absolute Error
export function MAE(yTrue, yPred) {
  const errors = yTrue.map((val, i) => Math.abs(val - yPred[i]));
  return errors.reduce((sum, err) => sum + err, 0) / errors.length;
}

// U Theil coefficient
export function UTheil(yTrue, yPred) {
  const errors = yTrue.map((val, i) => val - yPred[i]);
  const mseError = errors.reduce((sum, err) => sum + Math.pow(err, 2), 0) / errors.length;
  const msePred = yPred.reduce((sum, val) => sum + Math.pow(val, 2), 0) / yPred.length;
  const mseTrue = yTrue.reduce((sum, val) => sum + Math.pow(val, 2), 0) / yTrue.length;
  
  return Math.sqrt(mseError) / (Math.sqrt(msePred) + Math.sqrt(mseTrue));
}

// Calculate correlation coefficient
export function calculateCorrelation(x, y) {
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

// Create model summary table
export function createModelSummary(model, angles, targetAngle, indExo) {
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