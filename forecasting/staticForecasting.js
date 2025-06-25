// Static forecasting function
export function staticForecasting(model, testData, indEnd, indExo, scaler, targetAngleIndex) {
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