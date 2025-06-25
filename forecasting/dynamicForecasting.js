// Dynamic forecasting function
export function dynamicForecasting(model, testData, indEnd, indExo, scaler, targetAngleIndex) {
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