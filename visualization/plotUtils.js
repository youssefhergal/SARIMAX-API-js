import fs from 'fs';
import { MSE, MAE, UTheil, calculateCorrelation } from '../utils/metrics.js';

// Generate HTML plot with Plotly
export function createPlot(originalData, predictedData, title, filename, forecastType = 'Static', modelSummary = null) {
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

// Create console-based simple plot
export function createConsolePlot(originalData, predictedData, title, width = 80) {
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