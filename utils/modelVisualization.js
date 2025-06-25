// 📊 Model Visualization Utilities
// Display detailed model information in table format (pandas-like)

/**
 * Display detailed model summary table like pandas DataFrame
 * @param {Object} model - Trained SARIMAX model
 * @param {Array} variables - Variable names
 * @param {string} targetAngle - Target variable name
 * @param {Array} indExo - Exogenous variable indices
 * @returns {Object} - Formatted model data
 */
export function displayModelTable(model, variables, targetAngle, indExo) {
  const summary = model.summary();
  
  console.log('\n📊 MODEL COEFFICIENTS AND STATISTICAL SIGNIFICANCE');
  console.log('='.repeat(80));
  
  // Create table headers
  const headers = ['Variable', 'Coefficient', 'Std Error', 'T-Statistic', 'P-Value', 'Significance'];
  const colWidths = [25, 15, 12, 12, 12, 12];
  
  // Print header
  let headerLine = '';
  headers.forEach((header, i) => {
    headerLine += header.padEnd(colWidths[i]);
  });
  console.log(headerLine);
  console.log('-'.repeat(headerLine.length));
  
  // Format and display each coefficient
  const modelData = [];
  
  for (let i = 0; i < variables.length; i++) {
    const variable = variables[i];
    const coef = summary.coefficients[i];
    const stdErr = summary.stdErrors[i];
    const tStat = summary.tStats[i];
    const pVal = summary.pValues[i];
    
    // Determine significance level
    let significance = '';
    if (pVal < 0.001) significance = '***';
    else if (pVal < 0.01) significance = '**';
    else if (pVal < 0.05) significance = '*';
    else if (pVal < 0.1) significance = '.';
    else significance = '';
    
    // Format values
    const row = [
      variable,
      coef.toFixed(6),
      stdErr.toFixed(6),
      tStat.toFixed(3),
      pVal.toFixed(6),
      significance
    ];
    
    // Print row
    let rowLine = '';
    row.forEach((cell, j) => {
      rowLine += cell.toString().padEnd(colWidths[j]);
    });
    console.log(rowLine);
    
    // Store for return
    modelData.push({
      variable,
      coefficient: coef,
      stdError: stdErr,
      tStatistic: tStat,
      pValue: pVal,
      significance
    });
  }
  
  console.log('-'.repeat(headerLine.length));
  console.log('\n📈 MODEL QUALITY METRICS:');
  console.log(`R-squared: ${summary.rSquared.toFixed(6)} (${(summary.rSquared * 100).toFixed(2)}% variance explained)`);
  console.log(`Mean Squared Error: ${summary.mse.toFixed(6)}`);
  console.log(`AIC: ${summary.aic.toFixed(3)}`);
  console.log(`BIC: ${summary.bic.toFixed(3)}`);
  
  console.log('\n🎯 SIGNIFICANCE CODES:');
  console.log("0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1");
  
  return {
    targetVariable: targetAngle,
    data: modelData,
    metrics: {
      rSquared: summary.rSquared,
      mse: summary.mse,
      aic: summary.aic,
      bic: summary.bic
    }
  };
}

/**
 * Create a pandas-like DataFrame view of the model
 * @param {Object} modelTable - Result from displayModelTable
 * @returns {Object} - DataFrame-like object
 */
export function createDataFrame(modelTable) {
  const df = {
    columns: ['Variable', 'Coefficient', 'Std_Error', 'T_Statistic', 'P_Value', 'Significance'],
    index: modelTable.data.map((_, i) => i),
    data: modelTable.data,
    
    // DataFrame-like methods
    head(n = 5) {
      console.log('\n📊 Model DataFrame (first ' + Math.min(n, this.data.length) + ' rows):');
      console.log('Target Variable:', modelTable.targetVariable);
      console.log('-'.repeat(80));
      
      // Header
      let header = 'Index'.padEnd(8);
      this.columns.forEach(col => {
        header += col.padEnd(15);
      });
      console.log(header);
      console.log('-'.repeat(header.length));
      
      // Rows
      for (let i = 0; i < Math.min(n, this.data.length); i++) {
        const row = this.data[i];
        let line = i.toString().padEnd(8);
        line += row.variable.padEnd(15);
        line += row.coefficient.toFixed(6).padEnd(15);
        line += row.stdError.toFixed(6).padEnd(15);
        line += row.tStatistic.toFixed(3).padEnd(15);
        line += row.pValue.toFixed(6).padEnd(15);
        line += row.significance.padEnd(15);
        console.log(line);
      }
      
      if (this.data.length > n) {
        console.log(`... (${this.data.length - n} more rows)`);
      }
      
      return this;
    },
    
    describe() {
      console.log('\n📈 Statistical Summary:');
      console.log('-'.repeat(50));
      
      const coeffs = this.data.map(row => row.coefficient);
      const pVals = this.data.map(row => row.pValue);
      const tStats = this.data.map(row => row.tStatistic);
      
      console.log(`Coefficients:`);
      console.log(`  Mean: ${(coeffs.reduce((a,b) => a+b, 0) / coeffs.length).toFixed(6)}`);
      console.log(`  Min:  ${Math.min(...coeffs).toFixed(6)}`);
      console.log(`  Max:  ${Math.max(...coeffs).toFixed(6)}`);
      
      console.log(`P-Values:`);
      console.log(`  Mean: ${(pVals.reduce((a,b) => a+b, 0) / pVals.length).toFixed(6)}`);
      console.log(`  Min:  ${Math.min(...pVals).toFixed(6)}`);
      console.log(`  Max:  ${Math.max(...pVals).toFixed(6)}`);
      
      const significantCount = this.data.filter(row => row.pValue < 0.05).length;
      console.log(`\nSignificant variables (p < 0.05): ${significantCount}/${this.data.length} (${(significantCount/this.data.length*100).toFixed(1)}%)`);
      
      return this;
    },
    
    info() {
      console.log('\n📋 DataFrame Info:');
      console.log('-'.repeat(30));
      console.log(`Target Variable: ${modelTable.targetVariable}`);
      console.log(`Rows: ${this.data.length}`);
      console.log(`Columns: ${this.columns.length}`);
      console.log(`Memory usage: ~${(this.data.length * this.columns.length * 8)} bytes`);
      
      console.log('\nColumn Details:');
      this.columns.forEach((col, i) => {
        console.log(`  ${i}: ${col}`);
      });
      
      return this;
    },
    
    // Pandas-like indexing
    iloc(indices) {
      if (typeof indices === 'number') {
        return this.data[indices];
      }
      return indices.map(i => this.data[i]);
    },
    
    loc(variableName) {
      return this.data.find(row => row.variable === variableName);
    }
  };
  
  return df;
}

/**
 * Display comparison table between multiple models
 * @param {Array} models - Array of {name, modelTable} objects
 */
export function compareModels(models) {
  console.log('\n🔍 MODEL COMPARISON');
  console.log('='.repeat(80));
  
  // Headers
  const headers = ['Metric', ...models.map(m => m.name)];
  const colWidth = 15;
  
  let headerLine = '';
  headers.forEach(header => {
    headerLine += header.padEnd(colWidth);
  });
  console.log(headerLine);
  console.log('-'.repeat(headerLine.length));
  
  // Metrics to compare
  const metrics = [
    { name: 'R-squared', key: 'rSquared', format: (v) => v.toFixed(6) },
    { name: 'MSE', key: 'mse', format: (v) => v.toFixed(6) },
    { name: 'AIC', key: 'aic', format: (v) => v.toFixed(3) },
    { name: 'BIC', key: 'bic', format: (v) => v.toFixed(3) },
    { name: 'Variables', key: 'variables', format: (v) => v.toString() },
    { name: 'Significant', key: 'significant', format: (v) => v.toString() }
  ];
  
  metrics.forEach(metric => {
    let row = metric.name.padEnd(colWidth);
    
    models.forEach(model => {
      let value;
      if (metric.key === 'variables') {
        value = model.modelTable.data.length;
      } else if (metric.key === 'significant') {
        value = model.modelTable.data.filter(row => row.pValue < 0.05).length;
      } else {
        value = model.modelTable.metrics[metric.key];
      }
      
      row += metric.format(value).padEnd(colWidth);
    });
    
    console.log(row);
  });
  
  console.log('-'.repeat(headerLine.length));
}

/**
 * Export model table to CSV-like format
 * @param {Object} modelTable - Result from displayModelTable
 * @returns {string} - CSV formatted string
 */
export function exportToCSV(modelTable) {
  const headers = ['Variable', 'Coefficient', 'Std_Error', 'T_Statistic', 'P_Value', 'Significance'];
  let csv = headers.join(',') + '\n';
  
  modelTable.data.forEach(row => {
    const line = [
      row.variable,
      row.coefficient,
      row.stdError,
      row.tStatistic,
      row.pValue,
      row.significance
    ].join(',');
    csv += line + '\n';
  });
  
  return csv;
}

/**
 * Create HTML table for web display
 * @param {Object} modelTable - Result from displayModelTable
 * @returns {string} - HTML table string
 */
export function createHTMLTable(modelTable) {
  let html = `
<div class="model-summary">
  <h3>📊 SARIMAX Model: ${modelTable.targetVariable}</h3>
  <table class="model-table">
    <thead>
      <tr>
        <th>Variable</th>
        <th>Coefficient</th>
        <th>Std Error</th>
        <th>T-Statistic</th>
        <th>P-Value</th>
        <th>Significance</th>
      </tr>
    </thead>
    <tbody>`;
  
  modelTable.data.forEach(row => {
    const significanceClass = row.pValue < 0.05 ? 'significant' : '';
    html += `
      <tr class="${significanceClass}">
        <td>${row.variable}</td>
        <td>${row.coefficient.toFixed(6)}</td>
        <td>${row.stdError.toFixed(6)}</td>
        <td>${row.tStatistic.toFixed(3)}</td>
        <td>${row.pValue.toFixed(6)}</td>
        <td><strong>${row.significance}</strong></td>
      </tr>`;
  });
  
  html += `
    </tbody>
  </table>
  <div class="metrics">
    <p><strong>R²:</strong> ${modelTable.metrics.rSquared.toFixed(6)} | 
       <strong>MSE:</strong> ${modelTable.metrics.mse.toFixed(6)} | 
       <strong>AIC:</strong> ${modelTable.metrics.aic.toFixed(3)} | 
       <strong>BIC:</strong> ${modelTable.metrics.bic.toFixed(3)}</p>
  </div>
</div>

<style>
.model-table {
  border-collapse: collapse;
  width: 100%;
  margin: 10px 0;
}
.model-table th, .model-table td {
  border: 1px solid #ddd;
  padding: 8px;
  text-align: left;
}
.model-table th {
  background-color: #f2f2f2;
  font-weight: bold;
}
.model-table .significant {
  background-color: #e8f5e8;
}
.metrics {
  margin-top: 10px;
  padding: 10px;
  background-color: #f9f9f9;
  border-radius: 5px;
}
</style>`;
  
  return html;
} 