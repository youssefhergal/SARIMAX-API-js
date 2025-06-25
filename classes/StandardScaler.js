import * as math from 'mathjs';

// StandardScaler equivalent for JavaScript (like sklearn.preprocessing.StandardScaler)
export class StandardScaler {
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