import * as math from 'mathjs';

// MinMaxScaler equivalent for JavaScript
export class MinMaxScaler {
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