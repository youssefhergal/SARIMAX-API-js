# 🎪 VAR Module - Vector Autoregression for Motion Analysis

A complete JavaScript implementation of **Vector Autoregression (VAR)** models for full-body motion capture analysis and prediction.

## 📁 Module Structure

```
VAR/
├── 📄 VARModel.js              # Core VAR implementation
├── 📄 KfGom.js                 # Full-body motion model (Python equivalent)
├── 📄 index.js                 # Module exports and convenience functions
├── 📂 examples/
│   └── 📄 var_demo.js          # Comprehensive demonstration
├── 📂 templates/
│   └── 📄 var_template.js      # Simple template for your data
└── 📄 README.md                # This documentation
```

## 🎯 What is VAR?

**Vector Autoregression (VAR)** predicts **all variables simultaneously** using their past values. Unlike SARIMAX which predicts one joint at a time, VAR captures the complete interdependency between all body joints.

### 🔄 VAR vs SARIMAX Comparison

| Aspect | VAR | SARIMAX |
|--------|-----|---------|
| **Approach** | **Multivariate** (all joints together) | **Univariate** (one joint at a time) |
| **Predictions** | All 55+ joints simultaneously | 1 specific joint |
| **Coherence** | ✅ Guarantees motion coherence | ⚠️ No coherence guarantee |
| **Complexity** | High (thousands of parameters) | Low (6-10 parameters) |
| **Use Case** | Full-body motion generation | Specific joint analysis |

## 🚀 Quick Start

### 1. **Basic Usage**

```javascript
import { KfGom, UPPER_BODY_VARIABLES } from './VAR/index.js';

// Create and train model
const kfGom = new KfGom(UPPER_BODY_VARIABLES);
const { coef, dfPred } = kfGom.doGom(motionData);

// Make predictions
const predictions = kfGom.predAngCoef(newData, coef);
```

### 2. **Using the Template**

```bash
# Modify VAR/templates/var_template.js with your file paths
node VAR/templates/var_template.js
```

### 3. **Run the Demo**

```bash
# See comprehensive examples
node VAR/examples/var_demo.js
```

## 📊 Variable Sets

Choose the appropriate variable set for your needs:

### **CORE_VARIABLES** (6 joints - Fast)
```javascript
import { CORE_VARIABLES } from './VAR/index.js';
// Hips and Spine only - good for testing
```

### **UPPER_BODY_VARIABLES** (12 joints - Balanced)
```javascript
import { UPPER_BODY_VARIABLES } from './VAR/index.js';
// Torso, arms, neck - good balance of detail and speed
```

### **FULL_BODY_VARIABLES** (55+ joints - Complete)
```javascript
import { FULL_BODY_VARIABLES } from './VAR/index.js';
// Complete skeleton - requires significant computational resources
```

## 🏗️ Classes and Methods

### **VARModel Class**

Core VAR implementation with standard time series functionality.

```javascript
import { VARModel } from './VAR/VARModel.js';

const model = new VARModel(2); // VAR(2) - use last 2 time steps
model.fit(data, variableNames);
const predictions = model.predict(recentData, steps);
```

**Key Methods:**
- `fit(data, variableNames)` - Train the model
- `predict(data, steps)` - Make predictions
- `summary()` - Get model statistics

### **KfGom Class**

Full-body motion model equivalent to the Python implementation.

```javascript
import { KfGom } from './VAR/KfGom.js';

const kfGom = new KfGom(variables);
const { coef, dfPred } = kfGom.doGom(eulerAngles);
```

**Key Methods:**
- `doGom(eulerAngles)` - Train on motion data
- `predAngCoef(data, coef)` - Predict with coefficients
- `calculateMetrics()` - Get performance metrics
- `export()` / `import()` - Save/load models

## 📈 Performance Guidelines

### **Computational Complexity**

| Variable Set | Parameters | Training Time | Memory Usage |
|--------------|------------|---------------|--------------|
| CORE (6) | ~75 | ~1 second | Low |
| UPPER_BODY (12) | ~300 | ~5 seconds | Medium |
| FULL_BODY (55+) | ~6000+ | ~30+ seconds | High |

### **Quality Expectations**

- **Correlation > 0.8**: Excellent motion prediction
- **Correlation 0.6-0.8**: Good motion prediction  
- **Correlation < 0.6**: Poor - try different variables

## 🎯 Use Cases

### **✅ When to Use VAR:**

1. **Full-Body Motion Generation**
   - Animation and game development
   - Motion synthesis and completion
   - Virtual reality applications

2. **Motion Capture Processing**
   - Filling missing marker data
   - Noise reduction and smoothing
   - Data compression and reconstruction

3. **Biomechanical Analysis**
   - Study full-body coordination patterns
   - Analyze movement synergies
   - Research gait and posture

### **⚠️ When to Use SARIMAX Instead:**

1. **Specific Joint Analysis** - Understanding individual joint behavior
2. **Fast Prototyping** - Quick testing and iteration
3. **Limited Resources** - Computational constraints
4. **Educational Purposes** - Learning time series concepts

## 🔧 Advanced Usage

### **Custom Variable Sets**

```javascript
const customVariables = [
  'Hips_Xrotation', 'Hips_Yrotation', 'Hips_Zrotation',
  'Spine_Xrotation', 'Spine_Yrotation', 'Spine_Zrotation',
  'LeftArm_Xrotation', 'RightArm_Xrotation'
];

const kfGom = new KfGom(customVariables);
```

### **Model Export/Import**

```javascript
// Train and export
const kfGom = new KfGom();
kfGom.doGom(trainingData);
const modelData = kfGom.export();

// Import to new instance
const newModel = new KfGom();
newModel.import(modelData);
```

### **Performance Monitoring**

```javascript
const metrics = kfGom.calculateMetrics();

Object.entries(metrics).forEach(([joint, metric]) => {
  console.log(`${joint}: MSE=${metric.mse.toFixed(6)}, Corr=${metric.correlation.toFixed(4)}`);
});
```

## 🐛 Troubleshooting

### **Common Issues:**

1. **"Matrix singular" errors**
   - Reduce variable set size
   - Check for constant/zero columns
   - Add more training data

2. **Poor prediction quality**
   - Try different variable combinations
   - Increase training data size
   - Check data normalization

3. **Slow training**
   - Use smaller variable sets (CORE_VARIABLES)
   - Reduce sample size for testing
   - Consider using SARIMAX for single joints

### **Performance Optimization:**

```javascript
// For faster testing, use subset of data
const sampleSize = Math.min(500, fullData.length);
const sampledData = fullData.slice(0, sampleSize);
```

## 📚 Mathematical Background

### **VAR(p) Model Equation:**

```
Y(t) = c + Φ₁Y(t-1) + Φ₂Y(t-2) + ... + ΦₚY(t-p) + ε(t)
```

Where:
- **Y(t)**: Vector of all joint angles at time t
- **c**: Constant vector
- **Φᵢ**: Coefficient matrices for lag i
- **ε(t)**: Error vector

### **Implementation Details:**

- **Estimation**: Ordinary Least Squares (OLS) equation by equation
- **Normalization**: StandardScaler for numerical stability
- **Regularization**: Small ridge parameter to prevent singularity
- **Lag Selection**: Fixed at 2 (AR(2)) for motion capture data

## 🎉 Examples

### **Quick Motion Prediction:**

```javascript
import { KfGom, UPPER_BODY_VARIABLES } from './VAR/index.js';
import { extractDataFromBVH } from '../utils/bvhUtils.js';

// Load data
const data = extractDataFromBVH('motion.bvh', UPPER_BODY_VARIABLES[0], UPPER_BODY_VARIABLES.slice(1));
const motionMatrix = data.endog.map((y, i) => [y, ...data.exog[i]]);

// Train and predict
const kfGom = new KfGom(UPPER_BODY_VARIABLES);
const { coef } = kfGom.doGom(motionMatrix);
const predictions = kfGom.predAngCoef(newMotionData, coef);
```

### **Performance Analysis:**

```javascript
const metrics = kfGom.calculateMetrics();
const avgCorrelation = Object.values(metrics)
  .reduce((sum, m) => sum + m.correlation, 0) / Object.keys(metrics).length;

console.log(`Average motion prediction quality: ${avgCorrelation.toFixed(4)}`);
```

## 📝 License

This VAR implementation is part of the SARIMAX Motion Analysis project and follows the same MIT license.

---

**🎪 Happy Full-Body Motion Prediction!** The VAR module provides a powerful tool for understanding and generating coherent human movement patterns. 