# 🧮 SARIMAX Theory & Implementation Guide

## 📚 What is SARIMAX?

**SARIMAX** stands for **Seasonal AutoRegressive Integrated Moving Average with eXogenous regressors**

### 🔍 Breaking Down Each Component

#### 1. **S - Seasonal** (Not used in our current implementation)
- Handles repeating patterns over fixed periods
- Example: Daily/weekly/monthly cycles
- For motion: Could model walking cycles, breathing patterns

#### 2. **AR - AutoRegressive** (Order = 2 in our case)
- Uses past values of the same variable to predict future values
- **AR(2)** means we use the last 2 time steps

```mathematical
y(t) = φ₁ × y(t-1) + φ₂ × y(t-2) + ...
```

In our motion context:
```javascript
// Predicting hip rotation at time t using:
Hip_Xrotation(t) = φ₁ × Hip_Xrotation(t-1) + φ₂ × Hip_Xrotation(t-2) + exogenous_terms
```

#### 3. **I - Integrated** (Differencing)
- Makes the time series stationary by removing trends
- In our case: We work with raw angles (assuming stationarity)

#### 4. **MA - Moving Average**
- Uses past prediction errors to improve current predictions
- Not explicitly implemented in our simplified version

#### 5. **X - eXogenous regressors**
- External variables that influence the target variable
- **Key feature for motion analysis!**

## 🎯 SARIMAX Applied to Motion Capture

### Our Specific Problem Setup

#### **Bending Motion Model:**
```javascript
// Target: Hips_Xrotation (what we want to predict)
// Exogenous: ['Spine_Yrotation', 'Spine_Zrotation', 'LeftArm_Xrotation', 'RightArm_Xrotation']
```

**Mathematical Model:**
```
Hips_Xrotation(t) = β₀ + 
                   β₁ × Spine_Yrotation(t) + 
                   β₂ × Spine_Zrotation(t) + 
                   β₃ × LeftArm_Xrotation(t) + 
                   β₄ × RightArm_Xrotation(t) + 
                   φ₁ × Hips_Xrotation(t-1) + 
                   φ₂ × Hips_Xrotation(t-2) + 
                   ε(t)
```

Where:
- **β coefficients**: How much each body part influences hip rotation
- **φ coefficients**: How much past hip positions matter
- **ε(t)**: Random error term

#### **Glassblowing Motion Model:**
```javascript
// Target: LeftForeArm_Yrotation 
// Exogenous: ['Hips_Xrotation', 'Spine_Yrotation', 'Spine_Zrotation', 'RightForeArm_Yrotation']
```

## 🔬 Implementation Deep Dive

### 1. **Data Preparation**
```javascript
// Convert BVH motion capture to multi-channel format
const trainDataB = bvhTrainB.endog.map((endogValue, i) => [
  endogValue,        // Target: Hips_Xrotation
  ...bvhTrainB.exog[i]  // Exogenous: [Spine_Y, Spine_Z, LeftArm_X, RightArm_X]
]);

// Normalize data (critical for convergence)
const scaler = new StandardScaler();
const dataTrainB = scaler.fitTransform(trainDataB);
```

### 2. **Model Training Process**
```javascript
// In SARIMAX.fit() method:

// Step 1: Create lagged matrix for AR terms
const laggedEndog = this.laggedMatrix(this.endog, this.order);
// Creates: [[y(t-2), y(t-1)], [y(t-1), y(t)], ...]

// Step 2: Combine exogenous and lagged endogenous variables
for (let i = 0; i < laggedEndog.length; i++) {
  X.push([...laggedExog[i], ...laggedEndog[i]]);
  //      [exog variables,    AR terms    ]
  y.push(this.endog[i + this.order]);
}

// Step 3: Solve linear regression using least squares
// X * β = y  =>  β = (X'X)⁻¹X'y
const XMatrix = math.matrix(X);
const yVector = math.matrix(y);
const XT = math.transpose(XMatrix);
const XTX = math.multiply(XT, XMatrix);
const XTY = math.multiply(XT, yVector);
const beta = math.multiply(math.inv(XTX), XTY);
```

### 3. **Statistical Analysis**
```javascript
// Calculate standard errors
const sigma2 = sse / (n - k);  // Residual variance
const covMatrix = math.multiply(sigma2, math.inv(XTX));
const stdErrors = math.diag(covMatrix).map(val => Math.sqrt(Math.abs(val)));

// Calculate t-statistics and p-values
const tStats = this.coefficients.map((b, i) => b / stdErrors[i]);
const pValues = tStats.map(t => calculatePValue(t, degreesOfFreedom));
```

## 🎯 Why SARIMAX Works for Motion Analysis

### **Biomechanical Rationale:**

1. **Temporal Dependencies (AR terms):**
   - Human movement has momentum
   - Current position depends on recent positions
   - AR(2) captures immediate movement trends

2. **Inter-joint Dependencies (Exogenous terms):**
   - Body parts move in coordination
   - Spine movement influences hip rotation
   - Arm movements affect overall posture

3. **Predictive Power:**
   - **Static Forecasting**: Next frame prediction using real observations
   - **Dynamic Forecasting**: Multi-step prediction using predictions

## 📊 Model Interpretation

### **Coefficient Meanings:**

Looking at our results:
```javascript
// Example coefficients for bending motion:
Variables: ['Spine_Yrotation', 'Spine_Zrotation', 'LeftArm_Xrotation', 'RightArm_Xrotation', 
           'Hips_Xrotation_T-1', 'Hips_Xrotation_T-2']
Coefficients: [β₁, β₂, β₃, β₄, φ₁, φ₂]
```

**Interpretation:**
- **β₁ > 0**: Spine Y-rotation positively influences hip X-rotation
- **φ₁ ≈ 0.8**: Strong influence of previous hip position
- **φ₂ ≈ 0.2**: Moderate influence of position 2 steps ago
- **p-values < 0.05**: Statistically significant relationships

### **Model Quality Metrics:**

1. **R-squared**: How much variance is explained
   ```javascript
   rSquared = 1 - (SSE / SSTotal)  // Closer to 1 = better
   ```

2. **AIC/BIC**: Model complexity vs. fit trade-off
   ```javascript
   AIC = 2k - 2ln(L)  // Lower = better
   BIC = k×ln(n) - 2ln(L)  // Penalizes complexity more
   ```

## 🚀 Forecasting Strategies

### **Static Forecasting** (One-step ahead)
```javascript
// Use real observations for exogenous variables
for (let i = order; i < nob; i++) {
  const forecast = model.apply(
    endoData.slice(i - order, i),  // Real past values
    exogData.slice(i - order, i)   // Real exogenous values
  );
  predictions.push(forecast.getPrediction().predicted_mean[0]);
}
```
**Result**: Excellent performance (MSE ≈ 0.037 for bending)

### **Dynamic Forecasting** (Multi-step ahead)
```javascript
// Use predicted values for future predictions
for (let i = order; i < nob; i++) {
  const forecast = model.apply(
    predDynamic.slice(i - order, i),  // Predicted past values
    exogData.slice(i - order, i)      // Real exogenous values
  );
  predDynamic.push(forecast.getPrediction().predicted_mean[0]);
}
```
**Result**: Error accumulation causes instability

## 🎯 Practical Applications

### **Motion Analysis Use Cases:**
1. **Gait Analysis**: Predict foot placement from hip movement
2. **Rehabilitation**: Monitor recovery by predicting joint coordination
3. **Animation**: Generate realistic motion sequences
4. **Sports Analysis**: Predict performance based on technique
5. **Ergonomics**: Assess workplace movement patterns

### **Model Limitations:**
1. **Linear Relationships**: Assumes linear dependencies
2. **Stationarity**: Requires stable statistical properties
3. **Error Accumulation**: Dynamic forecasting degrades over time
4. **Complexity**: Cannot capture highly nonlinear biomechanics

## 🔮 Advanced Extensions

### **Potential Improvements:**
1. **Nonlinear Models**: Neural networks, kernel methods
2. **Multivariate Models**: Predict multiple joints simultaneously
3. **Regime-Switching**: Different models for different movement phases
4. **Bayesian Approaches**: Uncertainty quantification

### **Implementation Ideas:**
```javascript
// Vector SARIMAX for multiple targets
class VectorSARIMAX {
  constructor(multipleEndogenous, exogenous, order) {
    // Predict all joint angles simultaneously
  }
}

// Nonlinear extensions
class KernelSARIMAX extends SARIMAX {
  fit() {
    // Use kernel methods for nonlinear relationships
  }
}
```

This theoretical foundation explains why SARIMAX is particularly effective for motion capture analysis - it captures both the temporal dynamics of human movement and the inter-joint dependencies that define coordinated motion patterns. 