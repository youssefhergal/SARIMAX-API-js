# 🚀 SARIMAX Motion Analysis - Modular Architecture

A **modular JavaScript implementation** of SARIMAX (Seasonal AutoRegressive Integrated Moving Average with eXogenous regressors) for human motion capture analysis.

## 📁 Project Structure

```
SARIMAX_API/
├── 📂 classes/                    # Core classes
│   ├── 📄 StandardScaler.js       # Standard normalization (mean=0, std=1)
│   ├── 📄 MinMaxScaler.js         # Min-max normalization (0-1 range)
│   └── 📄 SARIMAX.js              # Main SARIMAX model implementation
├── 📂 utils/                      # Utility functions
│   ├── 📄 bvhUtils.js             # BVH file parsing and data extraction
│   └── 📄 metrics.js              # Evaluation metrics (MSE, MAE, U-Theil, etc.)
├── 📂 forecasting/                # Forecasting strategies
│   ├── 📄 staticForecasting.js    # One-step-ahead predictions
│   └── 📄 dynamicForecasting.js   # Multi-step predictions
├── 📂 visualization/              # Plotting and visualization
│   └── 📄 plotUtils.js            # HTML plots with Plotly.js
├── 📄 main.js                     # Main orchestration script
├── 📄 package_modular.json        # Dependencies and configuration
└── 📄 sarimax.js                  # Original monolithic implementation
```

## 🛠️ Installation & Setup

1. **Install dependencies:**
   ```bash
   npm install mathjs jstat bvh-parser
   ```

2. **Run the modular analysis:**
   ```bash
   # Using the new modular structure
   node main.js
   
   # Or with the new package.json
   cp package_modular.json package.json
   npm start
   ```

## 🏗️ Architecture Benefits

### ✨ **Separation of Concerns**
- **Classes**: Core algorithms and data processing
- **Utils**: Reusable utility functions  
- **Forecasting**: Different prediction strategies
- **Visualization**: Plotting and presentation logic

### 🔧 **Modularity Advantages**
- **Maintainability**: Easy to modify individual components
- **Reusability**: Import only what you need
- **Testing**: Test each module independently
- **Scalability**: Add new features without affecting existing code

### 📦 **ES Modules Support**
- Modern JavaScript `import/export` syntax
- Tree-shaking for optimized bundles
- Better IDE support and intellisense

## 🎯 Usage Examples

### Import Individual Modules
```javascript
// Import only what you need
import { SARIMAX } from './classes/SARIMAX.js';
import { StandardScaler } from './classes/StandardScaler.js';
import { extractDataFromBVH } from './utils/bvhUtils.js';
import { MSE, MAE } from './utils/metrics.js';
import { staticForecasting } from './forecasting/staticForecasting.js';
import { createPlot } from './visualization/plotUtils.js';
```

### Quick Model Training
```javascript
// Load and preprocess data
const data = extractDataFromBVH('motion.bvh', 'Hips_Xrotation', ['Spine_Yrotation']);
const scaler = new StandardScaler();
const scaledData = scaler.fitTransform(data.endog);

// Train model
const model = new SARIMAX(data.endog, data.exog, 2);
model.fit();

// Make predictions
const predictions = staticForecasting(model, testData, 0, [1,2], scaler, 0);

// Evaluate and visualize
const mse = MSE(predictions.origValues, predictions.predStatic);
createPlot(predictions.origValues, predictions.predStatic, 'Results', 'output.html');
```

## 📊 Key Features

### **Data Processing**
- **StandardScaler**: Z-score normalization (μ=0, σ=1)
- **MinMaxScaler**: Range normalization (0-1)
- **BVH Parser**: Extract joint angles from motion capture files

### **Statistical Modeling**
- **SARIMAX Implementation**: Full statistical analysis
- **P-values & T-statistics**: Model significance testing
- **Model Diagnostics**: R², AIC, BIC metrics
- **Stability Correction**: Automatic AR coefficient adjustment

### **Forecasting Methods**
- **Static Forecasting**: One-step-ahead using real observations
- **Dynamic Forecasting**: Multi-step using predicted values
- **Error Metrics**: MSE, MAE, U-Theil coefficient

### **Visualization**
- **Interactive HTML Plots**: Plotly.js-powered visualization
- **Model Summary Tables**: Detailed statistical results
- **Confidence Intervals**: 95% confidence bands
- **Console Plots**: Quick ASCII visualization

## 🎨 Generated Outputs

The system generates interactive HTML plots:
- `static_bending_plot.html` - Static forecasting for bending motion
- `static_glassblowing_plot.html` - Static forecasting for glassblowing motion  
- `dynamic_bending_plot.html` - Dynamic forecasting for bending motion
- `dynamic_glassblowing_plot.html` - Dynamic forecasting for glassblowing motion

## 🔬 Technical Details

### **Supported Motion Types**
- **Bending Motion**: Hip rotation prediction using spine and arm movements
- **Glassblowing Motion**: Forearm rotation prediction using torso movements

### **Model Configuration**
- **Order**: AR(2) - 2nd order autoregressive model
- **Exogenous Variables**: 4 joint angles per motion type
- **Normalization**: StandardScaler for consistent Python sklearn behavior
- **Regularization**: Ridge regression to prevent overfitting

### **Performance Metrics**
- **Excellent Static Performance**: MSE ≈ 0.037 for bending motion
- **Challenging Dynamic Performance**: Error accumulation in multi-step prediction
- **High Correlation**: r > 0.95 for static forecasting

## 🚀 Migration from Monolithic

To migrate from the original `sarimax.js`:

1. **Backup original**: Keep `sarimax.js` as reference
2. **Update imports**: Use new modular imports  
3. **Adjust configuration**: Use `package_modular.json`
4. **Test functionality**: Verify same results with `main.js`

## 🤝 Contributing

1. **Follow module structure**: Keep related functionality together
2. **Use ES modules**: Import/export syntax
3. **Document functions**: JSDoc comments recommended
4. **Test changes**: Verify against existing BVH data

## 📝 License

MIT License - Feel free to use and modify for your projects!

---

**🎉 Happy Motion Analysis!** The modular architecture makes it easier than ever to extend and customize your SARIMAX workflows. 