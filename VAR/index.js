// 🎪 VAR Module Index
// Vector Autoregression for full-body motion prediction

export { VARModel } from './VARModel.js';
export { KfGom } from './KfGom.js';

// Convenience function to create a quick VAR model
export function createVARModel(lags = 2) {
  return new VARModel(lags);
}

// Convenience function to create a KfGom model
export function createKfGom(variables = null) {
  return new KfGom(variables);
}

// Default variables for full-body motion
export const FULL_BODY_VARIABLES = [
  'Spine_Xrotation', 'Spine_Yrotation', 'Spine_Zrotation',
  'Spine1_Xrotation', 'Spine1_Yrotation', 'Spine1_Zrotation',
  'Spine2_Xrotation', 'Spine2_Yrotation', 'Spine2_Zrotation',
  'Spine3_Xrotation', 'Spine3_Yrotation', 'Spine3_Zrotation',
  'Hips_Xrotation', 'Hips_Yrotation', 'Hips_Zrotation',
  'Neck_Xrotation', 'Neck_Yrotation', 'Neck_Zrotation',
  'Head_Xrotation', 'Head_Yrotation', 'Head_Zrotation',
  'LeftArm_Xrotation', 'LeftArm_Yrotation', 'LeftArm_Zrotation',
  'LeftForeArm_Xrotation', 'LeftForeArm_Yrotation', 'LeftForeArm_Zrotation',
  'RightArm_Xrotation', 'RightArm_Yrotation', 'RightArm_Zrotation',
  'RightForeArm_Xrotation', 'RightForeArm_Yrotation', 'RightForeArm_Zrotation',
  'LeftShoulder_Xrotation', 'LeftShoulder_Yrotation', 'LeftShoulder_Zrotation',
  'LeftShoulder2_Xrotation', 'LeftShoulder2_Yrotation', 'LeftShoulder2_Zrotation',
  'RightShoulder_Xrotation', 'RightShoulder_Yrotation', 'RightShoulder_Zrotation',
  'RightShoulder2_Xrotation', 'RightShoulder2_Yrotation', 'RightShoulder2_Zrotation',
  'LeftUpLeg_Xrotation', 'LeftUpLeg_Yrotation', 'LeftUpLeg_Zrotation',
  'LeftLeg_Xrotation', 'LeftLeg_Yrotation', 'LeftLeg_Zrotation',
  'RightUpLeg_Xrotation', 'RightUpLeg_Yrotation', 'RightUpLeg_Zrotation',
  'RightLeg_Xrotation', 'RightLeg_Yrotation', 'RightLeg_Zrotation'
];

// Simplified variable sets for testing
export const UPPER_BODY_VARIABLES = [
  'Spine_Xrotation', 'Spine_Yrotation', 'Spine_Zrotation',
  'Neck_Xrotation', 'Neck_Yrotation', 'Neck_Zrotation',
  'LeftArm_Xrotation', 'LeftArm_Yrotation', 'LeftArm_Zrotation',
  'RightArm_Xrotation', 'RightArm_Yrotation', 'RightArm_Zrotation'
];

export const CORE_VARIABLES = [
  'Hips_Xrotation', 'Hips_Yrotation', 'Hips_Zrotation',
  'Spine_Xrotation', 'Spine_Yrotation', 'Spine_Zrotation'
]; 