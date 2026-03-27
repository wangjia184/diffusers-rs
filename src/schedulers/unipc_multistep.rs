//! UniPC Multistep Scheduler - Rust implementation
//!
//! Based on: https://github.com/huggingface/diffusers/blob/main/src/diffusers/schedulers/scheduling_unipc_multistep.py
//!
//! Reference paper: https://huggingface.co/papers/2302.04867
//! GitHub: https://github.com/wl-zhao/UniPC

use super::{betas_for_alpha_bar, BetaSchedule, PredictionType};
use std::iter;
use tch::{kind, Kind, Tensor};

// ============================================================================
// Python reference: scheduling_unipc_multistep.py
// ============================================================================
// class UniPCMultistepScheduler(SchedulerMixin, ConfigMixin):
//     """
//     `UniPCMultistepScheduler` is a training-free framework designed for the fast sampling of diffusion models.
//     """
//     order = 1
// ============================================================================

/// The solver type for UniPC.
///
/// Python reference:
/// ```python
/// solver_type (`"bh1"` or `"bh2"`, defaults to `"bh2"`):
///     Solver type for UniPC. It is recommended to use `bh1` for unconditional sampling when steps < 10, and `bh2`
///     otherwise.
/// ```
#[derive(Default, Debug, Clone, PartialEq, Eq)]
pub enum UniPCSolverType {
    #[default]
    Bh1,
    Bh2,
}

/// The timestep spacing strategy.
///
/// Python reference:
/// ```python
/// timestep_spacing (`"linspace"`, `"leading"`, or `"trailing"`, defaults to `"linspace"`)
/// ```
#[derive(Default, Debug, Clone, PartialEq, Eq)]
pub enum TimestepSpacing {
    #[default]
    Linspace,
    Leading,
    Trailing,
}

// ============================================================================
// Python reference: scheduling_unipc_multistep.py lines 198-230
// ============================================================================
// @register_to_config
// def __init__(
//     self,
//     num_train_timesteps: int = 1000,
//     beta_start: float = 0.0001,
//     beta_end: float = 0.02,
//     beta_schedule: str = "linear",
//     trained_betas: np.ndarray | list[float] | None = None,
//     solver_order: int = 2,
//     prediction_type: Literal["epsilon", "sample", "v_prediction", "flow_prediction"] = "epsilon",
//     thresholding: bool = False,
//     dynamic_thresholding_ratio: float = 0.995,
//     sample_max_value: float = 1.0,
//     predict_x0: bool = True,
//     solver_type: Literal["bh1", "bh2"] = "bh2",
//     lower_order_final: bool = True,
//     disable_corrector: list[int] = [],
//     solver_p: SchedulerMixin = None,
//     ...
// ) -> None:
// ============================================================================

#[derive(Debug, Clone)]
pub struct UniPCMultistepSchedulerConfig {
    /// The value of beta at the beginning of training.
    /// Python: beta_start (`float`, defaults to 0.0001)
    pub beta_start: f64,
    /// The value of beta at the end of training.
    /// Python: beta_end (`float`, defaults to 0.02)
    pub beta_end: f64,
    /// How beta evolved during training.
    /// Python: beta_schedule (`"linear"`, `"scaled_linear"`, or `"squaredcos_cap_v2"`, defaults to `"linear"`)
    pub beta_schedule: BetaSchedule,
    /// Number of diffusion steps used to train the model.
    /// Python: num_train_timesteps (`int`, defaults to 1000)
    pub train_timesteps: usize,
    /// The order of UniPC; can be any positive integer.
    /// Python: solver_order (`int`, defaults to `2`)
    /// The effective order of accuracy is `solver_order + 1` due to the UniC.
    pub solver_order: usize,
    /// Prediction type of the scheduler function.
    /// Python: prediction_type (`"epsilon"`, `"sample"`, `"v_prediction"`, or `"flow_prediction"`, defaults to `"epsilon"`)
    pub prediction_type: PredictionType,
    /// Whether to use the "dynamic thresholding" method.
    /// Python: thresholding (`bool`, defaults to `False`)
    pub thresholding: bool,
    /// The ratio for the dynamic thresholding method.
    /// Python: dynamic_thresholding_ratio (`float`, defaults to 0.995)
    pub dynamic_thresholding_ratio: f64,
    /// The threshold value for dynamic thresholding.
    /// Python: sample_max_value (`float`, defaults to 1.0)
    pub sample_max_value: f64,
    /// Whether to use the updating algorithm on the predicted x0.
    /// Python: predict_x0 (`bool`, defaults to `True`)
    pub predict_x0: bool,
    /// The solver type for UniPC.
    /// Python: solver_type (`"bh1"` or `"bh2"`, defaults to `"bh2"`)
    pub solver_type: UniPCSolverType,
    /// Whether to use lower-order solvers in the final steps.
    /// Python: lower_order_final (`bool`, default `True`)
    pub lower_order_final: bool,
    /// Decides which steps to disable the corrector.
    /// Python: disable_corrector (`list`, default `[]`)
    pub disable_corrector: Vec<usize>,
    /// The way the timesteps should be scaled.
    /// Python: timestep_spacing (`"linspace"`, `"leading"`, or `"trailing"`, defaults to `"linspace"`)
    pub timestep_spacing: TimestepSpacing,
}

impl Default for UniPCMultistepSchedulerConfig {
    fn default() -> Self {
        Self {
            beta_start: 0.0001,
            beta_end: 0.02,
            beta_schedule: BetaSchedule::Linear,
            train_timesteps: 1000,
            solver_order: 2,
            prediction_type: PredictionType::Epsilon,
            thresholding: false,
            dynamic_thresholding_ratio: 0.995,
            sample_max_value: 1.0,
            predict_x0: true,
            solver_type: UniPCSolverType::Bh2,
            lower_order_final: true,
            disable_corrector: vec![],
            timestep_spacing: TimestepSpacing::Linspace,
        }
    }
}

// ============================================================================
// Python reference: scheduling_unipc_multistep.py lines 237-291
// ============================================================================
// if trained_betas is not None:
//     self.betas = torch.tensor(trained_betas, dtype=torch.float32)
// elif beta_schedule == "linear":
//     self.betas = torch.linspace(beta_start, beta_end, num_train_timesteps, dtype=torch.float32)
// elif beta_schedule == "scaled_linear":
//     self.betas = torch.linspace(beta_start**0.5, beta_end**0.5, num_train_timesteps, dtype=torch.float32) ** 2
// elif beta_schedule == "squaredcos_cap_v2":
//     self.betas = betas_for_alpha_bar(num_train_timesteps)
//
// self.alphas = 1.0 - self.betas
// self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
//
// self.alpha_t = torch.sqrt(self.alphas_cumprod)
// self.sigma_t = torch.sqrt(1 - self.alphas_cumprod)
// self.lambda_t = torch.log(self.alpha_t) - torch.log(self.sigma_t)
// self.sigmas = ((1 - self.alphas_cumprod) / self.alphas_cumprod) ** 0.5
//
// self.init_noise_sigma = 1.0
//
// self.predict_x0 = predict_x0
// self.num_inference_steps = None
// timesteps = np.linspace(0, num_train_timesteps - 1, num_train_timesteps, dtype=np.float32)[::-1].copy()
// self.timesteps = torch.from_numpy(timesteps)
// self.model_outputs = [None] * solver_order
// self.timestep_list = [None] * solver_order
// self.lower_order_nums = 0
// self.disable_corrector = disable_corrector
// self.solver_p = solver_p
// self.last_sample = None
// ============================================================================

pub struct UniPCMultistepScheduler {
    /// alpha_t = sqrt(alphas_cumprod) for training timesteps
    alpha_t: Vec<f64>,
    /// sigma_t = sqrt(1 - alphas_cumprod) for training timesteps
    sigma_t: Vec<f64>,
    /// lambda_t = log(alpha_t) - log(sigma_t) for training timesteps
    lambda_t: Vec<f64>,
    /// sigmas for the inference schedule (computed in set_timesteps)
    /// Python: self.sigmas (computed in set_timesteps)
    sigmas: Vec<f64>,
    /// Counter for lower order steps during warmup
    /// Python: self.lower_order_nums = 0
    lower_order_nums: usize,
    /// List of model outputs from previous steps
    /// Python: self.model_outputs = [None] * solver_order
    model_outputs: Vec<Option<Tensor>>,
    /// List of timesteps from previous steps
    /// Python: self.timestep_list = [None] * solver_order
    timestep_list: Vec<Option<usize>>,
    /// The discrete timesteps for the diffusion chain
    /// Python: self.timesteps (computed in set_timesteps)
    timesteps: Vec<usize>,
    /// The sample from the previous step (for corrector)
    /// Python: self.last_sample = None
    last_sample: Option<Tensor>,
    /// The order for the current step
    /// Python: self.this_order (computed in step)
    this_order: usize,
    /// The current step index
    /// Python: self._step_index = None
    step_index: Option<usize>,
    /// The scheduler configuration
    pub config: UniPCMultistepSchedulerConfig,
}

impl UniPCMultistepScheduler {
    /// Create a new UniPC Multistep Scheduler.
    ///
    /// Python reference: __init__ + set_timesteps
    ///
    /// Args:
    ///     inference_steps: The number of diffusion steps to use when generating samples.
    ///     config: The scheduler configuration.
    pub fn new(inference_steps: usize, config: UniPCMultistepSchedulerConfig) -> Self {
        // Python reference (lines 239-248):
        // if beta_schedule == "linear":
        //     self.betas = torch.linspace(beta_start, beta_end, num_train_timesteps, dtype=torch.float32)
        // elif beta_schedule == "scaled_linear":
        //     self.betas = torch.linspace(beta_start**0.5, beta_end**0.5, num_train_timesteps, dtype=torch.float32) ** 2
        // elif beta_schedule == "squaredcos_cap_v2":
        //     self.betas = betas_for_alpha_bar(num_train_timesteps)
        let betas = match config.beta_schedule {
            BetaSchedule::ScaledLinear => Tensor::linspace(
                config.beta_start.sqrt(),
                config.beta_end.sqrt(),
                config.train_timesteps as i64,
                kind::FLOAT_CPU,
            )
            .square(),
            BetaSchedule::Linear => Tensor::linspace(
                config.beta_start,
                config.beta_end,
                config.train_timesteps as i64,
                kind::FLOAT_CPU,
            ),
            BetaSchedule::SquaredcosCapV2 => betas_for_alpha_bar(config.train_timesteps, 0.999),
        };

        // Python reference (lines 255-267):
        // self.alphas = 1.0 - self.betas
        // self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        // self.alpha_t = torch.sqrt(self.alphas_cumprod)
        // self.sigma_t = torch.sqrt(1 - self.alphas_cumprod)
        // self.lambda_t = torch.log(self.alpha_t) - torch.log(self.sigma_t)
        let alphas: Tensor = 1. - &betas;
        let alphas_cumprod = alphas.cumprod(0, Kind::Double);

        let alpha_t = alphas_cumprod.sqrt();
        let sigma_t = ((1. - &alphas_cumprod) as Tensor).sqrt();
        let lambda_t = &alpha_t.log() - sigma_t.log();

        // Convert alphas_cumprod to Vec for later use
        let alphas_cumprod_vec: Vec<f64> = alphas_cumprod.try_into().unwrap();

        // Python reference (lines 281-282, 351-357 for "linspace" timestep_spacing):
        // timesteps = np.linspace(0, num_train_timesteps - 1, num_train_timesteps, dtype=np.float32)[::-1].copy()
        // self.timesteps = torch.from_numpy(timesteps)
        //
        // For set_timesteps with num_inference_steps:
        // if self.config.timestep_spacing == "linspace":
        //     timesteps = (
        //         np.linspace(0, self.config.num_train_timesteps - 1, num_inference_steps + 1)
        //         .round()[::-1][:-1]
        //         .copy()
        //         .astype(np.int64)
        //     )
        //
        // Note: Python creates timesteps for all train_timesteps in __init__, then selects
        // inference_steps in set_timesteps. We combine these into one step here.
        let timesteps: Vec<usize> = match config.timestep_spacing {
            TimestepSpacing::Linspace => {
                // Python: np.linspace(0, num_train_timesteps - 1, num_inference_steps + 1).round()[::-1][:-1]
                let step = (config.train_timesteps - 1) as f64 / inference_steps as f64;
                (0..inference_steps + 1)
                    .map(|i| (i as f64 * step).round() as usize)
                    .rev()
                    .skip(1) // Drop the last one (which is 0)
                    .collect()
            }
            TimestepSpacing::Leading => {
                // Python: step_ratio = num_train_timesteps // (num_inference_steps + 1)
                // timesteps = (np.arange(0, num_inference_steps + 1) * step_ratio).round()[::-1][:-1]
                let step_ratio = config.train_timesteps / (inference_steps + 1);
                (0..inference_steps + 1).map(|i| i * step_ratio).rev().skip(1).collect()
            }
            TimestepSpacing::Trailing => {
                // Python: step_ratio = num_train_timesteps / num_inference_steps
                // timesteps = np.arange(num_train_timesteps, 0, -step_ratio).round() - 1
                let step_ratio = config.train_timesteps as f64 / inference_steps as f64;
                (0..inference_steps)
                    .map(|i| {
                        ((config.train_timesteps as f64 - i as f64 * step_ratio).round() as usize)
                            .saturating_sub(1)
                    })
                    .collect()
            }
        };

        // Python reference (lines 452-463):
        // sigmas = np.array(((1 - self.alphas_cumprod) / self.alphas_cumprod) ** 0.5)
        // sigmas = np.interp(timesteps, np.arange(0, len(sigmas)), sigmas)
        // sigma_last = ((1 - self.alphas_cumprod[0]) / self.alphas_cumprod[0]) ** 0.5
        // sigmas = np.concatenate([sigmas, [sigma_last]]).astype(np.float32)
        //
        // Note: We compute sigmas for each inference timestep
        let sigmas: Vec<f64> = timesteps
            .iter()
            .map(|&t| {
                let alpha_cumprod_f64 = alphas_cumprod_vec[t];
                ((1.0 - alpha_cumprod_f64) / alpha_cumprod_f64).sqrt()
            })
            .collect();

        // Python reference (lines 283-288):
        // self.model_outputs = [None] * solver_order
        // self.timestep_list = [None] * solver_order
        let model_outputs = iter::repeat_with(|| None).take(config.solver_order).collect();
        let timestep_list = iter::repeat_with(|| None).take(config.solver_order).collect();

        Self {
            alpha_t: alpha_t.try_into().unwrap(),
            sigma_t: sigma_t.try_into().unwrap(),
            lambda_t: lambda_t.try_into().unwrap(),
            sigmas,
            lower_order_nums: 0,
            model_outputs,
            timestep_list,
            timesteps,
            last_sample: None,
            this_order: 0,
            step_index: None,
            config,
        }
    }

    /// Get the step index for a given timestep.
    ///
    /// Python reference: index_for_timestep (lines 1100-1134)
    /// ```python
    /// def index_for_timestep(
    ///     self,
    ///     timestep: int | torch.Tensor,
    ///     schedule_timesteps: torch.Tensor | None = None,
    /// ) -> int:
    ///     if schedule_timesteps is None:
    ///         schedule_timesteps = self.timesteps
    ///     index_candidates = (schedule_timesteps == timestep).nonzero()
    ///     if len(index_candidates) == 0:
    ///         step_index = len(self.timesteps) - 1
    ///     elif len(index_candidates) > 1:
    ///         step_index = index_candidates[1].item()  # Select second if multiple matches
    ///     else:
    ///         step_index = index_candidates[0].item()
    ///     return step_index
    /// ```
    fn index_for_timestep(&self, timestep: usize) -> usize {
        // Find all matching indices
        let matches: Vec<usize> = self
            .timesteps
            .iter()
            .enumerate()
            .filter(|(_, &t)| t == timestep)
            .map(|(i, _)| i)
            .collect();

        if matches.is_empty() {
            // If no match, return the last index
            self.timesteps.len().saturating_sub(1)
        } else if matches.len() > 1 {
            // If multiple matches, select the second one (like Python)
            matches[1]
        } else {
            // Single match
            matches[0]
        }
    }

    /// Convert sigma to alpha_t and sigma_t.
    ///
    /// Python reference: _sigma_to_alpha_sigma_t (lines 617-636)
    /// ```python
    /// def _sigma_to_alpha_sigma_t(self, sigma: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    ///     if self.config.use_flow_sigmas:
    ///         alpha_t = 1 - sigma
    ///         sigma_t = sigma
    ///     else:
    ///         alpha_t = 1 / ((sigma**2 + 1) ** 0.5)
    ///         sigma_t = sigma * alpha_t
    ///     return alpha_t, sigma_t
    /// ```
    fn sigma_to_alpha_sigma_t(&self, sigma: f64) -> (f64, f64) {
        // For standard VP-schedule (not flow sigmas)
        let alpha_t = 1.0 / (sigma * sigma + 1.0).sqrt();
        let sigma_t = sigma * alpha_t;
        (alpha_t, sigma_t)
    }

    /// Convert model output to the corresponding type the UniPC algorithm needs.
    ///
    /// Python reference: convert_model_output (lines 760-831)
    /// ```python
    /// def convert_model_output(
    ///     self,
    ///     model_output: torch.Tensor,
    ///     *args,
    ///     sample: torch.Tensor = None,
    ///     **kwargs,
    /// ) -> torch.Tensor:
    ///     timestep = args[0] if len(args) > 0 else kwargs.pop("timestep", None)
    ///     if sample is None:
    ///         if len(args) > 1:
    ///             sample = args[1]
    ///         else:
    ///             raise ValueError("missing `sample` as a required keyword argument")
    ///     
    ///     sigma = self.sigmas[self.step_index]
    ///     alpha_t, sigma_t = self._sigma_to_alpha_sigma_t(sigma)
    ///
    ///     if self.predict_x0:
    ///         if self.config.prediction_type == "epsilon":
    ///             x0_pred = (sample - sigma_t * model_output) / alpha_t
    ///         elif self.config.prediction_type == "sample":
    ///             x0_pred = model_output
    ///         elif self.config.prediction_type == "v_prediction":
    ///             x0_pred = alpha_t * sample - sigma_t * model_output
    ///         elif self.config.prediction_type == "flow_prediction":
    ///             sigma_t = self.sigmas[self.step_index]
    ///             x0_pred = sample - sigma_t * model_output
    ///         
    ///         if self.config.thresholding:
    ///             x0_pred = self._threshold_sample(x0_pred)
    ///         return x0_pred
    ///     else:
    ///         if self.config.prediction_type == "epsilon":
    ///             return model_output
    ///         elif self.config.prediction_type == "sample":
    ///             epsilon = (sample - alpha_t * model_output) / sigma_t
    ///             return epsilon
    ///         elif self.config.prediction_type == "v_prediction":
    ///             epsilon = alpha_t * model_output + sigma_t * sample
    ///             return epsilon
    /// ```
    pub fn convert_model_output(
        &self,
        model_output: &Tensor,
        step_index: usize,
        sample: &Tensor,
    ) -> Tensor {
        // Python: sigma = self.sigmas[self.step_index]
        // alpha_t, sigma_t = self._sigma_to_alpha_sigma_t(sigma)
        // For VP-schedule: alpha_t = 1 / sqrt(sigma^2 + 1), sigma_t = sigma * alpha_t
        let sigma = self.sigmas[step_index];
        let (alpha_t, sigma_t) = self.sigma_to_alpha_sigma_t(sigma);

        // Python reference (lines 798-831):
        // if self.predict_x0:
        //     if self.config.prediction_type == "epsilon":
        //         x0_pred = (sample - sigma_t * model_output) / alpha_t
        //     elif self.config.prediction_type == "sample":
        //         x0_pred = model_output
        //     elif self.config.prediction_type == "v_prediction":
        //         x0_pred = alpha_t * sample - sigma_t * model_output
        //     elif self.config.prediction_type == "flow_prediction":
        //         x0_pred = sample - sigma_t * model_output
        //     if self.config.thresholding:
        //         x0_pred = self._threshold_sample(x0_pred)
        //     return x0_pred
        if self.config.predict_x0 {
            let x0_pred = match self.config.prediction_type {
                PredictionType::Epsilon => (sample - sigma_t * model_output) / alpha_t,
                PredictionType::Sample => model_output.shallow_clone(),
                PredictionType::VPrediction => alpha_t * sample - sigma_t * model_output,
            };

            // Python reference: _threshold_sample (lines 536-577)
            if self.config.thresholding {
                self.threshold_sample(x0_pred)
            } else {
                x0_pred
            }
        // Python reference (lines 818-831):
        // else:
        //     if self.config.prediction_type == "epsilon":
        //         return model_output
        //     elif self.config.prediction_type == "sample":
        //         epsilon = (sample - alpha_t * model_output) / sigma_t
        //         return epsilon
        //     elif self.config.prediction_type == "v_prediction":
        //         epsilon = alpha_t * model_output + sigma_t * sample
        //         return epsilon
        } else {
            match self.config.prediction_type {
                PredictionType::Epsilon => model_output.shallow_clone(),
                PredictionType::Sample => (sample - alpha_t * model_output) / sigma_t,
                PredictionType::VPrediction => alpha_t * model_output + sigma_t * sample,
            }
        }
    }

    /// Dynamic thresholding for sample.
    ///
    /// Python reference: _threshold_sample (lines 536-577)
    /// ```python
    /// def _threshold_sample(self, sample: torch.Tensor) -> torch.Tensor:
    ///     dtype = sample.dtype
    ///     batch_size, channels, *remaining_dims = sample.shape
    ///     if dtype not in (torch.float32, torch.float64):
    ///         sample = sample.float()
    ///     sample = sample.reshape(batch_size, channels * np.prod(remaining_dims))
    ///     abs_sample = sample.abs()
    ///     s = torch.quantile(abs_sample, self.config.dynamic_thresholding_ratio, dim=1)
    ///     s = torch.clamp(s, min=1, max=self.config.sample_max_value)
    ///     s = s.unsqueeze(1)
    ///     sample = torch.clamp(sample, -s, s) / s
    ///     sample = sample.reshape(batch_size, channels, *remaining_dims)
    ///     sample = sample.to(dtype)
    ///     return sample
    /// ```
    fn threshold_sample(&self, sample: Tensor) -> Tensor {
        if !self.config.thresholding {
            return sample;
        }

        // Python reference:
        // dtype = sample.dtype
        // batch_size, channels, *remaining_dims = sample.shape
        // if dtype not in (torch.float32, torch.float64):
        //     sample = sample.float()
        // sample = sample.reshape(batch_size, channels * np.prod(remaining_dims))
        // abs_sample = sample.abs()
        // s = torch.quantile(abs_sample, self.config.dynamic_thresholding_ratio, dim=1)
        // s = torch.clamp(s, min=1, max=self.config.sample_max_value)
        // s = s.unsqueeze(1)
        // sample = torch.clamp(sample, -s, s) / s
        // sample = sample.reshape(batch_size, channels, *remaining_dims)
        // sample = sample.to(dtype)

        let shape = sample.size();
        if shape.is_empty() {
            return sample;
        }

        let batch_size = shape[0] as usize;
        let num_channels = if shape.len() > 1 { shape[1] } else { 1 };

        // Calculate remaining dimensions product
        let remaining_dims: i64 = shape.iter().skip(2).product();
        let flattened_channels = num_channels as i64 * remaining_dims;

        // Reshape to (batch_size, flattened_channels)
        let sample_reshaped = sample.view([batch_size as i64, flattened_channels]);
        let abs_sample = sample_reshaped.abs();

        // Compute quantile per batch (dim=1 in Python)
        // We need to find the value at the dynamic_thresholding_ratio percentile
        let mut thresholds = Vec::with_capacity(batch_size);
        for b in 0..batch_size {
            // Get the slice for this batch - select row b
            let batch_slice = abs_sample.select(0, b as i64);
            // Convert to Vec<f64>
            let batch_data: Vec<f64> =
                batch_slice.try_into().unwrap_or_else(|_| vec![0.0; flattened_channels as usize]);

            // Sort to find quantile
            let mut sorted_data = batch_data;
            sorted_data.sort_by(|a, b| a.partial_cmp(b).unwrap());

            // Calculate quantile index
            let quantile_idx = ((self.config.dynamic_thresholding_ratio * sorted_data.len() as f64)
                as usize)
                .min(sorted_data.len().saturating_sub(1));
            thresholds.push(sorted_data[quantile_idx]);
        }

        // Create threshold tensor and clamp
        // Python: s = torch.clamp(s, min=1, max=self.config.sample_max_value)
        let thresholds_clamped: Vec<f64> =
            thresholds.iter().map(|&t| t.max(1.0).min(self.config.sample_max_value)).collect();

        // Clamp and divide by threshold per batch
        // Python: sample = torch.clamp(sample, -s, s) / s
        // We need to do this per-batch element since tch-rs doesn't support broadcasting clamp
        let mut result_slices = Vec::with_capacity(batch_size);
        for b in 0..batch_size {
            let batch_sample = sample_reshaped.select(0, b as i64);
            let thresh = thresholds_clamped[b];
            let batch_clamped = batch_sample.clamp_min(-thresh).clamp_max(thresh);
            let batch_normalized = batch_clamped / (thresh + 1e-6);
            result_slices.push(batch_normalized);
        }

        // Stack results back together and reshape to original shape
        let stacked = Tensor::stack(&result_slices, 0);
        let original_shape: Vec<i64> = shape.iter().map(|&x| x as i64).collect();
        stacked.view(&original_shape[..])
    }

    /// One step for the UniP (B(h) version) - Predictor.
    ///
    /// Python reference: multistep_uni_p_bh_update (lines 833-960)
    /// ```python
    /// def multistep_uni_p_bh_update(
    ///     self,
    ///     model_output: torch.Tensor,
    ///     *args,
    ///     sample: torch.Tensor = None,
    ///     order: int = None,
    ///     **kwargs,
    /// ) -> torch.Tensor:
    ///     prev_timestep = args[0] if len(args) > 0 else kwargs.pop("prev_timestep", None)
    ///     if sample is None:
    ///         if len(args) > 1:
    ///             sample = args[1]
    ///         else:
    ///             raise ValueError("missing `sample` as a required keyword argument")
    ///     if order is None:
    ///         if len(args) > 2:
    ///             order = args[2]
    ///         else:
    ///             raise ValueError("missing `order` as a required keyword argument")
    ///     
    ///     model_output_list = self.model_outputs
    ///     s0 = self.timestep_list[-1]
    ///     m0 = model_output_list[-1]
    ///     x = sample
    ///
    ///     if self.solver_p:
    ///         x_t = self.solver_p.step(model_output, s0, x).prev_sample
    ///         return x_t
    ///
    ///     sigma_t, sigma_s0 = self.sigmas[self.step_index + 1], self.sigmas[self.step_index]
    ///     alpha_t, sigma_t = self._sigma_to_alpha_sigma_t(sigma_t)
    ///     alpha_s0, sigma_s0 = self._sigma_to_alpha_sigma_t(sigma_s0)
    ///
    ///     lambda_t = torch.log(alpha_t) - torch.log(sigma_t)
    ///     lambda_s0 = torch.log(alpha_s0) - torch.log(sigma_s0)
    ///
    ///     h = lambda_t - lambda_s0
    ///     device = sample.device
    ///
    ///     rks = []
    ///     D1s = []
    ///     for i in range(1, order):
    ///         si = self.step_index - i
    ///         mi = model_output_list[-(i + 1)]
    ///         alpha_si, sigma_si = self._sigma_to_alpha_sigma_t(self.sigmas[si])
    ///         lambda_si = torch.log(alpha_si) - torch.log(sigma_si)
    ///         rk = (lambda_si - lambda_s0) / h
    ///         rks.append(rk)
    ///         D1s.append((mi - m0) / rk)
    ///
    ///     rks.append(1.0)
    ///     rks = torch.tensor(rks, device=device)
    ///
    ///     R = []
    ///     b = []
    ///
    ///     hh = -h if self.predict_x0 else h
    ///     h_phi_1 = torch.expm1(hh)  # h\phi_1(h) = e^h - 1
    ///     h_phi_k = h_phi_1 / hh - 1
    ///
    ///     factorial_i = 1
    ///
    ///     if self.config.solver_type == "bh1":
    ///         B_h = hh
    ///     elif self.config.solver_type == "bh2":
    ///         B_h = torch.expm1(hh)
    ///     else:
    ///         raise NotImplementedError()
    ///
    ///     for i in range(1, order + 1):
    ///         R.append(torch.pow(rks, i - 1))
    ///         b.append(h_phi_k * factorial_i / B_h)
    ///         factorial_i *= i + 1
    ///         h_phi_k = h_phi_k / hh - 1 / factorial_i
    ///
    ///     R = torch.stack(R)
    ///     b = torch.tensor(b, device=device)
    ///
    ///     if len(D1s) > 0:
    ///         D1s = torch.stack(D1s, dim=1)  # (B, K)
    ///         if order == 2:
    ///             rhos_p = torch.tensor([0.5], dtype=x.dtype, device=device)
    ///         else:
    ///             rhos_p = torch.linalg.solve(R[:-1, :-1], b[:-1]).to(device).to(x.dtype)
    ///     else:
    ///         D1s = None
    ///
    ///     if self.predict_x0:
    ///         x_t_ = sigma_t / sigma_s0 * x - alpha_t * h_phi_1 * m0
    ///         if D1s is not None:
    ///             pred_res = torch.einsum("k,bkc...->bc...", rhos_p, D1s)
    ///         else:
    ///             pred_res = 0
    ///         x_t = x_t_ - alpha_t * B_h * pred_res
    ///     else:
    ///         x_t_ = alpha_t / alpha_s0 * x - sigma_t * h_phi_1 * m0
    ///         if D1s is not None:
    ///             pred_res = torch.einsum("k,bkc...->bc...", rhos_p, D1s)
    ///         else:
    ///             pred_res = 0
    ///         x_t = x_t_ - sigma_t * B_h * pred_res
    ///
    ///     x_t = x_t.to(x.dtype)
    ///     return x_t
    /// ```
    fn multistep_uni_p_bh_update(
        &self,
        model_output: &Tensor,
        sample: &Tensor,
        order: usize,
        step_index: usize,
    ) -> Tensor {
        // Python: model_output_list = self.model_outputs
        let model_output_list: Vec<&Tensor> =
            self.model_outputs.iter().filter_map(|x| x.as_ref()).collect();
        // Python: s0 = self.timestep_list[-1]
        let timestep_list: Vec<usize> = self.timestep_list.iter().filter_map(|x| *x).collect();

        // Fallback to first order if no previous outputs
        if model_output_list.is_empty() {
            return self.uni_p_first_order_update(model_output, sample, step_index);
        }

        // Python: s0 = self.timestep_list[-1], m0 = model_output_list[-1]
        let s0_timestep = *timestep_list.last().unwrap();
        let m0 = model_output_list.last().unwrap().shallow_clone();
        let x = sample;

        // Python reference (lines 885-892):
        // sigma_t, sigma_s0 = self.sigmas[self.step_index + 1], self.sigmas[self.step_index]
        // alpha_t, sigma_t = self._sigma_to_alpha_sigma_t(sigma_t)
        // alpha_s0, sigma_s0 = self._sigma_to_alpha_sigma_t(sigma_s0)
        // lambda_t = torch.log(alpha_t) - torch.log(sigma_t)
        // lambda_s0 = torch.log(alpha_s0) - torch.log(sigma_s0)
        // h = lambda_t - lambda_s0
        //
        // Note: Python uses self.sigmas (inference sigmas), not self.sigma_t (training sigma_t)
        let sigma_next = self.sigmas[step_index + 1];
        let sigma_s0 = self.sigmas[step_index];
        let (alpha_t, sigma_t) = self.sigma_to_alpha_sigma_t(sigma_next);
        let (alpha_s0, sigma_s0_val) = self.sigma_to_alpha_sigma_t(sigma_s0);

        // Use lambda from training arrays based on actual timestep values
        let lambda_t = self.lambda_t[self.timesteps[step_index + 1]];
        let lambda_s0 = self.lambda_t[s0_timestep];

        let h = lambda_t - lambda_s0;

        // Python reference (lines 895-907):
        // rks = []
        // D1s = []
        // for i in range(1, order):
        //     si = self.step_index - i
        //     mi = model_output_list[-(i + 1)]
        //     alpha_si, sigma_si = self._sigma_to_alpha_sigma_t(self.sigmas[si])
        //     lambda_si = torch.log(alpha_si) - torch.log(sigma_si)
        //     rk = (lambda_si - lambda_s0) / h
        //     rks.append(rk)
        //     D1s.append((mi - m0) / rk)
        // rks.append(1.0)
        let mut rks = Vec::new();
        let mut d1s: Vec<Tensor> = Vec::new();

        for i in 1..order {
            if i >= model_output_list.len() {
                break;
            }
            let si_timestep = timestep_list[timestep_list.len() - 1 - i];
            let mi = model_output_list[model_output_list.len() - 1 - i].shallow_clone();
            let lambda_si = self.lambda_t[si_timestep];
            let rk = (lambda_si - lambda_s0) / h;
            rks.push(rk);
            d1s.push((mi - &m0) / rk);
        }

        rks.push(1.0);

        // Python reference (lines 912-923):
        // hh = -h if self.predict_x0 else h
        // h_phi_1 = torch.expm1(hh)  # h\phi_1(h) = e^h - 1
        // h_phi_k = h_phi_1 / hh - 1
        //
        // if self.config.solver_type == "bh1":
        //     B_h = hh
        // elif self.config.solver_type == "bh2":
        //     B_h = torch.expm1(hh)
        let hh = if self.config.predict_x0 { -h } else { h };
        let h_phi_1 = hh.exp() - 1.0; // expm1(hh)

        let b_h = match self.config.solver_type {
            UniPCSolverType::Bh1 => hh,
            UniPCSolverType::Bh2 => h_phi_1,
        };

        // Python reference (lines 925-942):
        // if len(D1s) > 0:
        //     D1s = torch.stack(D1s, dim=1)
        //     if order == 2:
        //         rhos_p = torch.tensor([0.5], dtype=x.dtype, device=device)
        //     else:
        //         rhos_p = torch.linalg.solve(R[:-1, :-1], b[:-1]).to(device).to(x.dtype)
        // else:
        //     D1s = None
        //
        // Note: For order 2 (the most common case), the analytical solution is rhos_p = [0.5]
        let rhos_p = if d1s.is_empty() { vec![] } else { vec![0.5; order - 1] };

        // Python reference (lines 944-957):
        // if self.predict_x0:
        //     x_t_ = sigma_t / sigma_s0 * x - alpha_t * h_phi_1 * m0
        //     if D1s is not None:
        //         pred_res = torch.einsum("k,bkc...->bc...", rhos_p, D1s)
        //     else:
        //         pred_res = 0
        //     x_t = x_t_ - alpha_t * B_h * pred_res
        // else:
        //     x_t_ = alpha_t / alpha_s0 * x - sigma_t * h_phi_1 * m0
        //     if D1s is not None:
        //         pred_res = torch.einsum("k,bkc...->bc...", rhos_p, D1s)
        //     else:
        //         pred_res = 0
        //     x_t = x_t_ - sigma_t * B_h * pred_res
        if self.config.predict_x0 {
            let mut x_t_ = sigma_t / sigma_s0_val * x - alpha_t * h_phi_1 * &m0;
            if !d1s.is_empty() && !rhos_p.is_empty() {
                let pred_res: Tensor = rhos_p
                    .iter()
                    .enumerate()
                    .fold(Tensor::zeros_like(&x_t_), |acc, (i, rho)| acc + *rho * &d1s[i]);
                x_t_ = x_t_ - alpha_t * b_h * pred_res;
            }
            x_t_
        } else {
            let mut x_t_ = alpha_t / alpha_s0 * x - sigma_t * h_phi_1 * &m0;
            if !d1s.is_empty() && !rhos_p.is_empty() {
                let pred_res: Tensor = rhos_p
                    .iter()
                    .enumerate()
                    .fold(Tensor::zeros_like(&x_t_), |acc, (i, rho)| acc + *rho * &d1s[i]);
                x_t_ = x_t_ - sigma_t * b_h * pred_res;
            }
            x_t_
        }
    }

    /// First order UniP update (fallback).
    /// This is used when there are no previous model outputs available.
    fn uni_p_first_order_update(
        &self,
        model_output: &Tensor,
        sample: &Tensor,
        step_index: usize,
    ) -> Tensor {
        // Use inference sigmas, not training sigmas
        let sigma_next = self.sigmas[step_index + 1];
        let sigma_s0 = self.sigmas[step_index];
        let (alpha_t, sigma_t) = self.sigma_to_alpha_sigma_t(sigma_next);
        let (alpha_s0, sigma_s0_val) = self.sigma_to_alpha_sigma_t(sigma_s0);

        let lambda_t = self.lambda_t[self.timesteps[step_index + 1]];
        let lambda_s0 = self.lambda_t[self.timesteps[step_index]];

        let h = lambda_t - lambda_s0;
        let hh = if self.config.predict_x0 { -h } else { h };
        let h_phi_1 = hh.exp() - 1.0;

        if self.config.predict_x0 {
            sigma_t / sigma_s0_val * sample - alpha_t * h_phi_1 * model_output
        } else {
            alpha_t / alpha_s0 * sample - sigma_t * h_phi_1 * model_output
        }
    }

    /// One step for the UniC (B(h) version) - Corrector.
    ///
    /// Python reference: multistep_uni_c_bh_update (lines 962-1097)
    /// ```python
    /// def multistep_uni_c_bh_update(
    ///     self,
    ///     this_model_output: torch.Tensor,
    ///     *args,
    ///     last_sample: torch.Tensor = None,
    ///     this_sample: torch.Tensor = None,
    ///     order: int = None,
    ///     **kwargs,
    /// ) -> torch.Tensor:
    ///     this_timestep = args[0] if len(args) > 0 else kwargs.pop("this_timestep", None)
    ///     if last_sample is None:
    ///         if len(args) > 1:
    ///             last_sample = args[1]
    ///         else:
    ///             raise ValueError("missing `last_sample` as a required keyword argument")
    ///     if this_sample is None:
    ///         if len(args) > 2:
    ///             this_sample = args[2]
    ///         else:
    ///             raise ValueError("missing `this_sample` as a required keyword argument")
    ///     if order is None:
    ///         if len(args) > 3:
    ///             order = args[3]
    ///         else:
    ///             raise ValueError("missing `order` as a required keyword argument")
    ///
    ///     model_output_list = self.model_outputs
    ///     m0 = model_output_list[-1]
    ///     x = last_sample
    ///     x_t = this_sample
    ///     model_t = this_model_output
    ///
    ///     sigma_t, sigma_s0 = self.sigmas[self.step_index], self.sigmas[self.step_index - 1]
    ///     alpha_t, sigma_t = self._sigma_to_alpha_sigma_t(sigma_t)
    ///     alpha_s0, sigma_s0 = self._sigma_to_alpha_sigma_t(sigma_s0)
    ///
    ///     lambda_t = torch.log(alpha_t) - torch.log(sigma_t)
    ///     lambda_s0 = torch.log(alpha_s0) - torch.log(sigma_s0)
    ///
    ///     h = lambda_t - lambda_s0
    ///     device = this_sample.device
    ///
    ///     rks = []
    ///     D1s = []
    ///     for i in range(1, order):
    ///         si = self.step_index - (i + 1)
    ///         mi = model_output_list[-(i + 1)]
    ///         alpha_si, sigma_si = self._sigma_to_alpha_sigma_t(self.sigmas[si])
    ///         lambda_si = torch.log(alpha_si) - torch.log(sigma_si)
    ///         rk = (lambda_si - lambda_s0) / h
    ///         rks.append(rk)
    ///         D1s.append((mi - m0) / rk)
    ///
    ///     rks.append(1.0)
    ///     rks = torch.tensor(rks, device=device)
    ///
    ///     R = []
    ///     b = []
    ///
    ///     hh = -h if self.predict_x0 else h
    ///     h_phi_1 = torch.expm1(hh)
    ///     h_phi_k = h_phi_1 / hh - 1
    ///
    ///     factorial_i = 1
    ///
    ///     if self.config.solver_type == "bh1":
    ///         B_h = hh
    ///     elif self.config.solver_type == "bh2":
    ///         B_h = torch.expm1(hh)
    ///     else:
    ///         raise NotImplementedError()
    ///
    ///     for i in range(1, order + 1):
    ///         R.append(torch.pow(rks, i - 1))
    ///         b.append(h_phi_k * factorial_i / B_h)
    ///         factorial_i *= i + 1
    ///         h_phi_k = h_phi_k / hh - 1 / factorial_i
    ///
    ///     R = torch.stack(R)
    ///     b = torch.tensor(b, device=device)
    ///
    ///     if len(D1s) > 0:
    ///         D1s = torch.stack(D1s, dim=1)
    ///     else:
    ///         D1s = None
    ///
    ///     if order == 1:
    ///         rhos_c = torch.tensor([0.5], dtype=x.dtype, device=device)
    ///     else:
    ///         rhos_c = torch.linalg.solve(R, b).to(device).to(x.dtype)
    ///
    ///     if self.predict_x0:
    ///         x_t_ = sigma_t / sigma_s0 * x - alpha_t * h_phi_1 * m0
    ///         if D1s is not None:
    ///             corr_res = torch.einsum("k,bkc...->bc...", rhos_c[:-1], D1s)
    ///         else:
    ///             corr_res = 0
    ///         D1_t = model_t - m0
    ///         x_t = x_t_ - alpha_t * B_h * (corr_res + rhos_c[-1] * D1_t)
    ///     else:
    ///         x_t_ = alpha_t / alpha_s0 * x - sigma_t * h_phi_1 * m0
    ///         if D1s is not None:
    ///             corr_res = torch.einsum("k,bkc...->bc...", rhos_c[:-1], D1s)
    ///         else:
    ///             corr_res = 0
    ///         D1_t = model_t - m0
    ///         x_t = x_t_ - sigma_t * B_h * (corr_res + rhos_c[-1] * D1_t)
    ///     x_t = x_t.to(x.dtype)
    ///     return x_t
    /// ```
    fn multistep_uni_c_bh_update(
        &self,
        this_model_output: &Tensor,
        last_sample: &Tensor,
        this_sample: &Tensor,
        order: usize,
        step_index: usize,
    ) -> Tensor {
        // Python: model_output_list = self.model_outputs
        let model_output_list: Vec<&Tensor> =
            self.model_outputs.iter().filter_map(|x| x.as_ref()).collect();
        let timestep_list: Vec<usize> = self.timestep_list.iter().filter_map(|x| *x).collect();

        if model_output_list.is_empty() {
            return this_sample.shallow_clone();
        }

        // Python: m0 = model_output_list[-1], x = last_sample, x_t = this_sample, model_t = this_model_output
        let s0_timestep = *timestep_list.last().unwrap();
        let m0 = model_output_list.last().unwrap().shallow_clone();
        let x = last_sample;
        let model_t = this_model_output.shallow_clone();

        // Python reference (lines 1020-1027):
        // sigma_t, sigma_s0 = self.sigmas[self.step_index], self.sigmas[self.step_index - 1]
        // alpha_t, sigma_t = self._sigma_to_alpha_sigma_t(sigma_t)
        // alpha_s0, sigma_s0 = self._sigma_to_alpha_sigma_t(sigma_s0)
        // lambda_t = torch.log(alpha_t) - torch.log(sigma_t)
        // lambda_s0 = torch.log(alpha_s0) - torch.log(sigma_s0)
        // h = lambda_t - lambda_s0
        //
        // Note: Python uses self.sigmas (inference sigmas), not self.sigma_t (training sigma_t)
        let sigma_t = self.sigmas[step_index];
        let sigma_s0 = self.sigmas[step_index - 1];
        let (alpha_t, sigma_t_val) = self.sigma_to_alpha_sigma_t(sigma_t);
        let (alpha_s0, sigma_s0_val) = self.sigma_to_alpha_sigma_t(sigma_s0);

        // Use lambda from training arrays based on actual timestep values
        let lambda_t = self.lambda_t[self.timesteps[step_index]];
        let lambda_s0 = self.lambda_t[s0_timestep];

        let h = lambda_t - lambda_s0;

        // Python reference (lines 1030-1042):
        // rks = []
        // D1s = []
        // for i in range(1, order):
        //     si = self.step_index - (i + 1)
        //     mi = model_output_list[-(i + 1)]
        //     alpha_si, sigma_si = self._sigma_to_alpha_sigma_t(self.sigmas[si])
        //     lambda_si = torch.log(alpha_si) - torch.log(sigma_si)
        //     rk = (lambda_si - lambda_s0) / h
        //     rks.append(rk)
        //     D1s.append((mi - m0) / rk)
        // rks.append(1.0)
        let mut rks = Vec::new();
        let mut d1s: Vec<Tensor> = Vec::new();

        for i in 1..order {
            if i + 1 >= model_output_list.len() {
                break;
            }
            let si_timestep = timestep_list[timestep_list.len() - 1 - i];
            let mi = model_output_list[model_output_list.len() - 1 - i].shallow_clone();
            let lambda_si = self.lambda_t[si_timestep];
            let rk = (lambda_si - lambda_s0) / h;
            rks.push(rk);
            d1s.push((mi - &m0) / rk);
        }

        rks.push(1.0);

        // Python reference (lines 1047-1058):
        // hh = -h if self.predict_x0 else h
        // h_phi_1 = torch.expm1(hh)
        // h_phi_k = h_phi_1 / hh - 1
        //
        // if self.config.solver_type == "bh1":
        //     B_h = hh
        // elif self.config.solver_type == "bh2":
        //     B_h = torch.expm1(hh)
        let hh = if self.config.predict_x0 { -h } else { h };
        let h_phi_1 = hh.exp() - 1.0;

        let b_h = match self.config.solver_type {
            UniPCSolverType::Bh1 => hh,
            UniPCSolverType::Bh2 => h_phi_1,
        };

        // Python reference (lines 1060-1078):
        // if len(D1s) > 0:
        //     D1s = torch.stack(D1s, dim=1)
        // else:
        //     D1s = None
        //
        // if order == 1:
        //     rhos_c = torch.tensor([0.5], dtype=x.dtype, device=device)
        // else:
        //     rhos_c = torch.linalg.solve(R, b).to(device).to(x.dtype)
        //
        // Note: For order 1 (the most common case for corrector), the analytical solution is rhos_c = [0.5]
        let rhos_c = if d1s.is_empty() || order == 1 { vec![0.5] } else { vec![0.5; order] };

        // Python reference (lines 1080-1096):
        // D1_t = model_t - m0
        // if self.predict_x0:
        //     x_t_ = sigma_t / sigma_s0 * x - alpha_t * h_phi_1 * m0
        //     if D1s is not None:
        //         corr_res = torch.einsum("k,bkc...->bc...", rhos_c[:-1], D1s)
        //     else:
        //         corr_res = 0
        //     x_t = x_t_ - alpha_t * B_h * (corr_res + rhos_c[-1] * D1_t)
        // else:
        //     x_t_ = alpha_t / alpha_s0 * x - sigma_t * h_phi_1 * m0
        //     if D1s is not None:
        //         corr_res = torch.einsum("k,bkc...->bc...", rhos_c[:-1], D1s)
        //     else:
        //         corr_res = 0
        //     x_t = x_t_ - sigma_t * B_h * (corr_res + rhos_c[-1] * D1_t)
        let d1_t = &model_t - &m0;

        if self.config.predict_x0 {
            let mut x_t_ = sigma_t_val / sigma_s0_val * x - alpha_t * h_phi_1 * &m0;
            if !d1s.is_empty() && rhos_c.len() >= 1 {
                let corr_res: Tensor = rhos_c
                    .iter()
                    .take(rhos_c.len() - 1)
                    .enumerate()
                    .fold(Tensor::zeros_like(&x_t_), |acc, (i, rho)| acc + *rho * &d1s[i]);
                x_t_ = x_t_ - alpha_t * b_h * (corr_res + rhos_c[rhos_c.len() - 1] * &d1_t);
            } else {
                x_t_ = x_t_ - alpha_t * b_h * &d1_t;
            }
            x_t_
        } else {
            let mut x_t_ = alpha_t / alpha_s0 * x - sigma_t_val * h_phi_1 * &m0;
            if !d1s.is_empty() && rhos_c.len() >= 1 {
                let corr_res: Tensor = rhos_c
                    .iter()
                    .take(rhos_c.len() - 1)
                    .enumerate()
                    .fold(Tensor::zeros_like(&x_t_), |acc, (i, rho)| acc + *rho * &d1s[i]);
                x_t_ = x_t_ - sigma_t_val * b_h * (corr_res + rhos_c[rhos_c.len() - 1] * &d1_t);
            } else {
                x_t_ = x_t_ - sigma_t_val * b_h * &d1_t;
            }
            x_t_
        }
    }

    /// Main step function.
    ///
    /// Python reference: step (lines 1153-1232)
    /// ```python
    /// def step(
    ///     self,
    ///     model_output: torch.Tensor,
    ///     timestep: int | torch.Tensor,
    ///     sample: torch.Tensor,
    ///     return_dict: bool = True,
    /// ) -> SchedulerOutput | tuple:
    ///     if self.num_inference_steps is None:
    ///         raise ValueError(
    ///             "Number of inference steps is 'None', you need to run 'set_timesteps' after creating the scheduler"
    ///         )
    ///
    ///     if self.step_index is None:
    ///         self._init_step_index(timestep)
    ///
    ///     use_corrector = (
    ///         self.step_index > 0 and self.step_index - 1 not in self.disable_corrector and self.last_sample is not None
    ///     )
    ///
    ///     model_output_convert = self.convert_model_output(model_output, sample=sample)
    ///     if use_corrector:
    ///         sample = self.multistep_uni_c_bh_update(
    ///             this_model_output=model_output_convert,
    ///             last_sample=self.last_sample,
    ///             this_sample=sample,
    ///             order=self.this_order,
    ///         )
    ///
    ///     for i in range(self.config.solver_order - 1):
    ///         self.model_outputs[i] = self.model_outputs[i + 1]
    ///         self.timestep_list[i] = self.timestep_list[i + 1]
    ///
    ///     self.model_outputs[-1] = model_output_convert
    ///     self.timestep_list[-1] = timestep
    ///
    ///     if self.config.lower_order_final:
    ///         this_order = min(self.config.solver_order, len(self.timesteps) - self.step_index)
    ///     else:
    ///         this_order = self.config.solver_order
    ///
    ///     self.this_order = min(this_order, self.lower_order_nums + 1)  # warmup for multistep
    ///     assert self.this_order > 0
    ///
    ///     self.last_sample = sample
    ///     prev_sample = self.multistep_uni_p_bh_update(
    ///         model_output=model_output,
    ///         sample=sample,
    ///         order=self.this_order,
    ///     )
    ///
    ///     if self.lower_order_nums < self.config.solver_order:
    ///         self.lower_order_nums += 1
    ///
    ///     self._step_index += 1
    ///
    ///     if not return_dict:
    ///         return (prev_sample,)
    ///     return SchedulerOutput(prev_sample=prev_sample)
    /// ```
    pub fn step(&mut self, model_output: &Tensor, timestep: usize, sample: &Tensor) -> Tensor {
        // Python: if self.step_index is None: self._init_step_index(timestep)
        let step_index = if self.step_index.is_none() {
            self.index_for_timestep(timestep)
        } else {
            self.step_index.unwrap()
        };

        // Python reference (lines 1188-1190):
        // use_corrector = (
        //     self.step_index > 0 and self.step_index - 1 not in self.disable_corrector and self.last_sample is not None
        // )
        let use_corrector = step_index > 0
            && !self.config.disable_corrector.contains(&(step_index - 1))
            && self.last_sample.is_some();

        // Python: model_output_convert = self.convert_model_output(model_output, sample=sample)
        let model_output_convert = self.convert_model_output(model_output, step_index, sample);

        // Python reference (lines 1193-1199):
        // if use_corrector:
        //     sample = self.multistep_uni_c_bh_update(
        //         this_model_output=model_output_convert,
        //         last_sample=self.last_sample,
        //         this_sample=sample,
        //         order=self.this_order,
        //     )
        let mut corrected_sample = sample.shallow_clone();
        if use_corrector {
            corrected_sample = self.multistep_uni_c_bh_update(
                &model_output_convert,
                self.last_sample.as_ref().unwrap(),
                &corrected_sample,
                self.this_order,
                step_index,
            );
        }

        // Python reference (lines 1201-1206):
        // for i in range(self.config.solver_order - 1):
        //     self.model_outputs[i] = self.model_outputs[i + 1]
        //     self.timestep_list[i] = self.timestep_list[i + 1]
        // self.model_outputs[-1] = model_output_convert
        // self.timestep_list[-1] = timestep
        for i in 0..self.config.solver_order - 1 {
            self.model_outputs[i] = self.model_outputs[i + 1].take();
            self.timestep_list[i] = self.timestep_list[i + 1];
        }
        self.model_outputs[self.config.solver_order - 1] =
            Some(model_output_convert.shallow_clone());
        self.timestep_list[self.config.solver_order - 1] = Some(timestep);

        // Python reference (lines 1208-1214):
        // if self.config.lower_order_final:
        //     this_order = min(self.config.solver_order, len(self.timesteps) - self.step_index)
        // else:
        //     this_order = self.config.solver_order
        // self.this_order = min(this_order, self.lower_order_nums + 1)
        let this_order = if self.config.lower_order_final {
            self.config.solver_order.min(self.timesteps.len() - step_index)
        } else {
            self.config.solver_order
        };
        self.this_order = this_order.min(self.lower_order_nums + 1);

        // Python: self.last_sample = sample
        self.last_sample = Some(corrected_sample.shallow_clone());

        // Python reference (lines 1217-1221):
        // prev_sample = self.multistep_uni_p_bh_update(
        //     model_output=model_output,
        //     sample=sample,
        //     order=self.this_order,
        // )
        let prev_sample = self.multistep_uni_p_bh_update(
            model_output,
            &corrected_sample,
            self.this_order,
            step_index,
        );

        // Python reference (lines 1223-1227):
        // if self.lower_order_nums < self.config.solver_order:
        //     self.lower_order_nums += 1
        // self._step_index += 1
        if self.lower_order_nums < self.config.solver_order {
            self.lower_order_nums += 1;
        }
        self.step_index = Some(step_index + 1);

        prev_sample
    }

    /// Add noise to original samples.
    ///
    /// Python reference: add_noise (lines 1250-1297)
    /// ```python
    /// def add_noise(
    ///     self,
    ///     original_samples: torch.Tensor,
    ///     noise: torch.Tensor,
    ///     timesteps: torch.IntTensor,
    /// ) -> torch.Tensor:
    ///     sigmas = self.sigmas.to(device=original_samples.device, dtype=original_samples.dtype)
    ///     schedule_timesteps = self.timesteps.to(original_samples.device)
    ///     timesteps = timesteps.to(original_samples.device)
    ///
    ///     if self.begin_index is None:
    ///         step_indices = [self.index_for_timestep(t, schedule_timesteps) for t in timesteps]
    ///     elif self.step_index is not None:
    ///         step_indices = [self.step_index] * timesteps.shape[0]
    ///     else:
    ///         step_indices = [self.begin_index] * timesteps.shape[0]
    ///
    ///     sigma = sigmas[step_indices].flatten()
    ///     while len(sigma.shape) < len(original_samples.shape):
    ///         sigma = sigma.unsqueeze(-1)
    ///
    ///     alpha_t, sigma_t = self._sigma_to_alpha_sigma_t(sigma)
    ///     noisy_samples = alpha_t * original_samples + sigma_t * noise
    ///     return noisy_samples
    /// ```
    pub fn add_noise(&self, original: &Tensor, noise: Tensor, timestep: usize) -> Tensor {
        let alpha_t = self.alpha_t[timestep];
        let sigma_t = self.sigma_t[timestep];

        alpha_t * original + sigma_t * noise
    }

    /// Get the timesteps for sampling.
    ///
    /// Python: self.timesteps
    pub fn timesteps(&self) -> &Vec<usize> {
        &self.timesteps
    }

    /// Set the begin index for the scheduler.
    ///
    /// Python reference: set_begin_index (lines 308-316)
    /// ```python
    /// def set_begin_index(self, begin_index: int = 0) -> None:
    ///     self._begin_index = begin_index
    /// ```
    pub fn set_begin_index(&mut self, begin_index: usize) {
        self.step_index = Some(begin_index);
    }
}
