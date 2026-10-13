import pytorch_lightning as pl
import warnings
import numpy as np
import pandas as pd
from scipy import linalg
from scipy.linalg import LinAlgWarning
from scipy.linalg import lu_factor, lu_solve, LinAlgError
from scipy.special import logsumexp
from scipy.optimize import minimize
from sklearn.metrics import f1_score, matthews_corrcoef



# FOLDED-NORMAL LOG-DENSITY
def folded_normal_logpdf(x, mu, sigma):
    
    x  = np.asarray(x, dtype=float)
    mu = np.asarray(mu, dtype=float)

    sigma = np.maximum(np.asarray(sigma, dtype=float), 1e-12)

    z_positive = (-0.5 * ((x - mu) / sigma) ** 2)

    z_negative = (-0.5 * ((x + mu) / sigma) ** 2)

    return (np.logaddexp(z_positive, z_negative,) - np.log(sigma) - 0.5 * np.log(2.0 * np.pi))


# SENSOR-WISE FOLDED-NORMAL MLE
def fit_folded_normal_mle(validation_errors, sigma_floor=1e-8, max_iter=2000):

    validation_errors = np.asarray(validation_errors, dtype=float)

    if validation_errors.ndim != 2:
        raise ValueError(
            "validation_errors must have shape "
            "(n_steps, n_sensors). "
            f"Received shape: {validation_errors.shape}")

    if np.any(~np.isfinite(validation_errors)):
        raise ValueError(
            "validation_errors contains NaN or infinite values.")

    if np.any(validation_errors < 0.0):
        minimum_value = float(np.min(validation_errors))

        raise ValueError(
            "Folded-normal observations must be non-negative. "
            f"Minimum observed value: {minimum_value}")

    number_of_sensors = validation_errors.shape[1]

    mu_hat    = np.full(number_of_sensors, np.nan, dtype=float)
    sigma_hat = np.full(number_of_sensors, np.nan, dtype=float)

    success_flags = np.zeros(number_of_sensors, dtype=bool)

    sqrt_two_pi_log = 0.5 * np.log(2.0 * np.pi)

    for sensor in range(number_of_sensors):
        observations = validation_errors[:,sensor]
        observations = observations[np.isfinite(observations)]

        if observations.size < 2:
            raise ValueError(
                f"Sensor {sensor}: fewer than two finite "
                "validation observations.")

        sample_mean               = float(np.mean(observations))
        sample_variance           = float(np.var(observations, ddof=0))
        sample_standard_deviation = float(np.sqrt(max(sample_variance, sigma_floor ** 2)))

        root_mean_square = float(np.sqrt(np.mean(observations ** 2)))

        
        def negative_log_likelihood(parameter_vector):
            parameter_vector = np.asarray(parameter_vector, dtype=float).reshape(-1)

            if parameter_vector.size != 2:
                return float(np.inf)

            log_mu    = float(parameter_vector[0])
            log_sigma = float(parameter_vector[1])

            mu    = float(np.exp(log_mu))
            sigma = float(np.exp(log_sigma))

            if (not np.isfinite(mu) or not np.isfinite(sigma) or sigma <= sigma_floor):
                return float(np.inf)

            denominator = (2.0 * sigma ** 2)
            exponent_positive = -((observations - mu) ** 2) / denominator
            exponent_negative = -((observations + mu) ** 2) / denominator

            stacked_exponents = np.vstack((exponent_positive, exponent_negative))

            log_density = (logsumexp(stacked_exponents, axis=0) - np.log(sigma) - sqrt_two_pi_log)

            if np.any(~np.isfinite(log_density)):
                return float(np.inf)
        
            negative_log_likelihood_value = -float(np.sum(log_density))
            return negative_log_likelihood_value

    
        small_mu = max(sample_mean * 0.05, 1e-6)
        initial_candidates = [(small_mu, sample_standard_deviation),
                              (max(sample_mean, 1e-6), sample_standard_deviation),
                              (max(0.5 * sample_mean, 1e-6), max(root_mean_square, sigma_floor)),
                              (max(root_mean_square, 1e-6), max(sample_standard_deviation, sigma_floor))]

        best_result = None
        best_objective = np.inf
        for initial_mu, initial_sigma in initial_candidates:

            initial_vector = np.array([np.log(max(float(initial_mu), 1e-12)), np.log(max(float(initial_sigma), sigma_floor * 10.0))], dtype=float)

            result = minimize(negative_log_likelihood, x0=initial_vector, method="L-BFGS-B",
                              bounds=[(np.log(1e-12), np.log(max(root_mean_square * 100.0, 100.0))), (np.log(sigma_floor), np.log(max(root_mean_square * 100.0, 100.0)))],
                              options={"maxiter": int(max_iter),
                                       "ftol": 1e-12,
                                       "gtol": 1e-8})

            result_fun_array = np.asarray(result.fun, dtype=float).reshape(-1)
            if result_fun_array.size != 1:
                continue

            objective_value = float(result_fun_array[0])

            optimizer_success = bool(np.asarray(result.success).reshape(-1)[0])

            candidate_is_valid = bool(optimizer_success and np.isfinite(objective_value) and np.asarray(result.x).size == 2 
                                      and np.all(np.isfinite(result.x)))

            if (candidate_is_valid and objective_value < best_objective):
                best_result = result
                best_objective = (objective_value)

        if best_result is None:
            raise RuntimeError(
                f"Sensor {sensor}: folded-normal MLE failed "
                "for all initial values. "
                f"Sample mean={sample_mean:.8g}, "
                f"sample std={sample_standard_deviation:.8g}, "
                f"number of observations={observations.size}.")

        best_parameters = np.asarray(best_result.x, dtype=float).reshape(-1)

        estimated_mu    = float(np.exp(best_parameters[0]))
        estimated_sigma = float(np.exp(best_parameters[1]))

        final_objective = float(negative_log_likelihood(best_parameters))

        final_success = bool(np.isfinite(estimated_mu)
                             and np.isfinite(estimated_sigma)
                             and estimated_mu >= 0.0
                             and estimated_sigma > sigma_floor
                             and np.isfinite(final_objective))

        if not final_success:
            raise RuntimeError(
                f"Sensor {sensor}: invalid folded-normal "
                "MLE result. "
                f"mu={estimated_mu}, "
                f"sigma={estimated_sigma}, "
                f"objective={final_objective}.")

        mu_hat[sensor]        = estimated_mu
        sigma_hat[sensor]     = estimated_sigma
        success_flags[sensor] = True

        print(
            f"Sensor {sensor:02d}: "
            f"mu_MLE={estimated_mu:.8f}, "
            f"sigma_MLE={estimated_sigma:.8f}, "
            f"NLL={final_objective:.6f}, "
            f"success={success_flags[sensor]}"
        )

    return (mu_hat, sigma_hat, success_flags)


# LOG-LIKELIHOOD-RATIO INCREMENTS
def compute_llr(errors, mu0, mu1, sigma):
    
    errors = np.asarray(errors, dtype=float)
    
    log_changed = (folded_normal_logpdf(errors, mu1, sigma))
    log_normal  = (folded_normal_logpdf(errors, mu0, sigma))

    return (log_changed - log_normal)


# VECTORIZED CUSUM FROM PRECOMPUTED INCREMENTS
def cusum_from_increments(increments, thresholds=None, burn_in=30):
    increments = np.asarray(increments, dtype=float)

    num_steps, num_sensors = (increments.shape)
    S = np.zeros((num_steps, num_sensors))
    G = np.zeros((num_steps, num_sensors))

    detection_times = np.full(num_sensors, np.nan)
    change_points   = np.full(num_sensors, np.nan)

    detected = np.zeros(num_sensors, dtype=bool)

    for t in range(burn_in, num_steps):
        
        if t == burn_in:
            previous_S = np.zeros(num_sensors)
            previous_G = np.zeros(num_sensors)

        else:
            previous_S = S[t - 1]
            previous_G = G[t - 1]

        S[t] = (previous_S + increments[t])
        G[t] = np.maximum(0.0, previous_G + increments[t])

        if thresholds is not None:
            triggers = ((G[t] > thresholds) & (~detected))
            triggered_sensors = np.where(triggers)[0]

            for sensor in (triggered_sensors):
                
                detection_times[sensor] = t
                change_points[sensor]   = (burn_in + int(np.argmin(S[burn_in:t + 1, sensor])))
                detected[sensor]        = True

    return (detection_times, change_points, S, G)


# STATE-CONDITIONED MATRIX
def state_conditioned_transition(validation_increments, threshold, num_states=10):

    validation_increments = np.asarray(validation_increments, dtype=float)
    validation_increments = (validation_increments[np.isfinite(validation_increments)])

    if validation_increments.size == 0:
        raise ValueError("No valid validation increments.")

    w    = threshold / (num_states - 0.5)
    bins = (np.arange(num_states) + 0.5) * w

    representative_values    = (np.arange(num_states, dtype=float)* w)
    representative_values[0] = 0.25 * w

    transition_matrix = np.zeros((num_states + 1, num_states + 1))

    for state, c_r in enumerate(representative_values):
        next_cusum  = np.maximum(0.0, c_r + validation_increments)
        next_states = np.digitize(next_cusum, bins, right=True)

        counts = np.bincount(next_states, minlength=num_states + 1)

        transition_matrix[state] = (counts / counts.sum())

    transition_matrix[-1, :]  = 0.0
    transition_matrix[-1, -1] = 1.0

    return transition_matrix


# VALID FINITE ARL
def calculate_finite_arl(transition_matrix, spectral_tolerance=1e-10, condition_limit=1e12):
    Q = transition_matrix[:-1, :-1]

    system_matrix   = (np.eye(Q.shape[0]) - Q)
    eigenvalues     = np.linalg.eigvals(Q)
    spectral_radius = float(np.max(np.abs(eigenvalues)))

    condition_number = float(np.linalg.cond(system_matrix))

    alarm_probability_sum = float(transition_matrix[:-1, -1].sum())

    matrix_is_valid = bool(alarm_probability_sum > 0.0 
                           and spectral_radius < 1.0 - spectral_tolerance
                           and np.isfinite(condition_number)
                           and condition_number < condition_limit)
    
    if not matrix_is_valid:
        return (np.inf, spectral_radius, condition_number, alarm_probability_sum)

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", LinAlgWarning)

            arl_vector = np.linalg.solve(system_matrix, np.ones(Q.shape[0]))

        valid_solution = bool(np.all(np.isfinite(arl_vector))
                              and np.all(arl_vector > 0.0))

        if valid_solution:
            arl_state_0 = float(arl_vector[0])
        else:
            arl_state_0 = np.inf

    except (np.linalg.LinAlgError, LinAlgWarning, ValueError):
        arl_state_0 = np.inf

    return (arl_state_0, spectral_radius, condition_number, alarm_probability_sum)


# SENSOR-WISE THRESHOLD CALIBRATION
def calibrate_cusum_thresholds(validation_increments, validation_cusum, fallback_factor, burn_in, target_arl=160, num_states=10, candidate_offsets=range(5, 40)):
    """
    For every sensor:
    1. Start from max validation CUSUM.
    2. Test thresholds lower_bound + 5, ..., lower_bound + 39.
    3. Construct the state-conditioned Markov matrix.
    4. Select the first valid threshold with ARL >= target.
    5. Use fallback only if no valid threshold is found.
    """

    num_sensors = (validation_increments.shape[1])
    lower_bounds = np.max(validation_cusum, axis=0)
    
    selected_thresholds = np.zeros(num_sensors)
    diagnostic_rows = []

    for sensor in range(num_sensors):
        chosen_threshold  = None
        increments_sensor = (validation_increments[burn_in:, sensor])

        for offset in (candidate_offsets):
            threshold = float(lower_bounds[sensor] + offset)

            transition_matrix = (state_conditioned_transition(validation_increments=(increments_sensor), threshold=threshold, num_states=num_states))
            (arl0, spectral_radius, condition_number, alarm_probability_sum) = calculate_finite_arl(transition_matrix)

            target_pass = bool(np.isfinite(arl0) and arl0 >= target_arl)

            diagnostic_rows.append({
                "sensor": sensor,
                "threshold": threshold,
                "arl0": arl0,
                "spectral_radius":
                    spectral_radius,
                "condition_number":
                    condition_number,
                "alarm_probability_sum":
                    alarm_probability_sum,
                "target_pass":
                    target_pass,
            })

            if target_pass:
                chosen_threshold = threshold
                break

        if chosen_threshold is None:
            selected_thresholds[sensor] = (fallback_factor * lower_bounds[sensor])

        else:
            selected_thresholds[sensor] = (chosen_threshold)

    diagnostics = pd.DataFrame(diagnostic_rows)

    return (selected_thresholds, lower_bounds, diagnostics)
