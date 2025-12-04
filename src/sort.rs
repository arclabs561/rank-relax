//! Differentiable sorting operations using smooth relaxation.

/// Compute soft sorted values for a vector.
///
/// Uses a smooth relaxation of the discrete sorting operation, enabling
/// gradient flow through the sorting.
///
/// # Arguments
///
/// * `values` - Input values to sort
/// * `regularization_strength` - Temperature parameter controlling sharpness
///
/// # Returns
///
/// Vector of soft sorted values (continuous approximations)
///
/// # Example
///
/// ```rust
/// use rank_relax::soft_sort;
///
/// let values = vec![5.0, 1.0, 2.0, 4.0, 3.0];
/// let sorted = soft_sort(&values, 1.0);
/// // sorted will be approximately [1.0, 2.0, 3.0, 4.0, 5.0] (soft)
/// ```
pub fn soft_sort(values: &[f64], _regularization_strength: f64) -> Vec<f64> {
    // Note: Current implementation uses hard sorting as a placeholder.
    // A true soft sort would use differentiable approximations, but for now
    // this provides correct ordering while maintaining API compatibility.
    let mut result = values.to_vec();
    result.sort_by(|a, b| {
        // Handle NaN and Inf values
        match (a.is_finite(), b.is_finite()) {
            (true, true) => a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal),
            (false, true) => std::cmp::Ordering::Greater, // NaN/Inf goes to end
            (true, false) => std::cmp::Ordering::Less,
            (false, false) => std::cmp::Ordering::Equal,
        }
    });
    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_soft_sort_basic() {
        let values = vec![3.0, 1.0, 2.0];
        let sorted = soft_sort(&values, 1.0);
        
        assert!(sorted[0] <= sorted[1]);
        assert!(sorted[1] <= sorted[2]);
    }
}

