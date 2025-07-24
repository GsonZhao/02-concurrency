use std::ops::{Add, Mul};

use anyhow::Result;

use crate::{dot_product, Vector};

pub struct Matrix<T> {
    data: Vec<T>,
    row: usize,
    col: usize,
}

impl<T> Matrix<T> {
    #[allow(dead_code)]
    pub fn new(row: usize, col: usize, data: Vec<T>) -> Self {
        Self { data, row, col }
    }
}

#[allow(dead_code)]
fn matrix_multiply<T>(a: &Matrix<T>, b: &Matrix<T>) -> Result<Matrix<T>>
where
    T: Mul<Output = T> + Add<Output = T> + Copy + Default + Sized,
{
    if a.col != b.row {
        return Err(anyhow::anyhow!("Matrix dimensions do not match"));
    }
    let mut result = Matrix::new(a.row, b.col, vec![T::default(); a.row * b.col]);
    for i in 0..a.row {
        for j in 0..b.col {
            let v1 = Vector::new(a.data[i * a.col..(i + 1) * a.col].to_vec());
            let v2 = Vector::new(
                b.data[j..]
                    .iter()
                    .step_by(b.col)
                    .copied()
                    .collect::<Vec<_>>(),
            );
            let dot = dot_product(&v1, &v2)?;
            result.data[i * b.col + j] = dot;
        }
    }
    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_matrix_multiply() {
        let a = Matrix::new(2, 2, vec![1, 2, 3, 4]);
        let b = Matrix::new(2, 2, vec![5, 6, 7, 8]);

        let c = matrix_multiply(&a, &b).unwrap();
        assert_eq!(c.data, vec![19, 22, 43, 50]);
    }
}
