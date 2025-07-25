use anyhow::Result;
use std::{
    ops::{AddAssign, Mul},
    sync::mpsc,
    thread,
};

use crate::{dot_product, dot_product_simd, Vector};

pub struct Matrix<T> {
    data: Vec<T>,
    row: usize,
    col: usize,
}

impl<T> Matrix<T> {
    pub fn new(data: Vec<T>, row: usize, col: usize) -> Self {
        Self { data, row, col }
    }
}

#[allow(dead_code)]
pub fn multiply<T>(a: &Matrix<T>, b: &Matrix<T>) -> Result<Matrix<T>>
where
    T: Mul<Output = T> + AddAssign<T> + Copy + Default,
{
    let mut result = Matrix::new(vec![T::default(); a.row * b.col], a.row, b.col);
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
            result.data[i * b.col + j] = dot_product(&v1, &v2);
        }
    }
    Ok(result)
}

#[allow(dead_code)]
pub fn multiply_simd(a: &Matrix<f32>, b: &Matrix<f32>) -> Result<Matrix<f32>> {
    let mut result = Matrix::new(vec![0.0; a.row * b.col], a.row, b.col);

    for i in 0..a.row {
        for j in 0..b.col {
            // Get row from matrix a
            let row = &a.data[i * a.col..(i + 1) * a.col];

            // Extract column from matrix b
            let col: Vec<f32> = b.data[j..].iter().step_by(b.col).copied().collect();

            // Use SIMD dot product
            result.data[i * b.col + j] = dot_product_simd(row, &col);
        }
    }

    Ok(result)
}

struct MsgInput<T> {
    idx: usize,
    row: Vector<T>,
    col: Vector<T>,
}

struct MsgOutput<T> {
    idx: usize,
    value: T,
}

struct Msg<T> {
    input: MsgInput<T>,
    output: oneshot::Sender<MsgOutput<T>>,
}

impl<T> MsgInput<T> {
    fn new(idx: usize, row: Vector<T>, col: Vector<T>) -> Self {
        Self { idx, row, col }
    }
}

impl<T> Msg<T> {
    fn new(input: MsgInput<T>, output: oneshot::Sender<MsgOutput<T>>) -> Self {
        Self { input, output }
    }
}

#[allow(dead_code)]
const THREAD_NUM: usize = 10;

#[allow(dead_code)]
pub fn multiply_threaded<T>(a: &Matrix<T>, b: &Matrix<T>) -> Result<Matrix<T>>
where
    T: Mul<Output = T> + AddAssign<T> + Copy + Default + Send + Sync + 'static,
{
    let mut result = Matrix::new(vec![T::default(); a.row * b.col], a.row, b.col);

    let senders = (0..THREAD_NUM)
        .map(|_| {
            let (tx, rx) = mpsc::channel::<Msg<T>>();
            thread::spawn(move || {
                for r in rx {
                    let v1 = r.input.row;
                    let v2 = r.input.col;
                    let value = dot_product(&v1, &v2);
                    r.output.send(MsgOutput {
                        idx: r.input.idx,
                        value,
                    })?;
                }
                Ok::<_, anyhow::Error>(())
            });

            tx
        })
        .collect::<Vec<_>>();

    let matrix_len = a.row * b.col;
    let mut receivers = Vec::with_capacity(matrix_len);
    for i in 0..a.row {
        for j in 0..b.col {
            let idx = i * b.col + j;
            let row = Vector::new(a.data[i * a.col..(i + 1) * a.col].to_vec());
            let col = Vector::new(
                b.data[j..]
                    .iter()
                    .step_by(b.col)
                    .copied()
                    .collect::<Vec<_>>(),
            );

            let (tx, rx) = oneshot::channel::<MsgOutput<T>>();
            let msg = Msg::new(MsgInput::new(idx, row, col), tx);
            senders[idx % THREAD_NUM].send(msg)?;
            receivers.push(rx);
        }
    }

    for receiver in receivers {
        let output = receiver.recv()?;
        result.data[output.idx] = output.value;
    }

    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_multiply() {
        let a = Matrix::new(vec![1, 2, 3, 4, 5, 6], 2, 3);
        let b = Matrix::new(vec![1, 2, 3, 4, 5, 6], 3, 2);
        let result = multiply(&a, &b).unwrap();
        assert_eq!(result.data, vec![22, 28, 49, 64]);
    }

    #[test]
    fn test_multiply_threaded() {
        let a = Matrix::new(vec![1, 2, 3, 4, 5, 6], 2, 3);
        let b = Matrix::new(vec![1, 2, 3, 4, 5, 6], 3, 2);
        let result = multiply_threaded(&a, &b).unwrap();
        assert_eq!(result.data, vec![22, 28, 49, 64]);
    }

    #[test]
    fn test_multiply_simd() {
        let a = Matrix::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], 2, 3);
        let b = Matrix::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], 3, 2);
        let result = multiply_simd(&a, &b).unwrap();
        assert_eq!(result.data, vec![22.0, 28.0, 49.0, 64.0]);
    }
}
