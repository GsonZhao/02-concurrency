use std::ops::{AddAssign, Deref, DerefMut, Mul};

use wide::f32x8;

pub struct Vector<T> {
    data: Vec<T>,
}

impl<T> Vector<T> {
    pub fn new(data: Vec<T>) -> Self {
        Self { data }
    }
}

impl<T> Deref for Vector<T> {
    type Target = Vec<T>;
    fn deref(&self) -> &Self::Target {
        &self.data
    }
}

impl<T> DerefMut for Vector<T> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.data
    }
}

pub fn dot_product<T>(a: &Vector<T>, b: &Vector<T>) -> T
where
    T: Mul<Output = T> + AddAssign<T> + Copy + Default,
{
    let mut result = T::default();
    for i in 0..a.len() {
        result += a[i] * b[i];
    }
    result
}

// SIMD version of dot product for f32 vectors
pub fn dot_product_simd(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len());
    let len = a.len();
    let chunk_size = 8; // f32x8 processes 8 elements at once
    let mut sum = f32x8::ZERO;

    // Process chunks of 8 elements
    let chunks = len / chunk_size;
    for i in 0..chunks {
        let start = i * chunk_size;
        let end = start + chunk_size;
        if end <= len {
            let a_chunk = f32x8::from([
                a[start],
                a[start + 1],
                a[start + 2],
                a[start + 3],
                a[start + 4],
                a[start + 5],
                a[start + 6],
                a[start + 7],
            ]);
            let b_chunk = f32x8::from([
                b[start],
                b[start + 1],
                b[start + 2],
                b[start + 3],
                b[start + 4],
                b[start + 5],
                b[start + 6],
                b[start + 7],
            ]);
            sum += a_chunk * b_chunk;
        }
    }

    // Sum the SIMD vector elements
    let mut result = sum.reduce_add();

    // Handle remaining elements
    for i in (chunks * chunk_size)..len {
        result += a[i] * b[i];
    }

    result
}
