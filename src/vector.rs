use anyhow::Result;
use std::ops::{Add, Deref, Mul};

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

pub fn dot_product<T>(a: &Vector<T>, b: &Vector<T>) -> Result<T>
where
    T: Mul<Output = T> + Add<Output = T> + Copy + Default,
{
    if a.len() != b.len() {
        return Err(anyhow::anyhow!("Vectors must be the same length"));
    }
    let mut result = T::default();
    for i in 0..a.len() {
        result = result + a[i] * b[i];
    }
    Ok(result)
}
