// Copyright (c) 2026 Ronnie Andrews, Jr. (Team Xcelerator Inc.®)
// All rights reserved. See LICENSE in the repository root.

//! Chunk-addressable vector storage and stored-operator contracts.
//!
//! These contracts do not require a vector to occupy one contiguous resident
//! allocation.  The included file-backed implementation is intentionally
//! simple and portable; production memory mapping or distributed stores can
//! implement the same traits without changing mathematical operators.

use crate::OperatorError;
use serde::{Deserialize, Serialize};
use std::fs::{File, OpenOptions};
use std::io::{Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};
use std::sync::Mutex;

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", tag = "kind")]
pub enum VectorStorageLayout {
    ContiguousMemory,
    SegmentedMemory {
        segment_elements: usize,
    },
    FileBacked {
        path: PathBuf,
        encoding: String,
        chunk_elements: usize,
    },
    Distributed {
        partitions: Vec<VectorPartition>,
        reduction_tree: String,
    },
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct VectorPartition {
    pub worker: String,
    pub start: usize,
    pub end: usize,
}

pub fn validate_distributed_partitions(
    dimension: usize,
    partitions: &[VectorPartition],
) -> Result<(), OperatorError> {
    if dimension == 0 || partitions.is_empty() {
        return Err(OperatorError::InvalidData(
            "distributed vector layout requires a positive dimension and partitions".to_owned(),
        ));
    }
    let mut expected_start = 0usize;
    for partition in partitions {
        if partition.worker.trim().is_empty()
            || partition.start != expected_start
            || partition.start >= partition.end
            || partition.end > dimension
        {
            return Err(OperatorError::InvalidData(
                "distributed partitions must be named, ordered, disjoint, and contiguous"
                    .to_owned(),
            ));
        }
        expected_start = partition.end;
    }
    if expected_start != dimension {
        return Err(OperatorError::InvalidData(
            "distributed partitions do not cover the full vector".to_owned(),
        ));
    }
    Ok(())
}

pub trait VectorReadF64: Send + Sync {
    fn dimension(&self) -> usize;
    fn preferred_chunk_elements(&self) -> usize;
    fn layout(&self) -> VectorStorageLayout;
    fn read_chunk(&self, start: usize, output: &mut [f64]) -> Result<(), OperatorError>;
}

pub trait VectorWriteF64: Send + Sync {
    fn dimension(&self) -> usize;
    fn preferred_chunk_elements(&self) -> usize;
    fn layout(&self) -> VectorStorageLayout;
    fn write_chunk(&self, start: usize, input: &[f64]) -> Result<(), OperatorError>;
    fn flush(&self) -> Result<(), OperatorError> {
        Ok(())
    }
}

pub trait VectorStorageF64: VectorReadF64 + VectorWriteF64 {}
impl<T> VectorStorageF64 for T where T: VectorReadF64 + VectorWriteF64 {}

fn validate_chunk(dimension: usize, start: usize, length: usize) -> Result<(), OperatorError> {
    if length == 0 || start > dimension || length > dimension.saturating_sub(start) {
        return Err(OperatorError::DimensionMismatch {
            expected: dimension,
            actual: start.saturating_add(length),
        });
    }
    Ok(())
}

/// Non-contiguous in-memory reference store used to validate chunk-aware
/// algorithms independently of file I/O.
#[derive(Debug)]
pub struct SegmentedVectorF64 {
    dimension: usize,
    segment_elements: usize,
    segments: Vec<Mutex<Vec<f64>>>,
}

impl SegmentedVectorF64 {
    pub fn zeros(dimension: usize, segment_elements: usize) -> Result<Self, OperatorError> {
        if dimension == 0 || segment_elements == 0 {
            return Err(OperatorError::InvalidData(
                "segmented vector dimensions must be positive".to_owned(),
            ));
        }
        let mut remaining = dimension;
        let mut segments = Vec::new();
        while remaining > 0 {
            let length = remaining.min(segment_elements);
            segments.push(Mutex::new(vec![0.0; length]));
            remaining -= length;
        }
        Ok(Self {
            dimension,
            segment_elements,
            segments,
        })
    }

    fn transfer(
        &self,
        start: usize,
        length: usize,
        mut visit: impl FnMut(&mut [f64], usize, usize) -> Result<(), OperatorError>,
    ) -> Result<(), OperatorError> {
        validate_chunk(self.dimension, start, length)?;
        let mut position = start;
        let end = start + length;
        while position < end {
            let segment_index = position / self.segment_elements;
            let segment_offset = position % self.segment_elements;
            let mut segment = self.segments[segment_index].lock().map_err(|_| {
                OperatorError::ApplicationFailed("segmented vector lock was poisoned".to_owned())
            })?;
            let take = (end - position).min(segment.len() - segment_offset);
            visit(
                &mut segment[segment_offset..segment_offset + take],
                position - start,
                take,
            )?;
            position += take;
        }
        Ok(())
    }
}

impl VectorReadF64 for SegmentedVectorF64 {
    fn dimension(&self) -> usize {
        self.dimension
    }

    fn preferred_chunk_elements(&self) -> usize {
        self.segment_elements
    }

    fn layout(&self) -> VectorStorageLayout {
        VectorStorageLayout::SegmentedMemory {
            segment_elements: self.segment_elements,
        }
    }

    fn read_chunk(&self, start: usize, output: &mut [f64]) -> Result<(), OperatorError> {
        self.transfer(start, output.len(), |segment, offset, take| {
            output[offset..offset + take].copy_from_slice(segment);
            Ok(())
        })
    }
}

impl VectorWriteF64 for SegmentedVectorF64 {
    fn dimension(&self) -> usize {
        self.dimension
    }

    fn preferred_chunk_elements(&self) -> usize {
        self.segment_elements
    }

    fn layout(&self) -> VectorStorageLayout {
        VectorStorageLayout::SegmentedMemory {
            segment_elements: self.segment_elements,
        }
    }

    fn write_chunk(&self, start: usize, input: &[f64]) -> Result<(), OperatorError> {
        self.transfer(start, input.len(), |segment, offset, take| {
            segment.copy_from_slice(&input[offset..offset + take]);
            Ok(())
        })
    }
}

/// Portable little-endian file-backed f64 vector.
#[derive(Debug)]
pub struct FileBackedVectorF64 {
    path: PathBuf,
    dimension: usize,
    chunk_elements: usize,
    file: Mutex<File>,
}

impl FileBackedVectorF64 {
    pub fn create(
        path: impl AsRef<Path>,
        dimension: usize,
        chunk_elements: usize,
    ) -> Result<Self, OperatorError> {
        if dimension == 0 || chunk_elements == 0 {
            return Err(OperatorError::InvalidData(
                "file-backed vector dimensions must be positive".to_owned(),
            ));
        }
        let byte_length = u64::try_from(dimension)
            .ok()
            .and_then(|value| value.checked_mul(8))
            .ok_or_else(|| OperatorError::InvalidData("vector byte length overflow".to_owned()))?;
        let path = path.as_ref().to_path_buf();
        let file = OpenOptions::new()
            .create(true)
            .truncate(true)
            .read(true)
            .write(true)
            .open(&path)
            .map_err(|error| {
                OperatorError::ApplicationFailed(format!(
                    "could not create file-backed vector {}: {error}",
                    path.display()
                ))
            })?;
        file.set_len(byte_length).map_err(|error| {
            OperatorError::ApplicationFailed(format!(
                "could not size file-backed vector {}: {error}",
                path.display()
            ))
        })?;
        Ok(Self {
            path,
            dimension,
            chunk_elements,
            file: Mutex::new(file),
        })
    }

    pub fn open_existing(
        path: impl AsRef<Path>,
        dimension: usize,
        chunk_elements: usize,
    ) -> Result<Self, OperatorError> {
        if dimension == 0 || chunk_elements == 0 {
            return Err(OperatorError::InvalidData(
                "file-backed vector dimensions must be positive".to_owned(),
            ));
        }
        let path = path.as_ref().to_path_buf();
        let expected_bytes = u64::try_from(dimension)
            .ok()
            .and_then(|value| value.checked_mul(8))
            .ok_or_else(|| OperatorError::InvalidData("vector byte length overflow".to_owned()))?;
        let file = OpenOptions::new()
            .read(true)
            .write(true)
            .open(&path)
            .map_err(|error| {
                OperatorError::ApplicationFailed(format!(
                    "could not open file-backed vector {}: {error}",
                    path.display()
                ))
            })?;
        let actual_bytes = file
            .metadata()
            .map_err(|error| OperatorError::ApplicationFailed(error.to_string()))?
            .len();
        if actual_bytes != expected_bytes {
            return Err(OperatorError::DimensionMismatch {
                expected: expected_bytes as usize,
                actual: actual_bytes as usize,
            });
        }
        Ok(Self {
            path,
            dimension,
            chunk_elements,
            file: Mutex::new(file),
        })
    }

    fn with_file<T>(
        &self,
        operation: impl FnOnce(&mut File) -> Result<T, std::io::Error>,
    ) -> Result<T, OperatorError> {
        let mut file = self.file.lock().map_err(|_| {
            OperatorError::ApplicationFailed("file-backed vector lock was poisoned".to_owned())
        })?;
        operation(&mut file).map_err(|error| {
            OperatorError::ApplicationFailed(format!(
                "file-backed vector {} I/O failed: {error}",
                self.path.display()
            ))
        })
    }
}

impl VectorReadF64 for FileBackedVectorF64 {
    fn dimension(&self) -> usize {
        self.dimension
    }

    fn preferred_chunk_elements(&self) -> usize {
        self.chunk_elements
    }

    fn layout(&self) -> VectorStorageLayout {
        VectorStorageLayout::FileBacked {
            path: self.path.clone(),
            encoding: "ieee754_binary64_little_endian_v1".to_owned(),
            chunk_elements: self.chunk_elements,
        }
    }

    fn read_chunk(&self, start: usize, output: &mut [f64]) -> Result<(), OperatorError> {
        validate_chunk(self.dimension, start, output.len())?;
        let mut bytes = vec![0u8; output.len() * 8];
        self.with_file(|file| {
            file.seek(SeekFrom::Start((start as u64) * 8))?;
            file.read_exact(&mut bytes)
        })?;
        let (encoded_values, _) = bytes.as_chunks::<8>();
        for (value, encoded) in output.iter_mut().zip(encoded_values) {
            *value = f64::from_le_bytes(*encoded);
        }
        Ok(())
    }
}

impl VectorWriteF64 for FileBackedVectorF64 {
    fn dimension(&self) -> usize {
        self.dimension
    }

    fn preferred_chunk_elements(&self) -> usize {
        self.chunk_elements
    }

    fn layout(&self) -> VectorStorageLayout {
        VectorStorageLayout::FileBacked {
            path: self.path.clone(),
            encoding: "ieee754_binary64_little_endian_v1".to_owned(),
            chunk_elements: self.chunk_elements,
        }
    }

    fn write_chunk(&self, start: usize, input: &[f64]) -> Result<(), OperatorError> {
        validate_chunk(self.dimension, start, input.len())?;
        let mut bytes = Vec::with_capacity(input.len() * 8);
        for value in input {
            if !value.is_finite() {
                return Err(OperatorError::InvalidData(
                    "file-backed vectors reject non-finite values".to_owned(),
                ));
            }
            bytes.extend_from_slice(&value.to_le_bytes());
        }
        self.with_file(|file| {
            file.seek(SeekFrom::Start((start as u64) * 8))?;
            file.write_all(&bytes)
        })
    }

    fn flush(&self) -> Result<(), OperatorError> {
        self.with_file(|file| file.sync_data())
    }
}

/// Operator route whose public contract consumes chunk-addressable stores.
pub trait StoredLinearOperatorF64: Send + Sync {
    fn dimension(&self) -> usize;
    fn apply_stored(
        &self,
        input: &dyn VectorReadF64,
        output: &dyn VectorWriteF64,
        maximum_workspace_elements: usize,
    ) -> Result<(), OperatorError>;
}

#[derive(Clone, Debug)]
pub struct StoredDiagonalF64 {
    diagonal: Vec<f64>,
}

impl StoredDiagonalF64 {
    pub fn new(diagonal: Vec<f64>) -> Result<Self, OperatorError> {
        if diagonal.is_empty() || diagonal.iter().any(|value| !value.is_finite()) {
            return Err(OperatorError::InvalidData(
                "stored diagonal must be finite and nonempty".to_owned(),
            ));
        }
        Ok(Self { diagonal })
    }
}

impl StoredLinearOperatorF64 for StoredDiagonalF64 {
    fn dimension(&self) -> usize {
        self.diagonal.len()
    }

    fn apply_stored(
        &self,
        input: &dyn VectorReadF64,
        output: &dyn VectorWriteF64,
        maximum_workspace_elements: usize,
    ) -> Result<(), OperatorError> {
        if input.dimension() != self.dimension() || output.dimension() != self.dimension() {
            return Err(OperatorError::DimensionMismatch {
                expected: self.dimension(),
                actual: input.dimension().min(output.dimension()),
            });
        }
        if maximum_workspace_elements == 0 {
            return Err(OperatorError::InvalidData(
                "stored operator workspace must be positive".to_owned(),
            ));
        }
        let chunk = maximum_workspace_elements
            .min(input.preferred_chunk_elements())
            .min(output.preferred_chunk_elements())
            .max(1);
        let mut buffer = vec![0.0; chunk];
        let mut start = 0usize;
        while start < self.dimension() {
            let take = (self.dimension() - start).min(chunk);
            input.read_chunk(start, &mut buffer[..take])?;
            for (offset, value) in buffer[..take].iter_mut().enumerate() {
                *value *= self.diagonal[start + offset];
            }
            output.write_chunk(start, &buffer[..take])?;
            start += take;
        }
        output.flush()
    }
}

/// Deterministic sequential reduction over arbitrary vector stores.
pub fn dot_stored_f64(
    left: &dyn VectorReadF64,
    right: &dyn VectorReadF64,
    maximum_workspace_elements: usize,
) -> Result<f64, OperatorError> {
    if left.dimension() != right.dimension() {
        return Err(OperatorError::DimensionMismatch {
            expected: left.dimension(),
            actual: right.dimension(),
        });
    }
    if maximum_workspace_elements == 0 {
        return Err(OperatorError::InvalidData(
            "stored dot-product workspace must be positive".to_owned(),
        ));
    }
    let chunk = maximum_workspace_elements
        .min(left.preferred_chunk_elements())
        .min(right.preferred_chunk_elements())
        .max(1);
    let mut left_buffer = vec![0.0; chunk];
    let mut right_buffer = vec![0.0; chunk];
    let mut total = 0.0;
    let mut start = 0usize;
    while start < left.dimension() {
        let take = (left.dimension() - start).min(chunk);
        left.read_chunk(start, &mut left_buffer[..take])?;
        right.read_chunk(start, &mut right_buffer[..take])?;
        for index in 0..take {
            total += left_buffer[index] * right_buffer[index];
        }
        start += take;
    }
    Ok(total)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::{SystemTime, UNIX_EPOCH};

    fn temporary_path(label: &str) -> PathBuf {
        let nonce = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("system clock")
            .as_nanos();
        std::env::temp_dir().join(format!(
            "xc_operator_{label}_{}_{}.bin",
            std::process::id(),
            nonce
        ))
    }

    #[test]
    fn segmented_vector_reads_and_writes_across_noncontiguous_boundaries() {
        let vector = SegmentedVectorF64::zeros(10, 3).unwrap();
        vector.write_chunk(1, &[1.0, 2.0, 3.0, 4.0, 5.0]).unwrap();
        let mut values = [0.0; 7];
        vector.read_chunk(0, &mut values).unwrap();
        assert_eq!(values, [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 0.0]);
        assert!(matches!(
            VectorReadF64::layout(&vector),
            VectorStorageLayout::SegmentedMemory {
                segment_elements: 3
            }
        ));
    }

    #[test]
    fn file_backed_operator_and_reduction_never_materialize_full_vectors() {
        let input_path = temporary_path("input");
        let output_path = temporary_path("output");
        let input = FileBackedVectorF64::create(&input_path, 7, 2).unwrap();
        let output = FileBackedVectorF64::create(&output_path, 7, 3).unwrap();
        input
            .write_chunk(0, &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0])
            .unwrap();
        let operator = StoredDiagonalF64::new(vec![2.0; 7]).unwrap();
        operator.apply_stored(&input, &output, 2).unwrap();
        let mut observed = [0.0; 7];
        output.read_chunk(0, &mut observed).unwrap();
        assert_eq!(observed, [2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0]);
        assert_eq!(dot_stored_f64(&input, &output, 2).unwrap(), 280.0);

        let reopened = FileBackedVectorF64::open_existing(&output_path, 7, 1).unwrap();
        let mut tail = [0.0; 2];
        reopened.read_chunk(5, &mut tail).unwrap();
        assert_eq!(tail, [12.0, 14.0]);
        drop(reopened);
        drop(output);
        drop(input);
        std::fs::remove_file(input_path).unwrap();
        std::fs::remove_file(output_path).unwrap();
    }

    #[test]
    fn distributed_layout_requires_exact_ordered_coverage() {
        let valid = vec![
            VectorPartition {
                worker: "worker-a".to_owned(),
                start: 0,
                end: 4,
            },
            VectorPartition {
                worker: "worker-b".to_owned(),
                start: 4,
                end: 10,
            },
        ];
        validate_distributed_partitions(10, &valid).unwrap();
        let mut gap = valid;
        gap[1].start = 5;
        assert!(validate_distributed_partitions(10, &gap).is_err());
    }
}
