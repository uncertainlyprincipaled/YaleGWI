# Debug Mode Implementation for S3 I/O Testing

## Overview

The debug mode allows you to process only one geological family instead of all families, which dramatically reduces preprocessing time from 10+ minutes to 1-2 minutes. This is perfect for quickly identifying and debugging S3 I/O issues that only appear during the full preprocessing pipeline.

## Problem Solved

**Original Issue**: S3 I/O errors that only appear after 10+ minutes of preprocessing, making debugging slow and frustrating.

**Solution**: Debug mode processes only one family, allowing you to:
- Test S3 connectivity quickly
- Identify I/O issues in minutes instead of hours
- Iterate on fixes rapidly
- Validate different families independently

## Implementation Details

### 1. Enhanced Functions

#### `complete_colab_setup()`
- Added `debug_mode: bool = False` parameter
- Added `debug_family: str = 'FlatVel_A'` parameter
- Passes debug parameters to `run_preprocessing()`

#### `quick_colab_setup()`
- Added same debug parameters
- Provides quick access to debug functionality

#### `run_preprocessing()`
- Added debug mode handling
- Skips data existence checks in debug mode
- Uses `load_data_debug()` instead of `load_data()` when in debug mode

### 2. New Function: `load_data_debug()`

Located in `src/core/preprocess.py`, this function:
- Processes only the specified family
- Creates simplified GPU datasets (all data in GPU0)
- Maintains the same output structure for compatibility
- Provides detailed logging with 🐛 emoji for easy identification

### 3. Debug Mode Features

#### Fast Processing
- Processes only 1 family instead of 10 families
- Reduces time from 10+ minutes to 1-2 minutes
- Maintains all preprocessing steps (downsampling, normalization, etc.)

#### Simplified GPU Splitting
- Puts all processed data in GPU0 for simplicity
- Creates empty GPU1 dataset to maintain structure
- Allows testing without complex GPU splitting logic

#### Enhanced Logging
- All debug messages prefixed with 🐛
- Clear indication of which family is being processed
- Detailed S3/local path information

#### Skip Data Checks
- Bypasses existing data detection in debug mode
- Forces reprocessing for testing
- Allows testing even if data already exists

## Usage Examples

### Basic Debug Mode
```python
from src.utils.colab_setup import quick_colab_setup

# Test S3 I/O with one family
results = quick_colab_setup(
    use_s3=True,
    debug_mode=True,
    debug_family='FlatVel_A'
)
```

### Test Different Families
```python
# Test different families to isolate issues
families = ['FlatVel_A', 'CurveVel_A', 'Style_A', 'FlatFault_A']

for family in families:
    print(f"Testing family: {family}")
    results = quick_colab_setup(
        use_s3=True,
        debug_mode=True,
        debug_family=family,
        run_tests=False  # Skip tests for faster iteration
    )
```

### Local vs S3 Testing
```python
# Test local processing first
results_local = quick_colab_setup(
    use_s3=False,
    debug_mode=True,
    debug_family='FlatVel_A'
)

# Then test S3 processing
results_s3 = quick_colab_setup(
    use_s3=True,
    debug_mode=True,
    debug_family='FlatVel_A'
)
```

### Full Setup with Debug Mode
```python
from src.utils.colab_setup import complete_colab_setup

results = complete_colab_setup(
    use_s3=True,
    mount_drive=True,
    debug_mode=True,
    debug_family='FlatVel_A',
    run_tests=True
)
```

## Testing

A comprehensive test suite is available in `tests/test_debug_mode.py`:

```bash
# Run debug mode tests
python tests/test_debug_mode.py
```

The test suite covers:
- Local processing in debug mode
- S3 processing in debug mode
- Different family testing
- Error handling

## Benefits

### For Development
- **Rapid Iteration**: Test changes in 1-2 minutes instead of 10+ minutes
- **Isolated Testing**: Test specific families independently
- **S3 Debugging**: Quickly identify S3 connectivity and I/O issues
- **Regression Testing**: Verify fixes work across different families

### For Production
- **Validation**: Test S3 setup before running full preprocessing
- **Troubleshooting**: Isolate issues to specific families or data types
- **Performance**: Test different families for performance characteristics

## Error Handling

Debug mode includes robust error handling:
- Graceful failure if family doesn't exist
- Clear error messages for S3 connectivity issues
- Fallback to local processing if S3 fails
- Detailed logging for troubleshooting

## Future Enhancements

Potential improvements:
- **Multi-family Debug**: Process subset of families (e.g., 2-3 families)
- **Selective Processing**: Skip specific preprocessing steps in debug mode
- **Performance Profiling**: Add timing information for each step
- **Memory Monitoring**: Track memory usage during debug processing

## Integration

Debug mode integrates seamlessly with existing functionality:
- Works with both `quick_colab_setup()` and `complete_colab_setup()`
- Maintains compatibility with all existing parameters
- Preserves output structure for downstream processing
- Supports both local and S3 processing modes

## Conclusion

The debug mode implementation provides a powerful tool for rapid S3 I/O testing and debugging. By reducing preprocessing time from 10+ minutes to 1-2 minutes, it enables much faster iteration and problem-solving, making the development process significantly more efficient. 