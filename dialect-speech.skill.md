# Skill: Dialect Speech Workflow

## Skill Name
`dialect-speech`

## Purpose

This skill provides quick access to common workflows for the dialect speech research project. It automates repetitive tasks and provides shortcuts for running experiments across different lessons.

---

## Available Commands

### 1. Setup Environment
```bash
/dialect-speech setup
```
- Creates conda environment
- Installs all dependencies
- Verifies MFA installation
- Configures VS Code settings

### 2. Run Lesson Experiment
```bash
/dialect-speech lesson <lesson_number>
```
- Extracts course materials
- Prepares data
- Runs training script
- Generates evaluation report

Example:
```bash
/dialect-speech lesson 4  # Run SVM vowel classifier
/dialect-speech lesson 6  # Run LSTM tone classifier
/dialect-speech lesson 7  # Run wav2vec IPA recognition
```

### 3. Extract Features
```bash
/dialect-speech extract-features --lesson <number> --type <feature_type>
```
- Extracts audio features using Praat/Parselmouth
- Supports: duration, pitch, formants, mfcc
- Saves to standardized CSV format

Example:
```bash
/dialect-speech extract-features --lesson 4 --type formants
/dialect-speech extract-features --lesson 6 --type pitch
```

### 4. Train Model
```bash
/dialect-speech train --model <model_type> --data <data_path>
```
- Trains specified model type
- Saves checkpoints automatically
- Logs to TensorBoard/Wandb

Supported models:
- `svm`: Support Vector Machine
- `lstm`: Long Short-Term Memory
- `wav2vec`: wav2vec 2.0 fine-tuning

Example:
```bash
/dialect-speech train --model svm --data data/lesson_4_features.csv
/dialect-speech train --model lstm --data data/lesson_6_pitch.csv
```

### 5. Evaluate Model
```bash
/dialect-speech evaluate --model <model_path> --test <test_data>
```
- Runs inference on test data
- Generates confusion matrix
- Creates t-SNE visualization
- Exports error analysis report

Example:
```bash
/dialect-speech evaluate --model checkpoints/svm_vowel.pkl --test data/test.csv
```

### 6. Run MFA Alignment
```bash
/dialect-speech mfa-align --audio <audio_dir> --text <text_dir> --output <output_dir>
```
- Runs Montreal Forced Aligner
- Uses pre-trained acoustic model
- Generates TextGrid files

Example:
```bash
/dialect-speech mfa-align --audio material/lesson_2/audio --text material/lesson_2/text --output results/textgrids
```

### 7. Generate Report
```bash
/dialect-speech report --lesson <number>
```
- Generates comprehensive experiment report
- Includes all metrics and visualizations
- Exports to PDF/HTML

Example:
```bash
/dialect-speech report --lesson 4
```

### 8. Clean Project
```bash
/dialect-speech clean [--all]
```
- Removes temporary files
- Cleans cache
- Optionally removes checkpoints and results

---

## Workflow Examples

### Complete Lesson 4 Workflow
```bash
# 1. Setup (first time only)
/dialect-speech setup

# 2. Extract features
/dialect-speech extract-features --lesson 4 --type formants

# 3. Train model
/dialect-speech train --model svm --data data/lesson_4_features.csv

# 4. Evaluate
/dialect-speech evaluate --model checkpoints/svm_vowel.pkl --test data/lesson_4_test.csv

# 5. Generate report
/dialect-speech report --lesson 4
```

### Complete Lesson 7 Workflow
```bash
# 1. Prepare wav2vec data
/dialect-speech extract-features --lesson 7 --type wav2vec

# 2. Fine-tune wav2vec
/dialect-speech train --model wav2vec --data data/lesson_7_wav2vec

# 3. Evaluate on test set
/dialect-speech evaluate --model checkpoints/wav2vec_ipa --test data/lesson_7_test

# 4. Generate report
/dialect-speech report --lesson 7
```

---

## Configuration

The skill reads configuration from `config.yaml`:

```yaml
# Project paths
data_dir: data/
checkpoint_dir: checkpoints/
result_dir: results/
log_dir: logs/

# Model defaults
svm:
  kernel: rbf
  C: 1.0
  gamma: auto

lstm:
  hidden_size: 128
  num_layers: 2
  dropout: 0.3
  learning_rate: 0.001

wav2vec:
  model_name: facebook/wav2vec2-base
  learning_rate: 1e-4
  freeze_feature_encoder: true

# Training defaults
batch_size: 32
epochs: 50
early_stopping_patience: 5

# Logging
use_tensorboard: true
use_wandb: false
log_level: INFO
```

---

## Implementation Notes

This skill should:
1. Check for required dependencies before running
2. Validate input paths and parameters
3. Use logging instead of print statements
4. Save all outputs to appropriate directories
5. Follow the project's commit and README protocols
6. Handle errors gracefully with informative messages

---

## Integration with Workflows

This skill respects the project's workflow protocols:
- Does NOT automatically commit changes
- Updates README.md when new experiments are run
- Excludes data files from git operations
- Maintains experiment logs in `logs/` directory

---

## Future Enhancements

Potential additions:
- Hyperparameter tuning automation
- Batch experiment runner
- Model comparison dashboard
- Data augmentation pipeline
- Automatic error analysis
- Cross-validation support
