"""
Iris Flower Classification using SVM.

A production-grade Machine Learning pipeline that classifies iris flower
species based on petal and sepal measurements using a Support Vector Machine
(SVM) classifier wrapped in a scikit-learn Pipeline.

The pipeline follows a three-stage workflow:
    1. Training   — Load data, run cross-validation, evaluate on a holdout
                     test set, retrain on full data, and persist the model
                     artifact to disk.
    2. Prediction — Load a previously saved model artifact and classify new
                     flower measurements with a confidence score.
    3. Interactive — A terminal-based REPL that accepts comma-separated
                     measurements and returns species predictions.

Usage:
    # Train the model and save it to disk (exits after training)
    python Petal_Detection.py --train

    # Run a single prediction from the command line
    python Petal_Detection.py --predict 5.1 3.5 1.4 0.2

    # Launch the interactive predictor (auto-trains if no saved model found)
    python Petal_Detection.py

Dependencies:
    pandas, numpy, scikit-learn, joblib
"""

from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


# ---------------------------------------------------------------------------
# Logging Configuration
# ---------------------------------------------------------------------------
# Use Python's built-in logging module instead of print() for structured,
# level-aware output. This allows log routing to files, cloud sinks, or
# monitoring tools in production environments.
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Constants & Configuration
# ---------------------------------------------------------------------------

# The four morphological features measured for each iris flower sample.
# These names must match exactly what the CSV columns are assigned during
# data loading, and they are reused when constructing DataFrames for
# prediction to avoid sklearn's "X does not have valid feature names" warning.
FEATURE_NAMES: tuple[str, ...] = (
    "Sepal Length",   # Length of the outer protective leaf (cm)
    "Sepal Width",    # Width of the outer protective leaf (cm)
    "Petal Length",   # Length of the inner colored leaf (cm)
    "Petal Width",    # Width of the inner colored leaf (cm)
)

# The name of the target/label column in the dataset.
TARGET_COLUMN: str = "Class"

# Biologically plausible measurement bounds (in centimeters) derived from
# the Iris dataset's observed min/max values with a small margin. These
# prevent the model from silently accepting nonsensical inputs like negative
# petal widths or absurdly large sepal lengths.
FEATURE_BOUNDS: dict[str, tuple[float, float]] = {
    "Sepal Length": (3.0, 10.0),   # Dataset range: 4.3 - 7.9
    "Sepal Width":  (1.5, 5.0),    # Dataset range: 2.0 - 4.4
    "Petal Length": (0.5, 8.0),    # Dataset range: 1.0 - 6.9
    "Petal Width":  (0.05, 3.5),   # Dataset range: 0.1 - 2.5
}


@dataclass(frozen=True)
class ModelConfig:
    """Immutable configuration for the entire training and inference pipeline.

    Using a frozen dataclass centralizes all hyperparameters and file paths
    into a single, auditable object that cannot be accidentally mutated
    after creation. This makes experiment tracking and reproducibility
    straightforward.

    Attributes:
        data_path:    Path to the Iris CSV dataset (no header row).
        model_path:   Path where the trained Pipeline artifact is saved/loaded.
        test_size:    Fraction of data reserved for holdout evaluation (0.0-1.0).
        cv_folds:     Number of folds for stratified cross-validation.
        random_state: Seed for reproducibility across splits and SVM training.
        svm_kernel:   SVM kernel type ('linear', 'rbf', 'poly', etc.).
        svm_c:        SVM regularization parameter (higher = less regularization).
    """

    data_path: Path = field(
        default_factory=lambda: Path(__file__).parent / "iris_no_header.csv"
    )
    model_path: Path = field(
        default_factory=lambda: Path(__file__).parent / "iris_pipeline.joblib"
    )
    test_size: float = 0.2
    cv_folds: int = 5
    random_state: int = 42
    svm_kernel: str = "linear"
    svm_c: float = 1.0


# ---------------------------------------------------------------------------
# Data Loading & Exploration
# ---------------------------------------------------------------------------

def load_dataset(data_path: Path) -> tuple[pd.DataFrame, pd.Series]:
    """Load the Iris CSV dataset and split it into features and target.

    The CSV file is expected to have no header row and exactly 5 columns:
    four numeric feature columns followed by one string class label column.

    Args:
        data_path: Absolute or relative path to the headerless CSV file.

    Returns:
        A tuple of:
            - X (pd.DataFrame): Feature matrix with shape (n_samples, 4).
            - y (pd.Series): Target labels (e.g. 'Iris-setosa').

    Raises:
        FileNotFoundError: If the dataset file does not exist at *data_path*.
    """
    if not data_path.exists():
        raise FileNotFoundError(f"Dataset not found: {data_path}")

    # Assign meaningful column names since the CSV has no header row.
    columns = list(FEATURE_NAMES) + [TARGET_COLUMN]
    df = pd.read_csv(data_path, names=columns, header=None)

    logger.info("Loaded dataset: %d samples, %d features", len(df), len(FEATURE_NAMES))

    # Separate features (X) from target labels (y).
    # Note: We pass string labels directly to scikit-learn — no LabelEncoder
    # needed, since SVC natively handles string class labels and returns
    # them directly from predict().
    return df[list(FEATURE_NAMES)], df[TARGET_COLUMN]


def print_eda(X: pd.DataFrame, y: pd.Series) -> None:
    """Print basic Exploratory Data Analysis (EDA) statistics to the console.

    Outputs three summaries:
        1. Feature means  — average value of each measurement across all samples.
        2. Correlation matrix — Pearson correlation between feature pairs.
           (High correlation between Petal Length & Petal Width indicates
           they carry similar discriminative information.)
        3. Class distribution — number of samples per species to verify
           the dataset is balanced (50 per class in the standard Iris set).

    Args:
        X: Feature matrix DataFrame.
        y: Target label Series.
    """
    logger.info("--- Data Statistics ---")

    # Feature means give a quick sanity check that data loaded correctly.
    print("\nFeature Means:")
    print(X.mean().to_string())

    # Correlation matrix helps identify redundant features. In the Iris
    # dataset, Petal Length and Petal Width are highly correlated (~0.96),
    # while Sepal Width has weak negative correlation with petal features.
    print("\nCorrelation Matrix:")
    print(X.corr().to_string())

    # Class distribution confirms dataset balance. An imbalanced dataset
    # would require stratified sampling (which we do anyway) and possibly
    # class weights or oversampling techniques.
    print(f"\nClass Distribution:\n{y.value_counts().to_string()}\n")


# ---------------------------------------------------------------------------
# Model Training & Evaluation
# ---------------------------------------------------------------------------

def build_pipeline(config: ModelConfig) -> Pipeline:
    """Construct a leak-free preprocessing and classification pipeline.

    Why a Pipeline instead of separate scaler + model?
    --------------------------------------------------
    If StandardScaler is fitted on the entire dataset before cross-validation,
    the test fold's statistics (mean, std) leak into the scaler — this is
    called "data leakage" and artificially inflates evaluation metrics.

    By bundling the scaler inside a Pipeline, scikit-learn automatically
    fits the scaler ONLY on the training portion of each CV fold or
    train/test split, and transforms the validation/test portion using
    those training-only statistics.

    Pipeline steps:
        1. "scaler"     — StandardScaler normalizes features to zero mean and
                          unit variance: z = (x - mean) / std. This is critical
                          for SVM, which relies on Euclidean distances.
        2. "classifier" — Linear SVC with probability estimates enabled
                          (needed for predict_proba confidence scores).

    Args:
        config: ModelConfig with SVM hyperparameters.

    Returns:
        An unfitted sklearn Pipeline ready for training.
    """
    return Pipeline([
        # Step 1: Feature scaling (zero mean, unit variance)
        ("scaler", StandardScaler()),

        # Step 2: Support Vector Classifier
        ("classifier", SVC(
            kernel=config.svm_kernel,       # 'linear' kernel for this dataset
            C=config.svm_c,                 # Regularization: C=1.0 (default)
            probability=True,               # Enable predict_proba() for confidence
            random_state=config.random_state,  # Reproducible results
        )),
    ])


def train_and_evaluate(config: ModelConfig) -> Pipeline:
    """Execute the full training workflow and persist the model artifact.

    Workflow:
        1. Load the dataset and print EDA statistics.
        2. Run stratified k-fold cross-validation to estimate generalization
           accuracy without data leakage.
        3. Perform a holdout train/test split and print a detailed
           classification report (precision, recall, F1-score per class).
        4. Retrain the pipeline on the FULL dataset for maximum predictive
           power in production use.
        5. Serialize the trained pipeline to disk using joblib.

    Why retrain on full data after evaluation?
    ------------------------------------------
    Cross-validation and holdout evaluation give us confidence in the model's
    quality. Once validated, we retrain on 100% of available data so the
    production model benefits from every sample. The evaluation metrics from
    step 2-3 serve as the documented performance baseline.

    Args:
        config: ModelConfig containing all paths and hyperparameters.

    Returns:
        The fitted Pipeline (trained on the complete dataset).
    """
    # --- Step 1: Load data and display exploratory statistics ---
    X, y = load_dataset(config.data_path)
    print_eda(X, y)

    pipeline = build_pipeline(config)

    # --- Step 2: Stratified K-Fold Cross-Validation ---
    # StratifiedKFold preserves the class distribution (50/50/50) in each
    # fold, which is critical for reliable accuracy estimates — especially
    # important if the dataset were ever imbalanced.
    cv = StratifiedKFold(
        n_splits=config.cv_folds,           # 5 folds by default
        shuffle=True,                        # Shuffle before splitting
        random_state=config.random_state,    # Reproducible fold assignments
    )
    cv_scores = cross_val_score(pipeline, X, y, cv=cv, scoring="accuracy")
    logger.info(
        "%d-Fold CV Accuracy: %.2f%% (+/- %.2f%%)",
        config.cv_folds,
        cv_scores.mean() * 100,    # Mean accuracy across all folds
        cv_scores.std() * 100,     # Standard deviation (measures stability)
    )

    # --- Step 3: Holdout Test Set Evaluation ---
    # This provides an independent, single-split evaluation with a detailed
    # per-class breakdown (precision, recall, F1) that CV's single accuracy
    # number cannot provide.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=config.test_size,         # Reserve 20% for testing
        stratify=y,                          # Maintain class balance in split
        random_state=config.random_state,    # Reproducible split
    )
    pipeline.fit(X_train, y_train)           # Train on 80% of data

    y_pred = pipeline.predict(X_test)        # Predict on held-out 20%
    print("\n--- Holdout Test Set Report ---")
    print(classification_report(y_test, y_pred))

    # --- Step 4: Retrain on Full Dataset for Production ---
    # After confirming model quality above, we retrain on ALL data so the
    # deployed model has access to every available training sample.
    pipeline.fit(X, y)

    # --- Step 5: Persist the trained pipeline to disk ---
    # joblib is preferred over pickle for numpy-heavy objects (like SVM
    # support vectors) due to more efficient serialization of large arrays.
    joblib.dump(pipeline, config.model_path)
    logger.info("Model saved to: %s", config.model_path)

    return pipeline


# ---------------------------------------------------------------------------
# Prediction & Input Validation
# ---------------------------------------------------------------------------

def validate_features(values: Sequence[float]) -> None:
    """Validate that input features are within biologically plausible bounds.

    This guard prevents the model from silently returning predictions for
    physically impossible inputs (e.g., negative petal widths, 999cm sepals).
    While the SVM would still produce a mathematical prediction, such inputs
    are far outside the training distribution and the results would be
    meaningless.

    Args:
        values: Sequence of float measurements to validate.

    Raises:
        ValueError: If the count is not exactly 4, or any value falls
                    outside the plausible range defined in FEATURE_BOUNDS.
    """
    # Check that exactly 4 feature values were provided.
    if len(values) != len(FEATURE_NAMES):
        raise ValueError(f"Expected {len(FEATURE_NAMES)} values, got {len(values)}.")

    # Check each feature against its biological bounds.
    for name, val in zip(FEATURE_NAMES, values):
        lo, hi = FEATURE_BOUNDS[name]
        if not (lo <= val <= hi):
            raise ValueError(f"{name} = {val} is outside plausible range [{lo}, {hi}] cm.")


def predict_species(pipeline: Pipeline, features: Sequence[float]) -> tuple[str, float]:
    """Run a single prediction and return the species name with confidence.

    The function validates the input, constructs a properly-named DataFrame
    (to match the feature names the Pipeline was trained on), and returns
    both the predicted class label and the model's confidence probability.

    Args:
        pipeline: A fitted sklearn Pipeline (scaler + SVC).
        features: Exactly four float measurements in the order:
                  [Sepal Length, Sepal Width, Petal Length, Petal Width].

    Returns:
        A tuple of:
            - species (str): The predicted class label (e.g. 'Iris-setosa').
            - confidence (float): The highest class probability from
              predict_proba(), representing the model's certainty (0.0-1.0).
    """
    # Validate that inputs are sane before feeding to the model.
    validate_features(features)

    # Wrap the raw list in a DataFrame with proper column names so that
    # the Pipeline's StandardScaler recognizes the feature names it was
    # fitted on. Without this, sklearn emits a UserWarning.
    input_df = pd.DataFrame([features], columns=list(FEATURE_NAMES))

    # predict() returns the class label directly (e.g. 'Iris-setosa')
    # because we pass string labels to SVC — no LabelEncoder needed.
    species: str = pipeline.predict(input_df)[0]

    # predict_proba() returns a probability distribution over all classes.
    # We take the maximum probability as the confidence score.
    confidence: float = float(np.max(pipeline.predict_proba(input_df)))

    return species, confidence


# ---------------------------------------------------------------------------
# Interactive CLI
# ---------------------------------------------------------------------------

def run_interactive_predictor(pipeline: Pipeline) -> None:
    """Start an interactive prediction loop (REPL) in the terminal.

    The user is prompted to enter four comma-separated measurements.
    Each input is validated, scaled, and classified in real-time.
    The loop continues until the user types 'exit', 'quit', 'q',
    or sends an EOF / KeyboardInterrupt signal (Ctrl+C / Ctrl+D).

    Args:
        pipeline: A fitted sklearn Pipeline ready for inference.
    """
    print("\n--- Iris Flower Species Predictor ---")
    print("    Format: SepalLength, SepalWidth, PetalLength, PetalWidth")
    print("    Example: 5.1, 3.5, 1.4, 0.2")
    print("    Type 'exit' to quit.\n")

    while True:
        # Gracefully handle EOF (piped input ends) and Ctrl+C interrupts.
        try:
            raw = input("Enter values: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting...")
            break

        # Allow multiple exit keywords for convenience.
        if raw.lower() in ("exit", "quit", "q"):
            print("Exiting...")
            break

        try:
            # Parse comma-separated input into a list of floats.
            values = [float(v.strip()) for v in raw.split(",")]

            # Run prediction (includes validation + scaling + inference).
            species, confidence = predict_species(pipeline, values)
            print(f"  >> Prediction: {species}  (Confidence: {confidence * 100:.1f}%)\n")

        except ValueError as err:
            # Catches both float parsing errors and validation errors from
            # validate_features() — both are ValueErrors by design.
            print(f"  [Error] {err}\n")


# ---------------------------------------------------------------------------
# Entry Point
# ---------------------------------------------------------------------------

def main() -> None:
    """Parse command-line arguments and dispatch to the appropriate workflow.

    Execution modes:
        1. ``--train``             : Train the model, print evaluation metrics,
                                     save the artifact, and exit.
        2. ``--predict SL SW PL PW``: Load a saved model and run a single
                                     prediction from the command line.
        3. (no flags)              : Load a saved model (or auto-train if none
                                     exists) and launch the interactive REPL.

    The model artifact ('iris_pipeline.joblib') acts as the bridge between
    training and inference. Once trained, subsequent runs skip training
    entirely and load the persisted Pipeline instantly.
    """
    parser = argparse.ArgumentParser(
        description="Iris Flower SVM Classifier & Predictor",
    )
    parser.add_argument(
        "--train", action="store_true",
        help="Force retrain the model (even if a saved model exists).",
    )
    parser.add_argument(
        "--predict", nargs=4, type=float,
        metavar=("SL", "SW", "PL", "PW"),
        help="Run a single prediction with four feature values.",
    )
    args = parser.parse_args()

    # Create configuration with default values.
    config = ModelConfig()

    # Determine whether training is needed:
    #   - Explicitly requested via --train flag, OR
    #   - No saved model artifact exists on disk (first run).
    needs_training = args.train or not config.model_path.exists()
    if needs_training:
        pipeline = train_and_evaluate(config)
        if args.train and not args.predict:
            # When --train was explicitly requested (not auto-triggered),
            # exit after training completes. The user's intent was to train,
            # not to enter interactive mode.
            return
    else:
        # Load the previously saved Pipeline artifact from disk.
        # This is near-instant compared to retraining from scratch.
        pipeline = joblib.load(config.model_path)
        logger.info("Loaded saved model from: %s", config.model_path)

    # --- Single prediction mode ---
    # Activated when --predict is provided with 4 float values.
    if args.predict:
        species, confidence = predict_species(pipeline, args.predict)
        print(f"\nPrediction: {species}  (Confidence: {confidence * 100:.1f}%)")
        return

    # --- Interactive mode (default) ---
    # No flags provided: launch the terminal-based prediction REPL.
    run_interactive_predictor(pipeline)


# Guard: Only execute main() when this file is run directly as a script.
# When imported as a module (e.g., for testing), nothing executes automatically.
if __name__ == "__main__":
    main()
