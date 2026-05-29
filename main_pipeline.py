import warnings
warnings.filterwarnings(action="ignore")

from data.loader import load_from_db, get_connection
from data.validator import data_validation, print_validation_report
from features.engineer import add_features, prepare_Xy, time_split
from features.pipeline import FeaturePipeline
from models.train import train_all, save_models
from models.tune import tune_all, build_base_models
from models.evaluate import evaluate_all, plot_results, predict_tomorrow
from config import TEST_SIZE, RANDOM_STATE


def run_pipeline(tune=False):
    conn = get_connection()
    try:
        # ── Step 1: Load ──────────────────────────────────────────────────────
        print("========== Layer 1: Loading Data ==========")
        df = load_from_db(conn)
        df = df.copy()
        if df.empty:
            raise ValueError("No data in DB. Run ingest.py before running the pipeline.")
        print(f"Loaded {df.shape[0]} rows, {df.shape[1]} cols\n")

        # ── Step 2: Validate ──────────────────────────────────────────────────
        print("========== Layer 2: Validating Data ==========")
        results = data_validation(df)
        print_validation_report(results)

        errors = []
        if not results["ohlc_clean"]:
            errors.append(f"OHLC violations: {results['ohlc_violations']}")
        if results["has_nulls"]:
            errors.append(f"Nulls found: {results['missing_values']}")
        if results["duplicate_dates"] > 0:
            errors.append(f"{results['duplicate_dates']} duplicate dates")
        if errors:
            raise ValueError(" | ".join(errors))

        if results["suspicious_price_jumps"] > 0:
            print(
                f"WARNING: {results['suspicious_price_jumps']} suspicious jumps on "
                f"{results.get('suspicious_dates')}. Review manually.\n"
            )

        # ── Step 3: Feature Engineering ───────────────────────────────────────
        print("========== Layer 3: Feature Engineering ==========")
        obj = FeaturePipeline()
        X_train, X_test, y_train, y_test, df_feat = obj.full_run(df)
        print(f"Train: {len(X_train)} rows | Test: {len(X_test)} rows\n")
        # df_feat = add_features(df)
        # X, y, df_feat = prepare_Xy(df_feat)
        # X_train, X_test, y_train, y_test = time_split(X, y, test_size=TEST_SIZE)
        # print(f"Train: {len(X_train)} rows | Test: {len(X_test)} rows\n")

        # ── Baseline (always-UP accuracy) — used for comparison ───────────────
        baseline_acc = float(y_test.mean())
        print(f"Baseline (always predict UP): {baseline_acc:.4f}\n")

        # ── Step 4: Train or Tune ─────────────────────────────────────────────
        if tune:
            print("========== Layer 4: Tuning Models ==========")
            print("WARNING: This will take several minutes.\n")
            trained_models = tune_all(X_train, y_train)
        else:
            print("========== Layer 4: Training Models ==========")
            base_models = build_base_models()
            models = {name: model for name, (model, _) in base_models.items()}
            trained_models = train_all(models, X_train, y_train)

        # ── Step 5: Evaluate ──────────────────────────────────────────────────
        print("========== Layer 5: Evaluating Models ==========")
        eval_results = evaluate_all(trained_models, X_test, y_test)

        # ── Step 6: Plot ──────────────────────────────────────────────────────
        print("========== Layer 6: Plotting Results ==========")
        plot_results(eval_results, trained_models)

        # ── Step 7: Predict Tomorrow ──────────────────────────────────────────
        print("========== Layer 7: Tomorrow's Prediction ==========")
        predict_tomorrow(trained_models, df_feat)

        # ── Step 8: Save ──────────────────────────────────────────────────────
        save_models(trained_models)
        print("\nPipeline completed successfully.")

    except ValueError as e:
        # Validation / data errors — no DB rollback needed
        print(f"Pipeline stopped: {e}")

    except Exception as e:
        print(f"Pipeline crashed: {e}")
        # Only roll back if we're in the middle of a DB write
        try:
            conn.rollback()
        except Exception:
            pass
        raise

    finally:
        conn.close()
        print("Connection closed.")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--tune", action="store_true", help="Run hyperparameter tuning")
    args = parser.parse_args()
    run_pipeline(tune=args.tune)