import os
import shutil
import time
import numpy as np
import pandas as pd
from PIL import Image
import joblib

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.ensemble import BaggingClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

def main():
    print("=" * 60)
    print("COVID-19 Chest X-Ray Model Training Pipeline")
    print("=" * 60)

    image_folder = r"C:\Users\User\Downloads\covid 2"
    base_dir = os.path.dirname(os.path.abspath(__file__))
    output_model_path = os.path.join(base_dir, "pipeline.joblib")
    samples_dir = os.path.join(base_dir, "sample_images")
    os.makedirs(samples_dir, exist_ok=True)

    image_size = (64, 64)

    print(f"\n[1/6] Scanning image folder: {image_folder}...")
    valid_files = [
        f for f in os.listdir(image_folder)
        if f.lower().endswith(('.png', '.jpg', '.jpeg'))
    ]
    print(f"Found {len(valid_files)} images.")

    print("\n[2/6] Loading & preprocessing images (64x64 grayscale)...")
    start_load = time.time()
    data = []
    filenames = []
    labels = []

    covid_samples = []
    normal_samples = []

    for idx, filename in enumerate(valid_files):
        img_path = os.path.join(image_folder, filename)
        is_covid = 'covid' in filename.lower()
        label = 'covid' if is_covid else 'non_covid'

        # Collect sample images for the UI gallery (first 6 of each)
        if is_covid and len(covid_samples) < 6:
            dest = os.path.join(samples_dir, f"sample_covid_{len(covid_samples)+1}.png")
            shutil.copy2(img_path, dest)
            covid_samples.append({
                "label": "COVID-19",
                "filename": filename,
                "path": dest,
                "key": f"covid_{len(covid_samples)+1}"
            })
        elif not is_covid and len(normal_samples) < 6:
            dest = os.path.join(samples_dir, f"sample_normal_{len(normal_samples)+1}.png")
            shutil.copy2(img_path, dest)
            normal_samples.append({
                "label": "Normal (Non-COVID)",
                "filename": filename,
                "path": dest,
                "key": f"normal_{len(normal_samples)+1}"
            })

        try:
            with Image.open(img_path) as img:
                img_gray = img.convert('L').resize(image_size)
                pixel_array = np.array(img_gray, dtype=np.float32).flatten()
                data.append(pixel_array)
                filenames.append(filename)
                labels.append(label)
        except Exception as e:
            print(f"Error loading {filename}: {e}")

        if (idx + 1) % 1500 == 0 or (idx + 1) == len(valid_files):
            print(f"  Processed {idx + 1}/{len(valid_files)} images ({time.time() - start_load:.1f}s)")

    X_raw = np.array(data)
    y = np.array(labels)
    print(f"Data shape: {X_raw.shape}, Class counts: {dict(pd.Series(y).value_counts())}")

    print("\n[3/6] Fitting StandardScaler...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_raw)

    print("\n[4/6] Fitting PCA(n_components=100)...")
    pca = PCA(n_components=100, random_state=42)
    X_pca = pca.fit_transform(X_scaled)
    total_var = np.sum(pca.explained_variance_ratio_) * 100
    print(f"  Total variance retained by 100 components: {total_var:.2f}%")

    print("\n[5/6] Splitting train/test (80/20 stratified)...")
    X_train, X_test, y_train, y_test = train_test_split(
        X_pca, y, test_size=0.20, random_state=42, stratify=y
    )

    print("\n[6/6] Training BaggingClassifier with SVC(probability=True)...")
    start_train = time.time()
    bag_model = BaggingClassifier(
        estimator=SVC(probability=True, kernel='rbf', C=1.0, random_state=42),
        n_estimators=50,
        max_samples=0.5,
        bootstrap=True,
        random_state=42,
        verbose=1,
        n_jobs=-1
    )
    bag_model.fit(X_train, y_train)
    train_duration = time.time() - start_train
    print(f"  Model training finished in {train_duration:.2f}s")

    # Evaluate
    y_pred = bag_model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred, labels=['covid', 'non_covid'])
    report = classification_report(y_test, y_pred, output_dict=True)

    print(f"\nTest Accuracy: {accuracy * 100:.2f}%")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    print("\nConfusion Matrix [covid, non_covid]:")
    print(cm)

    # Subsample test projections for interactive scatter plot (300 points)
    indices = np.random.choice(len(X_test), size=min(300, len(X_test)), replace=False)
    test_pca_samples = {
        'pc1': X_test[indices, 0].tolist(),
        'pc2': X_test[indices, 1].tolist(),
        'pc3': X_test[indices, 2].tolist(),
        'true_label': y_test[indices].tolist(),
        'pred_label': y_pred[indices].tolist()
    }

    # Package and serialize
    pipeline_data = {
        'scaler': scaler,
        'pca': pca,
        'model': bag_model,
        'image_size': image_size,
        'classes': list(bag_model.classes_),
        'metrics': {
            'accuracy': float(accuracy),
            'confusion_matrix': cm.tolist(),
            'report': report,
            'explained_variance_ratio': pca.explained_variance_ratio_.tolist(),
            'total_variance': float(total_var)
        },
        'test_pca_samples': test_pca_samples,
        'sample_gallery': covid_samples + normal_samples
    }

    joblib.dump(pipeline_data, output_model_path)
    print(f"\nPipeline successfully saved to: {output_model_path}")
    print("=" * 60)

if __name__ == '__main__':
    main()
