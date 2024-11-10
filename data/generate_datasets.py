
from synthetic_data_generator import SyntheticDataGenerator


if __name__ == "__main__":

    # Data split: 70% - 15% - 15%

    # Generate base dataset
    generator_base = SyntheticDataGenerator(num_samples=2000, num_features=4, noise_level=10,
                                            datasets_dir="datasets/base_dataset")
    X, y = generator_base.generate_data()
    X_train, X_val, X_test, y_train, y_val, y_test = generator_base.split_data(X, y, training_size=0.7)
    generator_base.save_to_csv(X_train, X_val, X_test, y_train, y_val, y_test)

    # Generate larger dataset (double base dataset)
    generator_larger = SyntheticDataGenerator(num_samples=4000, num_features=8, noise_level=10,
                                              datasets_dir="datasets/larger_dataset")
    X, y = generator_larger.generate_data()
    X_train, X_val, X_test, y_train, y_val, y_test = generator_larger.split_data(X, y, training_size=0.7)
    generator_larger.save_to_csv(X_train, X_val, X_test, y_train, y_val, y_test)
    
    # Generate noisier dataset
    generator_noisy = SyntheticDataGenerator(num_samples=2000, num_features=4, noise_level=50,
                                             datasets_dir="datasets/noisy_dataset")
    X, y = generator_noisy.generate_data()
    X_train, X_val, X_test, y_train, y_val, y_test = generator_noisy.split_data(X, y, training_size=0.7)
    generator_noisy.save_to_csv(X_train, X_val, X_test, y_train, y_val, y_test)