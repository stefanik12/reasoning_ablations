from evaluations.counterfactual_transform import generate_dataset
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate counterfactual dataset")
    parser.add_argument("-n", "--n_samples_per_skill", type=int, help="Number of samples to generate per skill (note, duplicates will be discarded so actual samples per skill will be less than this)")
    parser.add_argument("--output_path", help="Path to save the generated dataset")
    
    args = parser.parse_args()

    generate_dataset(args.n_samples_per_skill,
                     args.output_path)