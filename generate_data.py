from data.data_fetcher import generate_sample_light_curves

if __name__ == "__main__":
    output_dir = "./sample_data"
    generate_sample_light_curves(count=10, output_dir=output_dir)
