import argparse
from train import train_model
from minicamels import MiniCamels


def summarize_data():
    ds = MiniCamels()
    basins = ds.basins()

    print("Number of basins:", len(basins))
    print("\nFirst 5 basins:")
    print(basins.head())

    basin_id = basins.iloc[0]["basin_id"]
    data = ds.open_basin(basin_id)

    print("\nSample basin:", basin_id)
    print(data)
    print("\nVariables:", list(data.data_vars))


def main():
    parser = argparse.ArgumentParser(description="Streamflow prediction project")

    subparsers = parser.add_subparsers(dest="command")

    subparsers.add_parser("summarize-data")
    subparsers.add_parser("train")

    args = parser.parse_args()

    if args.command == "summarize-data":
        summarize_data()

    elif args.command == "train":
        train_model()

    else:
        parser.print_help()


if __name__ == "__main__":
    main()